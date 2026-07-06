#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────
PROJECT_NAME="huggett-tree"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${REGION}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}"
VCPUS=16
MEMORY=32768   # MiB
MAX_VCPUS=32
OUTPUT_S3_BUCKET="${PROJECT_NAME}-results-${ACCOUNT_ID}"

echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"

# ─── 1. S3 bucket ──────────────────────────────────────────────────
echo "==> Creating S3 bucket (if needed)..."
if [ "${REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${OUTPUT_S3_BUCKET}" 2>/dev/null || true
else
    aws s3api create-bucket --bucket "${OUTPUT_S3_BUCKET}" \
        --create-bucket-configuration LocationConstraint="${REGION}" 2>/dev/null || true
fi

# ─── 2. ECR repository ─────────────────────────────────────────────
echo "==> Creating ECR repository..."
aws ecr create-repository --repository-name "${PROJECT_NAME}" --region "${REGION}" 2>/dev/null || true

echo "==> Building and pushing Docker image..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker build -t "${PROJECT_NAME}" "${SCRIPT_DIR}"
docker tag "${PROJECT_NAME}:latest" "${ECR_REPO}:latest"
docker push "${ECR_REPO}:latest"

# ─── 3. IAM roles ──────────────────────────────────────────────────
echo "==> Creating IAM roles..."

# Task execution role (ECR pull + CloudWatch)
EXEC_ROLE_NAME="${PROJECT_NAME}-batch-exec-role"
aws iam create-role \
    --role-name "${EXEC_ROLE_NAME}" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' 2>/dev/null || true

aws iam attach-role-policy \
    --role-name "${EXEC_ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy 2>/dev/null || true

EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EXEC_ROLE_NAME}"

# Task role (S3 write)
TASK_ROLE_NAME="${PROJECT_NAME}-batch-task-role"
aws iam create-role \
    --role-name "${TASK_ROLE_NAME}" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' 2>/dev/null || true

# Inline policy for S3 write
aws iam put-role-policy \
    --role-name "${TASK_ROLE_NAME}" \
    --policy-name "s3-write" \
    --policy-document "{
        \"Version\": \"2012-10-17\",
        \"Statement\": [{
            \"Effect\": \"Allow\",
            \"Action\": [\"s3:PutObject\", \"s3:GetObject\"],
            \"Resource\": \"arn:aws:s3:::${OUTPUT_S3_BUCKET}/*\"
        }]
    }"

TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}"

# Instance role (EC2 instances need this to join the ECS cluster)
INSTANCE_ROLE_NAME="${PROJECT_NAME}-ecs-instance-role"
aws iam create-role \
    --role-name "${INSTANCE_ROLE_NAME}" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' 2>/dev/null || true

aws iam attach-role-policy \
    --role-name "${INSTANCE_ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role 2>/dev/null || true

# Create instance profile and attach the role
aws iam create-instance-profile \
    --instance-profile-name "${INSTANCE_ROLE_NAME}" 2>/dev/null || true

aws iam add-role-to-instance-profile \
    --instance-profile-name "${INSTANCE_ROLE_NAME}" \
    --role-name "${INSTANCE_ROLE_NAME}" 2>/dev/null || true

INSTANCE_PROFILE_ARN="arn:aws:iam::${ACCOUNT_ID}:instance-profile/${INSTANCE_ROLE_NAME}"

# Spot fleet role
SPOT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/aws-service-role/spotfleet.amazonaws.com/AWSServiceRoleForEC2SpotFleet"
aws iam create-service-linked-role --aws-service-name spotfleet.amazonaws.com 2>/dev/null || true
aws iam create-service-linked-role --aws-service-name batch.amazonaws.com 2>/dev/null || true

# Wait for roles to propagate
echo "==> Waiting for IAM role propagation..."
sleep 10

# ─── 4. Compute environment (SPOT) ─────────────────────────────────
echo "==> Creating compute environment..."
COMP_ENV_NAME="${PROJECT_NAME}-spot-ce"

aws batch create-compute-environment \
    --compute-environment-name "${COMP_ENV_NAME}" \
    --type MANAGED \
    --compute-resources "{
        \"type\": \"SPOT\",
        \"allocationStrategy\": \"SPOT_CAPACITY_OPTIMIZED\",
        \"minvCpus\": 0,
        \"maxvCpus\": ${MAX_VCPUS},
        \"instanceTypes\": [\"optimal\"],
        \"instanceRole\": \"${INSTANCE_PROFILE_ARN}\",
        \"subnets\": $(aws ec2 describe-subnets --region ${REGION} --query 'Subnets[*].SubnetId' --output json),
        \"securityGroupIds\": $(aws ec2 describe-security-groups --region ${REGION} \
            --filters Name=group-name,Values=default \
            --query 'SecurityGroups[0:1].GroupId' --output json)
    }" 2>/dev/null || echo "  (compute environment may already exist)"

# Wait for compute environment to be VALID
echo "==> Waiting for compute environment to become VALID..."
for i in $(seq 1 30); do
    STATUS=$(aws batch describe-compute-environments \
        --compute-environments "${COMP_ENV_NAME}" \
        --query 'computeEnvironments[0].status' --output text 2>/dev/null || echo "CREATING")
    [ "${STATUS}" = "VALID" ] && break
    sleep 5
done

# ─── 5. Job queue ───────────────────────────────────────────────────
echo "==> Creating job queue..."
JOB_QUEUE_NAME="${PROJECT_NAME}-queue"

aws batch create-job-queue \
    --job-queue-name "${JOB_QUEUE_NAME}" \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=${COMP_ENV_NAME}" \
    2>/dev/null || echo "  (job queue may already exist)"

# ─── 6. Job definition ─────────────────────────────────────────────
echo "==> Registering job definition..."
JOB_DEF_NAME="${PROJECT_NAME}-job"

aws batch register-job-definition \
    --job-definition-name "${JOB_DEF_NAME}" \
    --type container \
    --container-properties "{
        \"image\": \"${ECR_REPO}:latest\",
        \"vcpus\": ${VCPUS},
        \"memory\": ${MEMORY},
        \"jobRoleArn\": \"${TASK_ROLE_ARN}\",
        \"executionRoleArn\": \"${EXEC_ROLE_ARN}\",
        \"environment\": [
            {\"name\": \"OUTPUT_S3_PATH\", \"value\": \"s3://${OUTPUT_S3_BUCKET}/\"}
        ],
        \"logConfiguration\": {
            \"logDriver\": \"awslogs\",
            \"options\": {
                \"awslogs-group\": \"/aws/batch/${PROJECT_NAME}\",
                \"awslogs-region\": \"${REGION}\",
                \"awslogs-stream-prefix\": \"job\"
            }
        }
    }"

# Create the CloudWatch log group
aws logs create-log-group --log-group-name "/aws/batch/${PROJECT_NAME}" --region "${REGION}" 2>/dev/null || true

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo "Submit a job with:"
echo "  aws batch submit-job \\"
echo "    --job-name ${PROJECT_NAME}-run \\"
echo "    --job-queue ${JOB_QUEUE_NAME} \\"
echo "    --job-definition ${JOB_DEF_NAME}"
echo ""
echo "Or use: bash submit-job.sh"
echo ""
echo "Results will be uploaded to: s3://${OUTPUT_S3_BUCKET}/"
echo "Logs: CloudWatch log group /aws/batch/${PROJECT_NAME}"

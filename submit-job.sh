#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="huggett-tree"
JOB_QUEUE="${PROJECT_NAME}-queue"
JOB_DEF="${PROJECT_NAME}-job"
JOB_NAME="${PROJECT_NAME}-$(date +%Y%m%d-%H%M%S)"

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${REGION}"

# Optional: override S3 output path
S3_PATH="${1:-}"

if [ -n "${S3_PATH}" ]; then
    OVERRIDES="{\"environment\": [{\"name\": \"OUTPUT_S3_PATH\", \"value\": \"${S3_PATH}\"}]}"
    echo "Submitting job with OUTPUT_S3_PATH=${S3_PATH}"
    aws batch submit-job \
        --job-name "${JOB_NAME}" \
        --job-queue "${JOB_QUEUE}" \
        --job-definition "${JOB_DEF}" \
        --container-overrides "${OVERRIDES}"
else
    echo "Submitting job with default S3 path"
    aws batch submit-job \
        --job-name "${JOB_NAME}" \
        --job-queue "${JOB_QUEUE}" \
        --job-definition "${JOB_DEF}"
fi

echo "Job submitted: ${JOB_NAME}"

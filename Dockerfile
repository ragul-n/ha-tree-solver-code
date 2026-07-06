FROM julia:1.11

RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli git && \
    rm -rf /var/lib/apt/lists/*

ARG REPO_URL=https://github.com/ragul-n/logs.git
ARG BRANCH=master

RUN git clone --branch ${BRANCH} --depth 1 ${REPO_URL} /app

WORKDIR /app/projects/ha-tree-solver/code

RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

ENTRYPOINT ["julia", "--project=.", "--threads=auto", "run.jl"]

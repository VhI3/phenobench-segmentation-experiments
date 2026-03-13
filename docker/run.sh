#!/usr/bin/env bash
# Be strict: stop on errors, catch missing vars, and fail if any piped command fails.
set -euo pipefail

# Name for the local Docker image we build and run.
IMAGE_NAME=phenobench-semseg

# Build the image from `docker/Dockerfile` and tag it so we can run it by name.
docker build -t ${IMAGE_NAME} -f docker/Dockerfile .

# Usage examples:
# ./docker/run.sh python -m src.train_semseg --data_root /workspace/PhenoBench --epochs 5
# ./docker/run.sh python -m src.eval_semseg --data_root /workspace/PhenoBench --ckpt runs/ckpt_last.pt

# Run the container:
# - clean it up when done (`--rm`)
# - keep an interactive terminal open (`-it`)
# - pass through all GPUs (`--gpus all`)
# - mount this repo at `/workspace` in the container
# - execute whatever command you pass to this script (`"$@"`)
docker run --rm -it \
  --gpus all \
  -v "$(pwd)":/workspace \
  ${IMAGE_NAME} \
  "$@"

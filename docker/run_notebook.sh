#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE=phenobench-semseg
NOTEBOOK_IMAGE=phenobench-semseg-notebook
HOST_PORT="${HOST_PORT:-8888}"
CONTAINER_PORT="${CONTAINER_PORT:-8888}"

docker build -t ${BASE_IMAGE} -f docker/Dockerfile .
docker build -t ${NOTEBOOK_IMAGE} -f docker/Dockerfile.notebook .

docker run --rm -it \
  --gpus all \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ${NOTEBOOK_IMAGE} \
  jupyter lab --ip=0.0.0.0 --port="${CONTAINER_PORT}" --no-browser --allow-root

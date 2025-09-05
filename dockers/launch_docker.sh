#! /bin/bash

CONTAINER_NAME=$1
if [[ -z "${CONTAINER_NAME}" ]]; then
  USERNAME="${HOME##*/}"
  CONTAINER_NAME=${USERNAME}_bench_dev
fi

echo "launch docker container with name: ${CONTAINER_NAME}"

DOCKER_IMAGE_NAME=$2
if [[ -z "${DOCKER_IMAGE_NAME}" ]]; then
  DOCKER_IMAGE_NAME="bench-rtx3080:latest"
fi

echo "using docker image: ${DOCKER_IMAGE_NAME}"

# Get Project Root
CURRENT_DIR="${PWD##*/}"
if [[ "$CURRENT_DIR" == *dockers* ]]; then
    PROJECT_DIR=$(dirname ${PWD})
else
    PROJECT_DIR=${PWD}
fi
echo "Project directory: ${PROJECT_DIR}"

# Get Data Dir
DATA_DIR="${HOME}/bench_data/"

# Launch
docker run -it --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ${DATA_DIR}:/data \
    -v ${PROJECT_DIR}:/app \
    --name ${CONTAINER_NAME} \
    -p 8000:8000 \
    --ipc=host \
    ${DOCKER_IMAGE_NAME} \
    bash

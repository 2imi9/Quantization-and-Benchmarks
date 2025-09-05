#!/bin/bash
FILE=$1
IMAGE_NAME=$2
if [[ -z "${IMAGE_NAME}" ]]; then
  IMAGE_NAME="bench-rtx3080:latest"
fi

echo "Building Docker image: ${FILE}"

docker build -t ${IMAGE_NAME} -f ${FILE} .

echo "Done - Docker image name: ${IMAGE_NAME}"

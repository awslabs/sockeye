#!/usr/bin/env bash

DOCKERFILE="sockeye_contrib/horovod/Dockerfile"

if [ ! -f ${DOCKERFILE} ]; then
  echo "This script should be run from the sockeye root directory (containing ${DOCKERFILE})."
  exit 2
fi

REPO="sockeye"
TAG=$(git rev-parse --short HEAD)
SOCKEYE_COMMIT=$(git rev-parse HEAD)

docker build -t ${REPO}:${TAG} -f $DOCKERFILE . --build-arg SOCKEYE_COMMIT=${SOCKEYE_COMMIT}

#!/usr/bin/env bash

DOCKERFILE="sockeye_contrib/horovod/Dockerfile"
REPO="sockeye"
TAG="2-horovod"

if [ ! -f ${DOCKERFILE} ]; then
  echo "This script should be run from the sockeye root directory (containing ${DOCKERFILE})."
  exit 2
fi

SOCKEYE_COMMIT=$(git rev-parse HEAD)

docker build -t ${REPO}:${TAG} -f $DOCKERFILE . --build-arg SOCKEYE_COMMIT=${SOCKEYE_COMMIT}

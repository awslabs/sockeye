#!/bin/bash

# Use CUDA stubs when running without nvidia-docker
command -v nvidia-smi >/dev/null 2>&1 || sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs

exec "$@"

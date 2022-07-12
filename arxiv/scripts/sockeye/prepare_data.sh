#!/usr/bin/env bash

sockeye-prepare-data \
  --source train.src.bpe.filter --target train.trg.bpe.filter --shared-vocab \
  --word-min-count 2 --pad-vocab-to-multiple-of 8 --max-seq-len 95 \
  --num-samples-per-shard 10000000 --output prepared --max-processes $(nproc)

#!/usr/bin/env bash

if [[ $# < 4 ]]; then
  echo "Usage: $0 <model> <gpu-or-cpu> <batch-size> <out-prefix> [shortlist]"
  echo "Ex: $0 sockeye.big gpu 64 sockeye.big.gpu.64"
  echo "Ex: $0 sockeye.big cpu 1 sockeye.big.cpu.1 sockeye.shortlist"
  exit 2
fi

MODEL=$1
DEVICE_ARGS="--dtype float16"
if [[ $2 == "cpu" ]]; then
  DEVICE_ARGS="--use-cpu"
fi
BATCH_SIZE=$3
OUT_PREFIX=$4

LEXICON_ARGS=""
if [[ $# > 4 ]]; then
  LEXICON_ARGS="--restrict-lexicon $5"
fi

(time -p sockeye-translate \
  --input test.src.bpe --output $OUT_PREFIX.bpe --models $MODEL --beam-size 5 \
  --batch-size $BATCH_SIZE --chunk-size 9999 $DEVICE_ARGS $LEXICON_ARGS) \
  2>&1 |tee $OUT_PREFIX.log

sed -re 's/@@( |$)//g' <$OUT_PREFIX.bpe >$OUT_PREFIX.tok

sacrebleu -tok none test.trg <$OUT_PREFIX.tok |tee $OUT_PREFIX.bleu

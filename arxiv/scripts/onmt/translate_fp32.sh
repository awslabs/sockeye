#!/usr/bin/env bash

CONFIG=$(dirname $0)/big.yaml

if [[ $# != 4 ]]; then
  echo "Usage: $0 <model> <gpu-or-cpu> <batch-size> <out-prefix>"
  echo "Ex: $0 fairseq.big gpu 64 fairseq.big.gpu.64"
  echo "Ex: $0 fairseq.big cpu 1 fairseq.big.cpu.1"
  exit 2
fi

MODEL=$1
DEVICE_ARGS="--gpu 0"
if [[ $2 == "cpu" ]]; then
  DEVICE_ARGS=""
fi
BATCH_SIZE=$3
OUT_PREFIX=$4

(time -p onmt_translate --src test.src.bpe --model $MODEL/big_average.pt \
  --beam_size 5 --batch_size $BATCH_SIZE $DEVICE_ARGS \
  --output $OUT_PREFIX.out) 2>&1 |tee $OUT_PREFIX.log

sed -re 's/@@( |$)//g' <$OUT_PREFIX.out >$OUT_PREFIX.tok

sacrebleu -tok none test.trg <$OUT_PREFIX.tok |tee $OUT_PREFIX.bleu


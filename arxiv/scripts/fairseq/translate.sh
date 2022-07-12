#!/usr/bin/env bash

if [[ $# != 4 ]]; then
  echo "Usage: $0 <model> <gpu-or-cpu> <batch-size> <out-prefix>"
  echo "Ex: $0 fairseq.big gpu 64 fairseq.big.gpu.64"
  echo "Ex: $0 fairseq.big cpu 1 fairseq.big.cpu.1"
  exit 2
fi

MODEL=$1
DEVICE_ARGS="--fp16"
if [[ $2 == "cpu" ]]; then
  DEVICE_ARGS="--cpu"
fi
BATCH_SIZE=$3
OUT_PREFIX=$4

cat test.src.bpe |(time -p fairseq-interactive data-bin/src_trg \
  --path $MODEL/checkpoint_best.pt --beam 5 --batch-size $BATCH_SIZE \
  --buffer-size 9999 --required-batch-size-multiple 1 $DEVICE_ARGS \
  >$OUT_PREFIX.out) 2>&1 |tee $OUT_PREFIX.log

grep '^H' $OUT_PREFIX.out |sed -e 's/^H-//g' |sort -g |cut -f3 \
  |sed -re 's/@@( |$)//g' >$OUT_PREFIX.tok

sacrebleu -tok none test.trg <$OUT_PREFIX.tok |tee $OUT_PREFIX.bleu

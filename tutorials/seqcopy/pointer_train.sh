#!/bin/bash

export PYTHONPATH=~/workspace/src/Sockeye

rm -rf model

python3 -m sockeye.train \
	--prepared-data prepared_data \
	-vs data/dev.source -vt data/dev.target \
  --encoder rnn --decoder rnn \
  --num-embed 32 --rnn-num-hidden 64 --rnn-attention-type dot --num-layers 1:1 \
  --use-cpu \
  --metrics perplexity accuracy \
  --max-num-checkpoint-not-improved 3 \
  -o model \
  --use-pointer-nets \
  --shared-vocab \
  --checkpoint-frequency 1000 \
	"$@"

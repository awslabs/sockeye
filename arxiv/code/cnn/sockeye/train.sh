#!/bin/bash

. env.sh

python -m sockeye.train
    -s $DATADIR/$PAIR/train.tok.bpe.$SOURCE \
    -t $DATADIR/$PAIR/train.tok.bpe.$TARGET \
    -vs $DATADIR/$PAIR/dev.bpe.$SOURCE \
    -vt $DATADIR/$PAIR/dev.bpe.$TARGET \
    -o model \
    --seed=1 \
    --batch-type word \
    --batch-size 4000 \
    --checkpoint-frequency 4000 \
    --device-ids -4 \
    --encoder cnn \
    --decoder cnn \
    --num-layers 8 \
    --num-embed 512 \
    --cnn-num-hidden 512 \
    --cnn-project-qkv \
    --cnn-kernel-width 3 \
    --cnn-hidden-dropout 0.2 \
    --cnn-positional-embedding-type learned \
    --max-seq-len 150 \
    --loss-normalization-type valid \
    --word-min-count 1 \
    --optimizer adam \
    --initial-learning-rate 0.0002 \
    --learning-rate-reduce-num-not-improved 8 \
    --learning-rate-reduce-factor 0.7 \
    --learning-rate-decay-param-reset \
    --max-num-checkpoint-not-improved 16 \
    --monitor-bleu 500 \
    --user-tensorboard


#!/bin/bash

. env.sh

PYTHONPATH=$SOCKEYE python3 -m sockeye.train \
    -s $DATADIR/$PAIR/train.tok.bpe.$SOURCE \
    -t $DATADIR/$PAIR/train.tok.bpe.$TARGET \
    -vs $DATADIR/$PAIR/dev.bpe.$SOURCE \
    -vt $DATADIR/$PAIR/dev.bpe.$TARGET \
    -o $SOURCE-$TARGET \
    --seed=1 --batch-type=word --batch-size=4096 --checkpoint-frequency=4000 --device-ids=-4 --embed-dropout=0.1:0.1 --rnn-decoder-hidden-dropout=0.2 --layer-normalization --num-layers=4:4 --max-seq-len=100:100 --label-smoothing 0.1 --weight-tying --weight-tying-type=src_trg --num-embed 512:512 --num-words 50000:50000 --word-min-count 1:1 --optimizer=adam  --initial-learning-rate=0.0002 --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 --learning-rate-scheduler-type=plateau-reduce --max-num-checkpoint-not-improved=32 --min-num-epochs=0 --rnn-attention-type mlp

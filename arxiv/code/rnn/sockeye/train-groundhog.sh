#!/bin/bash

. env.sh

PYTHONPATH=$SOCKEYE python3 -m sockeye.train \
    -s $DATADIR/$PAIR/train.tok.bpe.$SOURCE \
    -t $DATADIR/$PAIR/train.tok.bpe.$TARGET \
    -vs $DATADIR/$PAIR/dev.bpe.$SOURCE \
    -vt $DATADIR/$PAIR/dev.bpe.$TARGET \
    -o $SOURCE-$TARGET \
    --seed=1 --batch-type=sentence --batch-size=80 --bucket-width=10 --checkpoint-frequency=2000 --device-ids=1 --embed-dropout=0.3:0.3 --encoder=rnn --decoder=rnn --num-layers=1:1 --rnn-cell-type=lstm --rnn-num-hidden=1000 --rnn-residual-connections --layer-normalization --rnn-attention-type=mlp --rnn-attention-num-hidden=512 --rnn-attention-use-prev-word --rnn-attention-in-upper-layers --rnn-attention-coverage-num-hidden=1 --rnn-attention-coverage-type=count --rnn-decoder-state-init=zero --rnn-attention-use-prev-word --rnn-dropout-inputs=0:0 --rnn-dropout-states=0.0:0.0 --rnn-dropout-recurrent=0.0:0.0 --rnn-decoder-hidden-dropout=0.3 --fill-up=replicate --max-seq-len=50:50 --loss=cross-entropy --normalize-loss --num-embed 500:500 --num-words 50000:50000 --word-min-count 1:1 --optimizer=adam --optimized-metric=perplexity --clip-gradient=1.0 --initial-learning-rate=0.0002 --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 --learning-rate-scheduler-type=plateau-reduce --learning-rate-warmup=0 --max-num-checkpoint-not-improved=16 --min-num-epochs=1 --monitor-bleu=500 --keep-last-params=60 --lock-dir /var/lock --use-tensorboard

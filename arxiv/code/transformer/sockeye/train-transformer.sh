#!/bin/bash

. env.sh

PYTHONPATH=$SOCKEYE python3 -m sockeye.train \
    -s $DATADIR/$PAIR/train.tok.bpe.$SOURCE \
     -t $DATADIR/$PAIR/train.tok.bpe.$TARGET \
     -vs $DATADIR/$PAIR/dev.bpe.$SOURCE \
     -vt $DATADIR/$PAIR/dev.bpe.$TARGET \
     -o $SOURCE-$TARGET \
     --seed=1 --batch-type=word --batch-size=4096 --checkpoint-frequency=4000 --device-ids=-2 --embed-dropout=0:0 --encoder=transformer --decoder=transformer --num-layers=6:6 --transformer-model-size=512 --transformer-attention-heads=8 --transformer-feed-forward-num-hidden=2048 --transformer-preprocess=n --transformer-postprocess=dr --transformer-dropout-attention=0.1 --transformer-dropout-relu=0.1 --transformer-dropout-prepost=0.1 --transformer-positional-embedding-type fixed --fill-up=replicate --max-seq-len=100:100 --label-smoothing 0.1 --weight-tying --weight-tying-type=src_trg_softmax --num-embed 512:512 --num-words 50000:50000 --word-min-count 1:1 --optimizer=adam --optimized-metric=perplexity --clip-gradient=-1 --initial-learning-rate=0.0001 --learning-rate-reduce-num-not-improved=8 --learning-rate-reduce-factor=0.7 --learning-rate-scheduler-type=plateau-reduce --learning-rate-warmup=0 --max-num-checkpoint-not-improved=32 --min-num-epochs=0 --max-updates 1001000 \
     	--weight-init xavier --weight-init-scale 3.0 --weight-init-xavier-factor-type avg --use-tensorboard

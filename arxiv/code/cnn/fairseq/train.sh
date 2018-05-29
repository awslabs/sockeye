#!/bin/bash

. env.sh

pytorch_data_dir=bin_data_$PAIR

# prepare data
python $FAIRSEQ/preprocess.py \
       --source-lang $SOURCE \
       --target-lang $TARGET \
       --trainpref $DATADIR/$PAIR/train.tok.bpe \
       --validpref $DATADIR/$PAIR/dev.bpe \
       --testpref $DATADIR/$PAIR/test.bpe \
       --thresholdsrc 3 --thresholdtgt 3 \
       --destdir $pytorch_data_dir

# train
export CUDA_VISIBLE_DEVICES=0,1,2,3
python $FAIRSEQ/train.py \
       $pytorch_data_dir \
       --max-tokens 5000  --dropout 0.2 --force-anneal 100 \
       --workers 2 \
       --lr 1.25 --clip-norm 0.1 \
       --encoder-embed-dim 512 \
       --encoder-layers "[(512, 3)] * 8" \
       --decoder-embed-dim 512 \
       --decoder-layers "[(512, 3)] * 8"\
       --decoder-out-embed-dim 512 \
       --no-progress-bar \
       --save-dir model > model.log

# use parameters from when model has not improved for two epochs
python early_stop.py


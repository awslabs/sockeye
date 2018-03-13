#!/bin/bash

. env.sh

USRDIR=t2t_usr
DATADIR=data_$PAIR
TMPDIR=tmp_$PAIR
PROBLEM=wmt17_${SOURCE}_${TARGET}_bpe32k
HPARAMS=transformer_wmt17_base
MODEL=transformer.$PROBLEM.$HPARAMS

INPUTS=$DATDIR/$PAIR/test.bpe.$SOURCE
OUTPUTS=$(basename $INPUTS).transformer.$HPARAMS.$PROBLEM.beam$BEAM.alpha$ALPHA.decodes

t2t-trainer --t2t_usr_dir=$USRDIR --data_dir=$DATADIR --tmp_dir=$TMPDIR --problems=$PROBLEM \
    --model=transformer \
    --hparams_set=$HPARAMS \
    --train_steps=1000000 \
    --save_checkpoints_secs=1200 \
    --local_eval_frequency=0 \
    --schedule=train \
    --worker_gpu 2 \
    --output_dir=$MODEL 2> $MODEL.log

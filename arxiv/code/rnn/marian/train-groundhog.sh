#!/bin/bash

. env.sh

set -u

[[ ! -d model ]] && mkdir model
[[ ! -d validate ]] && mkdir validate

$MARIAN/build/marian \
    --model model/model.npz \
    --devices $DEVICES \
    --train-sets $DATADIR/$PAIR/train.tok.bpe.{$SOURCE,$TARGET} \
    --vocabs model/vocab.$SOURCE.yml model/vocab.$TARGET.yml \
    --dropout-rnn 0.3 --dim-emb 500 --dim-rnn 1000 --mini-batch 80 \
    --enc-depth 1 --enc-type bidirectional --dec-depth 1 \
    --enc-cell lstm --dec-cell lstm \
    --early-stopping 5 \
    --type s2s --valid-freq 10000 --save-freq 10000 --disp-freq 1000 \
    --valid-sets $DATADIR/$PAIR/dev.bpe.{$SOURCE,$TARGET} \
    --valid-metrics cross-entropy valid-script \
    --valid-script-path ../validate.sh \
    --valid-log model/valid.log --log model/train.log

#!/bin/bash

. env.sh

DEVICES="0 1 2 3"

[[ ! -d model ]] && mkdir model
[[ ! -d validate ]] && mkdir validate

$MARIAN/build/marian \
    --model model/model.npz \
    --devices $DEVICES \
    --train-sets $DATADIR/$PAIR/train.tok.bpe.{$SOURCE,$TARGET} \
    --vocabs model/vocab.$SOURCE.yml model/vocab.$TARGET.yml \
    --mini-batch-fit -w 10000 --sync-sgd \
    --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
    --exponential-smoothing \
    --early-stopping 5 \
    --tied-embeddings --skip --layer-normalization \
    --enc-depth 4 --enc-type alternating --enc-cell-depth 2 \
    --dec-depth 4 --dec-cell-base-depth 4 --dec-cell-high-depth 2 \
    --type s2s --valid-freq 10000 --save-freq 10000 --disp-freq 1000 \
    --valid-sets $DATADIR/$PAIR/dev.bpe.{$SOURCE,$TARGET} \
    --valid-metrics cross-entropy valid-script \
    --valid-script-path ../validate.sh \
    --valid-translation-output validate/dev.output.bpe --quiet-translation \
    --valid-log model/valid.log --log model/train.log

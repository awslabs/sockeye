#!/bin/bash

set -u

. env.sh

# model prefix
prefix=${1:-model/model.npz}

test=${2:-$DATDIR/$PAIR/test.bpe.$SOURCE}

[[ ! -d test ]] && mkdir test

echo "Decoding $test with $prefix..."

# decode
cat $test \
    | $MARIAN/build/marian-decoder \
          -m model/model.npz \
          -v model/vocab.{$SOURCE,$TARGET}.yml \
          -b 5 \
          -n \
          --devices $DEVICES \
          2> test/log \
    | sed -u 's/\@\@ //g' \
    | tee test/out \
    | $MOSES/scripts/tokenizer/detokenizer.perl -q -l $TARGET > test/out.detok

cat test/out.detok | sacrebleu -t wmt17 -l $PAIR | tee test/out.detok.bleu

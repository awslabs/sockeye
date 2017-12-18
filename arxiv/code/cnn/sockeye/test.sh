#!/bin/bash

. env.sh

test=${2:-$DATDIR/$PAIR/test.bpe.$SOURCE}

cat $test \
 | python -m sockeye.translate \
          -m model \
          --beam-size 5 \
          --batch-size 16
    | sed -u 's/\@\@ //g' | tee out \
    | $MOSES/scripts/tokenizer/detokenizer.perl -q -l $TARGET > out.detok

# get BLEU
cat out.detok | sacrebleu -t wmt17 -l $PAIR | tee out.detok.bleu


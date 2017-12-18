#!/bin/bash

. env.sh

ref=$DATADIR/$PAIR/dev.$TARGET

# decode

cat $1 \
    | sed 's/\@\@ //g' \
    | $MOSES/scripts/tokenizer/detokenizer.perl -l $TARGET -q 2> /dev/null \
    | tee validate/dev.output \
    | sacrebleu $ref -b

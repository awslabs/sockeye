#!/bin/bash

# Marian decodes internally and provides the name of the raw output
# file as an argument to this script.

. env.sh

REF=$DATADIR/$PAIR/dev.$TARGET

# decode

cat $1 \
    | sed 's/\@\@ //g' \
    | $MOSES/scripts/tokenizer/detokenizer.perl -l $TARGET -q 2> /dev/null \
    | tee validate/dev.output \
    | sacrebleu $REF -b

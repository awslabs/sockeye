#!/bin/bash

. env.sh

INPUT=$DATADIR/$PAIR/test.bpe.$SOURCE
OUTPUT=$PAIR.output
PYTHONPATH=$SOCKEYE python3 -m sockeye.translate \
    -m $PAIR -c 250 \
    --batch-size 16 --chunk-size 1000 \
    -i $INPUT -o $OUTPUT \
    --beam-size 5 --length-penalty-alpha 1.0 \
    2> $OUTPUT.log

cat $OUTPUT | \
    sed 's/\@\@ //g' | \
    $MOSES/scripts/tokenizer/detokenizer.perl -q -l $TARGET | \
    tee $OUTPUT.detok | \
    sacrebleu -t wmt17 -l $PAIR > $OUTPUT.bleu

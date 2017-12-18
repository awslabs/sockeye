#!/bin/bash

languagePair=${1:?First argument must be the language pair}
inputFile=${2:-/dev/stdin}

source=${languagePair/-*}
target=${languagePair/*-}

cat $inputFile \
    | sed -u 's/\@\@ //g' \
    | $MOSES/scripts/tokenizer/detokenizer.perl -q -l $target \
    | sacrebleu -t wmt17 -l $languagePair

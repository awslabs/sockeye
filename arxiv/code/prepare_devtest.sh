#!/bin/bash

. env.sh

if [[ -z $4 ]]; then
    echo "Usage: cat RAW_FILE | prepare.sh BPE_MODEL LANG PREFIX"
    echo "  where BPE_MODEL is the path to the joint BPE model"
    echo "        LANG is the ISO 639-1 two-character language code"
    echo "        PREFIX is an output prefix"
    echo "  Generates PREFIX.tok.LANG and PREFIX.bpe.LANG"
    exit
fi

bpe_model=$1
lang=$2
prefix=$3

$MOSES/scripts/tokenizer/normalize-punctuation.perl -l $lang \
    | $MOSES/scripts/tokenizer/remove-non-printing-char.perl \
    | $MOSES/scripts/tokenizer/tokenizer.perl -q -no-escape -protected $(dirname $0)/../tokenizer/basic-protected-patterns -l $lang \
    | tee $prefix.tok.$lang \
    | $BPE apply_bpe.py -c $bpe_model \
    > $prefix.bpe.$lang

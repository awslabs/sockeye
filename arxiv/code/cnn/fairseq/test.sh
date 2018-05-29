#!/bin/bash

. env.sh

# needed by fairseq to get the language pair
pytorch_data_dir=bin_data_$PAIR

test=${2:-$DATDIR/$PAIR/test.bpe.$SOURCE}

c=`cat early_stop.txt`

export CUDA_VISIBLE_DEVICES=0
    python $FAIRSEQ/generate.py \
             $pytorch_data_dir \
             --path model/checkpoint$c.pt \
             --batch-size 16 -i --beam 5 --gen-subset test \
    | sed -u 's/\@\@ //g' | tee test/out \
    | $MOSES/scripts/tokenizer/detokenizer.perl -q -l $TARGET > out.detok

# get BLEU
cat out.detok | sacrebleu -t wmt17 -l $PAIR | tee out.detok.bleu



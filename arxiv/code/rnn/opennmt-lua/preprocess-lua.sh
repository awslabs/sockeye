#!/bin/bash

. env.sh

th $OPENNMT/preprocess.lua \
   -train_src $DATADIR/$PAIR/train.tok.bpe.$SOURCE \
   -train_tgt $DATADIR/$PAIR/train.tok.bpe.$TARGET \
   -valid_src $DATADIR/$PAIR/dev.bpe.$SOURCE \
   -valid_tgt $DATADIR/$PAIR/dev.bpe.$TARGET \
   -save_data ende -src_words_min_frequency 1 \
   -tgt_words_min_frequency 1 -preprocess_pthreads 8 \
   -src_vocab $PAIR.dict -tgt_vocab $PAIR.dict

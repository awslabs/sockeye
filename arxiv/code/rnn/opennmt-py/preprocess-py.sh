#!/bin/bash

. env.sh

python $OPENNMT/preprocess.py \
    -train_src $DATADIR/$PAIR/train.tok.bpe.$SOURCE \
    -train_tgt $DATADIR/$PAIR/train.tok.bpe.$TARGET \
    -valid_src $DATADIR/$PAIR/dev.bpe.$SOURCE -valid_tgt $DATADIR/$PAIR/dev.bpe.$TARGET \
    -save_data ende -src_words_min_frequency 1 -tgt_words_min_frequency 1 \
    -share_vocab -src_vocab en-de.dict -tgt_vocab en-de.dict -tgt_seq_length_trunc 50 -src_seq_length_trunc 50

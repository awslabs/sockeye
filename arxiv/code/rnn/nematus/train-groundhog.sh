#!/bin/bash

. env.sh

# Parameters taken from WMT'17 system description
# https://arxiv.org/pdf/1708.00726.pdf
#
# Unsure about the "multiple GRU transitions", though. May be
# --enc_recurrence_transition_depth

echo "Starting at `date`" | tee log
#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda \
THEANO_FLAGS=floatX=float32 \
$NEMATUS/nematus/nmt.py
    --dataset $DATADIR/$PAIR/train.tok.bpe.{$SOURCE,$TARGET} \
    --dictionaries $PAIR.dict.json $PAIR.dict.json \
    --valid_dataset $DATADIR/dev.bpe.{$SOURCE,$TARGET} \
    --encoder lstm --decoder lstm_cond --decoder_deep lstm \
    --dim_word 500 \
    --dim 1000 \
    --enc_depth 1 \
    --dec_depth 1 \
    --dec_deep_context \
    --optimizer adam \
    --lrate 0.0001 \
    --batch_size 80 \
    --maxlen 80 \
    --tie_decoder_embeddings \
    --patience 10 \
    --validFreq 10000 \
    --layer_normalisation \
    --dropout_hidden 0.3 \
    --clip_c 1.0 \
    2>&1 | tee -a log
echo "Finished at `date`" | tee -a log

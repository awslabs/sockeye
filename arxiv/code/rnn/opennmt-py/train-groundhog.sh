#!/bin/bash

echo "Starting at `date`" | tee groundhog.log
$OPENNMT/train.py \
    -gpuid 0 \
    -data $PAIR \
    -save_model groundhog.model \
    -encoder_type rnn \
    -decoder_type rnn \
    -rnn_size 1000 \
    -word_vec_size 500 \
    -enc_layers 1 \
    -dec_layers 1 \
    -input_feed 0 \
    -optim adam \
    -learning_rate 0.001 \
    -batch_size 80 \
    -dropout 0.3 \
    -global_attention mlp \
    -brnn_merge concat \
    -max_grad_norm 1.0 \
2>&1 | tee -a groundhog.log
echo "Finished at `date`" | tee -a groundhog.log

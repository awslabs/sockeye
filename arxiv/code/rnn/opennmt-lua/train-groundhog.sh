#!/bin/bash

. env.sh

echo "Starting at `date`" | tee groundhog.log
THC_CACHING_ALLOCATOR=0 th $OPENNMT/train.lua \
    -gpuid 0 \
    -data $PAIR-train.t7 \
    -save_model groundhog.model \
    -encoder_type brnn \
    -src_seq_length 50 -tgt_seq_length 50 \
    -rnn_size 1000 \
    -word_vec_size 500 \
    -enc_layers 1 \
    -dec_layers 1 \
    -bridge none \
    -optim adam \
    -max_batch_size 80 \
    -input_feed 0 \
    -end_epoch 20 \
    -learning_rate 0.001 \
    -dropout 0.3 \
    -global_attention general \
    -brnn_merge concat \
    -max_grad_norm 1.0 \
2>&1 | tee -a groundhog.log
echo "Finished at `date`" | tee -a groundhog.log

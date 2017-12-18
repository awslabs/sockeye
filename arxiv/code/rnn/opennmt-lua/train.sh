#!/bin/bash

. env.sh

# dimensions of the hidden and embedding layers are from "Machine Translation at Booking.com: Journey and Lessons Learned"
# otherse are from "Toward a full-scale neural machine translation in production: the Booking.com use case"

echo "Starting at `date`" | tee booking.log
THC_CACHING_ALLOCATOR=0 th $OPENNMT/train.lua \
    -gpuid 0 \
    -data $PAIR-train.t7 \
    -save_model booking.model \
    -encoder_type brnn \
    -rnn_size 1000 \
    -word_vec_size 1000 \
    -enc_layers 4 \
    -dec_layers 4 \
    -input_feed 1 \
    -optim adam \
    -learning_rate 0.0002 \
    -end_epoch 20 \
    -max_batch_size 64 \
    -dropout 0.3 \
    -global_attention general \
    -brnn_merge concat \
    -max_grad_norm 1.0 \
2>&1 | tee -a booking.log
echo "Finished at `date`" | tee -a booking.log

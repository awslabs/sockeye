#!/usr/bin/env bash

if [[ $# != 1 ]]; then
  echo "Usage: $0 <max-updates>"
  echo "Max updates: 25000 (for wmt17_en_de), 70000 (for wmt17_ru_en)"
  exit 2
fi

MAX_UPDATES=$1

fairseq-train data-bin/src_trg --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 --dropout 0.1 --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 5000 --fp16 --update-freq 10 --max-update $MAX_UPDATES \
  --save-interval 9999 --save-interval-updates 500 --save-dir fairseq.big \
  --seed 1 2>&1 |tee fairseq.big.log

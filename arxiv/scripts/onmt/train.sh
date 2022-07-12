#!/usr/bin/env bash

CONFIG=$(dirname $0)/big.yaml

if [[ $# != 1 ]]; then
  echo "Usage: $0 <max-updates>"
  echo "Max updates: 25000 (for wmt17_en_de), 70000 (for wmt17_ru_en)"
  exit 2
fi

MAX_UPDATES=$1

mkdir -p onmt
onmt_build_vocab --config $CONFIG --n_sample -1 2>&1 |tee onmt/vocab.log
onmt_train --config $CONFIG --train_steps $MAX_UPDATES 2>&1 |tee onmt/train.log

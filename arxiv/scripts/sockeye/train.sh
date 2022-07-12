#!/usr/bin/env bash

if [[ $# != 2 ]]; then
  echo "Usage: $0 <model> <max-updates>"
  echo "Models: big, big_20_2, big_20_2_ssru"
  echo "Max updates: 25000 (for wmt17_en_de), 70000 (for wmt17_ru_en)"
  exit 2
fi

MODEL=$1
MAX_UPDATES=$2

case $MODEL in
  big)
    torchrun --no_python --nproc_per_node 8 sockeye-train \
      --prepared-data prepared --validation-source dev.src.bpe \
      --validation-target dev.trg.bpe --output sockeye.$MODEL --num-layers 6 \
      --transformer-model-size 1024 --transformer-attention-heads 16 \
      --transformer-feed-forward-num-hidden 4096 --apex-amp \
      --batch-type max-word --batch-size 5000 --update-interval 10 \
      --checkpoint-interval 500 --max-updates $MAX_UPDATES \
      --optimizer-betas 0.9:0.98 --dist --initial-learning-rate 0.06325 \
      --learning-rate-scheduler-type inv-sqrt-decay \
      --learning-rate-warmup 4000 --seed 1 --quiet-secondary-workers \
      --keep-last-params 8 --cache-last-best-params 8
    ;;
  big_20_2)
    torchrun --no_python --nproc_per_node 8 sockeye-train \
      --prepared-data prepared --validation-source dev.src.bpe \
      --validation-target dev.trg.bpe --output sockeye.$MODEL \
      --num-layers 20:2 --transformer-model-size 1024 \
      --transformer-attention-heads 16 \
      --transformer-feed-forward-num-hidden 4096 --apex-amp \
      --batch-type max-word --batch-size 3334 --update-interval 15 \
      --checkpoint-interval 500 --max-updates $MAX_UPDATES \
      --optimizer-betas 0.9:0.98 --dist --initial-learning-rate 0.06325 \
      --learning-rate-scheduler-type inv-sqrt-decay \
      --learning-rate-warmup 4000 --seed 1 --quiet-secondary-workers \
      --keep-last-params 8 --cache-last-best-params 8
    ;;
  big_20_2_ssru)
    torchrun --no_python --nproc_per_node 8 sockeye-train \
      --prepared-data prepared --validation-source dev.src.bpe \
      --validation-target dev.trg.bpe --output sockeye.$MODEL \
      --num-layers 20:2 --transformer-model-size 1024 \
      --transformer-attention-heads 16 \
      --transformer-feed-forward-num-hidden 4096 --decoder ssru_transformer \
      --apex-amp --batch-type max-word --batch-size 3334 --update-interval 15 \
      --checkpoint-interval 500 --max-updates $MAX_UPDATES \
      --optimizer-betas 0.9:0.98 --dist --initial-learning-rate 0.06325 \
      --learning-rate-scheduler-type inv-sqrt-decay \
      --learning-rate-warmup 4000 --seed 1 --quiet-secondary-workers \
      --keep-last-params 8 --cache-last-best-params 8
    ;;
esac

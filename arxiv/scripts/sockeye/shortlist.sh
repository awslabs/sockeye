#!/usr/bin/env bash

if [[ $# != 2 ]]; then
  echo "Usage: $0 <model> <fast-align-lex>"
  exit 2
fi

MODEL=$1
FAST_ALIGN_LEX=$2

sockeye-lexicon create --model $MODEL -k 200 --input $FAST_ALIGN_LEX \
  --output sockeye.shortlist

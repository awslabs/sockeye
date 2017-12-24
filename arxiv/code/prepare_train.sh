#!/bin/bash

set -eu

. env.sh

if [[ ! -d $MOSES ]]; then
    echo "Please set \$MOSES to point to your Moses installation."
    exit 1
fi

if [[ ! -d $BPE ]]; then
    echo "Please set \$BPE to point to your subword-nmt installation."
    exit 1
fi

for pair in en-de lv-en; do
    src=$(echo $pair | cut -d- -f1)
    tgt=$(echo $pair | cut -d- -f2)

    [[ ! -d data/$pair ]] && mkdir -p data/$pair

    # Tokenize, normalize
    for lang in $src $tgt; do
        for prefix in $(cat train.$pair.txt); do
	          cat $prefix.$lang
        done | $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $lang \
            | $MOSES/scripts/tokenizer/remove-non-printing-char.perl \
            | $MOSES/scripts/tokenizer/tokenizer.perl -q -no-escape -protected $(dirname $0)/../tokenizer/basic-protected-patterns -l $lang > data/$pair/train.tok.$lang &
    done

    wait

    cd data/$pair

    # Length filtering
    $MOSES/scripts/training/clean-corpus-n.perl train.tok $src $tgt train.tok.clean 1 100

    # BPE training
    cat train.tok.clean.{$src,$tgt} | $BPE/learn_bpe.py -s 32000 > bpe.model

    # Apply BPE
    for ext in $src $tgt; do
        cat train.tok.clean.$ext | $BPE/apply_bpe.py -c bpe.model > train.tok.bpe.$ext
    done

    cd ../..
done

# Download en-de dev and test sets
sacrebleu -t wmt15 -l en-de --echo src | ./prepare_devtest.sh data/en-de/bpe.model en data/en-de/dev
sacrebleu -t wmt17 -l en-de --echo src | ./prepare_devtest.sh data/en-de/bpe.model en data/en-de/test

# And lv-en
sacrebleu -t wmt17/dev -l lv-en --echo src | ./prepare_devtest.sh data/lv-en/bpe.model en data/lv-en/dev
sacrebleu -t wmt17 -l lv-en --echo src | ./prepare_devtest.sh data/lv-en/bpe.model en data/lv-en/test

#! /bin/bash

# Code taken from (modified):
# https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh
# MIT licensed. "Copyright (c) Facebook, Inc. and its affiliates. All rights reserved."

SRCS=(
    "de"
    "it"
)
TGT=en

ROOT=$(dirname "$0")/..

ORIG=$ROOT/iwslt17_orig
DATA=$ROOT/data
mkdir -p "$ORIG" "$DATA"

UNARCHIVED_NAME="DeEnItNlRo-DeEnItNlRo"

VALID_SETS=(
    "IWSLT17.TED.dev2010.de-en"
    "IWSLT17.TED.dev2010.it-en"
)

TEST_FILE="IWSLT17.TED.tst2010"

TEST_PAIRS=(
    "de en"
    "en de"
    "it en"
    "en it"
    "de it"
    "it de"
)

# download and extract data

gdown "https://drive.google.com/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp" -O - --quiet | \
    tar -C "$ORIG" -zxvf -

# un-tar for the second time

tar -C "$ORIG" -xzvf $ORIG/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        cat "$ORIG/$UNARCHIVED_NAME/train.tags.${SRC}-${TGT}.${LANG}" \
            | grep -v '<url>' \
            | grep -v '<talkid>' \
            | grep -v '<keywords>' \
            | grep -v '<speaker>' \
            | grep -v '<reviewer' \
            | grep -v '<translator' \
            | grep -v '<doc' \
            | grep -v '</doc>' \
            | sed -e 's/<title>//g' \
            | sed -e 's/<\/title>//g' \
            | sed -e 's/<description>//g' \
            | sed -e 's/<\/description>//g' \
            | sed 's/^\s*//g' \
            | sed 's/\s*$//g' \
            > "$DATA/train.${SRC}-${TGT}.${LANG}"
    done
done

echo "pre-processing valid data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=${VALID_SETS[i]}
    for FILE in "${VALID_SET[@]}"; do
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/$UNARCHIVED_NAME/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                >> "$DATA/valid.${SRC}-${TGT}.${LANG}"
        done
    done
done

echo "pre-processing test data..."

for TEST_PAIR in "${TEST_PAIRS[@]}"; do
    TEST_PAIR=($TEST_PAIR)
    SRC=${TEST_PAIR[0]}
    TGT=${TEST_PAIR[1]}
    for LANG in "$SRC" "$TGT"; do
        grep '<seg id' "$ORIG/$UNARCHIVED_NAME/${TEST_FILE}.${SRC}-${TGT}.${LANG}.xml" \
            | sed -e 's/<seg id="[0-9]*">\s*//g' \
            | sed -e 's/\s*<\/seg>\s*//g' \
            | sed -e "s/\’/\'/g" \
            > "$DATA/test.${SRC}-${TGT}.${LANG}"
    done
done

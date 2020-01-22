#! /bin/bash

# Code taken from (modified):
# https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh

SRCS=(
    "de"
    "fr"
)
TGT=en

ROOT=$(dirname "$0")/..

ORIG=$ROOT/iwslt17_orig
DATA=$ROOT/data
mkdir -p "$ORIG" "$DATA"

URLS=(
    "https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz"
    "https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz"
)
ARCHIVES=(
    "de-en.tgz"
    "fr-en.tgz"
)
VALID_SETS=(
    "IWSLT17.TED.dev2010.de-en IWSLT17.TED.tst2010.de-en IWSLT17.TED.tst2011.de-en IWSLT17.TED.tst2012.de-en"
    "IWSLT17.TED.dev2010.fr-en IWSLT17.TED.tst2010.fr-en IWSLT17.TED.tst2011.fr-en IWSLT17.TED.tst2012.fr-en"
)
TEST_SETS=(
    "IWSLT17.TED.tst2013.de-en IWSLT17.TED.tst2014.de-en IWSLT17.TED.tst2015.de-en"
    "IWSLT17.TED.tst2013.fr-en IWSLT17.TED.tst2014.fr-en IWSLT17.TED.tst2015.fr-en"
)

# download and extract data
for ((i=0;i<${#URLS[@]};++i)); do
    ARCHIVE=$ORIG/${ARCHIVES[i]}
    if [ -f "$ARCHIVE" ]; then
        echo "$ARCHIVE already exists, skipping download"
    else
        URL=${URLS[i]}
        wget -P "$ORIG" "$URL"
        if [ -f "$ARCHIVE" ]; then
            echo "$URL successfully downloaded."
        else
            echo "$URL not successfully downloaded."
            exit 1
        fi
    fi
    FILE=${ARCHIVE: -4}
    if [ -e "$FILE" ]; then
        echo "$FILE already exists, skipping extraction"
    else
        tar -C "$ORIG" -xzvf "$ARCHIVE"
    fi
done

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        cat "$ORIG/${SRC}-${TGT}/train.tags.${SRC}-${TGT}.${LANG}" \
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
    for FILE in ${VALID_SET[@]}; do
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/${SRC}-${TGT}/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                >> "$DATA/valid.${SRC}-${TGT}.${LANG}"
        done
    done
done

echo "pre-processing test data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    TEST_SET=${TEST_SETS[i]}
    for FILE in ${TEST_SET[@]}; do
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/${SRC}-${TGT}/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                >> "$DATA/test.${SRC}-${TGT}.${LANG}"
        done
    done
done

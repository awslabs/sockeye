# Multilingual Zero-shot Translation IWSLT 2017

In this tutorial we will train a multilingual Sockeye model that can translate between several language pairs,
including ones that we did not have training data for (this is called _zero-shot translation_).

Please note: this tutorial assumes that you are familiar with the introductory tutorials on [copying
sequences](https://awslabs.github.io/sockeye/tutorials/seqcopy.html)
and [training a standard WMT model](https://awslabs.github.io/sockeye/tutorials/wmt.html).

## Approach

There are several ways to train a multilingual translation system. This tutorial follows the approach described in [Johnson et al (2016)](https://arxiv.org/abs/1611.04558).

In a nutshell,

- We only change our _data_, but do not change the model architecture or training procedure at all.
- We need training data for several language pairs.
- For each pair of (source_sentence, target_sentence), such as:

```
Wieder@@ aufnahme der Sitzungs@@ periode
Re@@ sumption of the session
```

we prefix the source sentence with a special token to indicate the desired target language:

```
<2en> Wieder@@ aufnahme der Sitzungs@@ periode
```

(We do not change the target sentence at all.)

- Training batches are _mixed_: they always contain examples from all language pairs.

## Setup

Make sure to create a new Python virtual environment and activate it:

```bash
virtualenv -p python3 sockeye3
source sockeye3/bin/activate
```

Then [install the correct version of Sockeye](https://awslabs.github.io/sockeye/setup.html). Then we install several libraries for preprocessing,
monitoring and evaluation:

```bash
mkdir tools

pip install matplotlib mxboard

# install BPE library

pip install subword-nmt

# install sacrebleu for evaluation

pip install sacrebleu

# install Moses scripts for preprocessing

git clone https://github.com/bricksdont/moses-scripts $tools/moses-scripts
```


## Data

We will use data provided by the [IWSLT 2017 multilingual shared task](https://sites.google.com/site/iwsltevaluation2017/TED-tasks).

We will limit ourselves to using the training data of just 3 languages (DE, EN and FR), but in principle you could use all the training data.
Using only DE, EN and FR is inspired by the [Fairseq example for preparing IWSLT17 data](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh).

## Preprocessing

The preprocessing consists of the following steps:

- Extract raw texts from input files.
- Tokenize the text and split with a learned BPE model.
- Prefix the source sentences with a special target language indicator token.

```bash
# Code taken from (slightly modified):
# https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh

SRCS=(
    "de"
    "fr"
)
TGT=en

ROOT=$(dirname "$0")

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
    "IWSLT17.TED.dev2010.de-en IWSLT17.TED.tst2010.de-en IWSLT17.TED.tst2011.de-en IWSLT17.TED.tst2012.de-en IWSLT17.TED.tst2013.de-en IWSLT17.TED.tst2014.de-en IWSLT17.TED.tst2015.de-en"
    "IWSLT17.TED.dev2010.fr-en IWSLT17.TED.tst2010.fr-en IWSLT17.TED.tst2011.fr-en IWSLT17.TED.tst2012.fr-en IWSLT17.TED.tst2013.fr-en IWSLT17.TED.tst2014.fr-en IWSLT17.TED.tst2015.fr-en"
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
                > "$DATA/valid.${SRC}-${TGT}.${LANG}"
        done
    done
done
```

We then normalize and tokenize all texts:

```bash
MOSES=tools/moses-scripts/scripts

for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        for corpus in train valid; do
            cat "$DATA/train.${SRC}-${TGT}.${LANG}" | perl $MOSES/tokenizer/normalize-punctuation.perl | perl $MOSES/tokenizer/tokenizer.perl -a -q -l $LANG  > "$DATA/train.${SRC}-${TGT}.tok.${LANG}"
        done
    done
done
```

On tokenized text, we learn a BPE model as follows:

```bash
cat $DATA/train.*.tok.* > train.tmp

subword-nmt learn-joint-bpe-and-vocab -i train.tmp \
  --write-vocabulary bpe.vocab \
  --total-symbols --symbols 32000 -o bpe.codes

rm train.tmp
```

This will create a joint source and target BPE vocabulary.
Next, we use apply the Byte Pair Encoding to our training and development data:

```bash
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        for corpus in train valid; do
            python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab --vocabulary-threshold 50 < "$DATA/${corpus}.${SRC}-${TGT}.tok.${LANG}" > "$DATA/${corpus}.${SRC}-${TGT}.bpe.${LANG}"
        done
    done
done
```

Then we also need to prefix the source sentences with a special tag to indicate target language:

```bash
# link reverse directions

for corpus in train valid; do
    ln -s $DATA/$corpus.de-en.bpe.de $DATA/$corpus.en-de.bpe.de
    ln -s $DATA/$corpus.de-en.bpe.en $DATA/$corpus.en-de.bpe.en
    
    ln -s $DATA/$corpus.fr-en.bpe.fr $DATA/$corpus.en-fr.bpe.fr
    ln -s $DATA/$corpus.fr-en.bpe.en $DATA/$corpus.en-fr.bpe.en
done

# add tags to BPE versions of files

for corpus in train valid; do
    cat $DATA/$corpus.de-en.bpe.de | python add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.de-en.tag.de
    cat $DATA/$corpus.de-en.bpe.en | python add_tag_to_lines.py --tag "<2de>" > $DATA/$corpus.de-en.tag.en

    cat $DATA/$corpus.en-de.bpe.de | python add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.en-de.tag.de
    cat $DATA/$corpus.en-de.bpe.en | python add_tag_to_lines.py --tag "<2de>" > $DATA/$corpus.en-de.tag.en

    cat $DATA/$corpus.fr-en.bpe.fr | python add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.fr-en.tag.fr
    cat $DATA/$corpus.fr-en.bpe.en | python add_tag_to_lines.py --tag "<2fr>" > $DATA/$corpus.fr-en.tag.en

    cat $DATA/$corpus.en-fr.bpe.fr | python add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.en-fr.tag.fr
    cat $DATA/$corpus.en-fr.bpe.en | python add_tag_to_lines.py --tag "<2fr>" > $DATA/$corpus.en-fr.tag.en
done

```

Concatenate all individual files to obtain final training and development files:

```bash
for corpus in train valid; do
    touch $DATA/$corpus.tag.src
    touch $DATA/$corpus.tag.trg

    cat $DATA/$corpus.de-en.tag.de $DATA/$corpus.en-de.tag.en $DATA/$corpus.fr-en.tag.fr $DATA/$corpus.en-fr.tag.en > $DATA/$corpus.tag.src
    cat $DATA/$corpus.de-en.tag.en $DATA/$corpus.en-de.tag.de $DATA/$corpus.fr-en.tag.en $DATA/$corpus.en-fr.tag.fr > $DATA/$corpus.tag.trg
done
```

For our test data, we exploit the fact that IWSLT17 development data is multi-way parallel:

```bash
# link existing files

ln -s $DATA/valid.de-en.de $DATA/test.de-fr.de
ln -s $DATA/valid.fr-en.fr $DATA/test.de-fr.fr

ln -s $DATA/valid.de-en.de $DATA/test.fr-de.de
ln -s $DATA/valid.fr-en.fr $DATA/test.fr-de.fr

ln -s $DATA/valid.de-en.bpe.de $DATA/test.de-fr.bpe.de
ln -s $DATA/valid.fr-en.bpe.fr $DATA/test.de-fr.bpe.fr

ln -s $DATA/valid.fr-en.bpe.fr $DATA/test.fr-de.bpe.fr
ln -s $DATA/valid.de-en.bpe.de $DATA/test.fr-de.bpe.de

# add special target language tag

cat $DATA/test.de-fr.bpe.de | python add_tag_to_lines.py --tag "<2fr" > $DATA/test.de-fr.tag.de
cat $DATA/test.de-fr.bpe.fr | python add_tag_to_lines.py --tag "<2de" > $DATA/test.de-fr.tag.fr

cat $DATA/test.fr-de.bpe.de | python add_tag_to_lines.py --tag "<2fr" > $DATA/test.fr-de.tag.de
cat $DATA/test.fr-de.bpe.fr | python add_tag_to_lines.py --tag "<2de" > $DATA/test.fr-de.tag.fr
```

We link both the raw text and create a tagged version, the tagged file as input for translation, the raw text for evaluation,
to compute detokenized BLEU.

As a sanity check, compute number of lines in all files:

```bash
wc -l $DATA/*
```

Parallel files should still have the same number of lines.

## Training

Before we start training we will prepare the training data by splitting it into shards and serializing it in matrix format:
```bash
python -m sockeye.prepare_data \
                        -s $DATA/train.src \
                        -t $DATA/train.trg \
                        -o train_data
```

We can now kick off the training process:
```bash
python -m sockeye.train -d train_data \
                        -vs $DATA/valid.src \
                        -vt $DATA/valid.trg \
                        --encoder rnn \
                        --decoder rnn \
                        --num-embed 256 \
                        --rnn-num-hidden 512 \
                        --rnn-attention-type dot \
                        --max-seq-len 60 \
                        --decode-and-evaluate 500 \
                        --device-ids 0 \
                        -o iwslt_model
```

## Translation and Evaluation for Zero-Shot Directions

An interesting outcome of multilingual training is that a trained model is (to some extent) capable of translating between language pairs
that is has not seen training examples for.

To test the zero-shot condition, we translate from German to French and vice versa. Both pairs are unknown to the model.

```bash
mkdir -p translations

python -m sockeye.translate \
                        -i $DATA/test.de-fr.tag.de \
                        -o translations/test.de-fr.tag.fr \
                        -m iwslt_model \
                        --beam-size 10 \
                        --length-penalty-alpha 1.0 \
                        --device-ids 0 \
                        --batch-size 64

python -m sockeye.translate \
                        -i $DATA/test.fr-de.tag.fr \
                        -o translations/test.fr-de.tag.de \
                        -m iwslt_model \
                        --beam-size 10 \
                        --length-penalty-alpha 1.0 \
                        --device-ids 0 \
                        --batch-size 64
```

Next we post-process the translations, first removing the special target language tag, then removing BPE,
then detokenizing:

```bash
cat translations/test.de-fr.tag.fr | python remove_tag_from_translations.py --tag "<2en>" > translations/test.de-fr.bpe.fr
cat translations/test.de-fr.bpe.fr | sed -r 's/@@( |$)//g' > translations/test.de-fr.tok.fr
cat translations/test.de-fr.tok.fr | $MOSES/tokenizer/detokenizer.perl -l "fr" > translations/test.de-fr.fr

cat translations/test.fr-de.tag.de | python remove_tag_from_translations.py --tag "<2en>" > translations/test.fr-de.bpe.de
cat translations/test.fr-de.bpe.de | sed -r 's/@@( |$)//g' > translations/test.fr-de.tok.de
cat translations/test.fr-de.tok.de | $MOSES/tokenizer/detokenizer.perl -l "de" > translations/test.fr-de.de
```

Finally, we compute BLEU scores for both zero-shot directions with [sacreBLEU](https://github.com/mjpost/sacreBLEU):

```bash
cat translations/test.de-fr.fr | sacrebleu $DATA/test.de-fr.fr
cat translations/test.fr-de.de | sacrebleu $DATA/test.fr-de.de
```

## Summary

In this tutorial you trained a multilingual Sockeye model that can translate between several languages,
including zero-shot pairs that did not occur in the training data.

You now know how to modify the training
data to include special target language tags and how to translate and evaluate zero-shot directions.

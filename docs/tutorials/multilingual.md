# Multilingual Zero-shot Translation IWSLT 2017

In this tutorial we will train a multilingual Sockeye model that can translate between several language pairs,
including ones that we did not have training data for (this is called _zero-shot translation_).

Please note: this tutorial assumes that you are familiar with the introductory tutorials on
[copying sequences](seqcopy_tutorial.md)
and [training a standard WMT model](wmt.md).

## Approach

There are several ways to train a multilingual translation system. This tutorial follows the approach
described in [Johnson et al (2016)](https://arxiv.org/abs/1611.04558).

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

Then [install the correct version of Sockeye](../setup.md).
We also install several libraries for preprocessing, monitoring and evaluation:

```bash
pip install matplotlib tensorboard

# install BPE library

pip install subword-nmt

# install sacrebleu for evaluation

pip install sacrebleu

# install Moses scripts for preprocessing

mkdir -p tools

git clone https://github.com/bricksdont/moses-scripts tools/moses-scripts

# install library to download Google drive files

pip install gdown

# download helper scripts

wget https://raw.githubusercontent.com/awslabs/sockeye/main/docs/tutorials/multilingual/prepare-iwslt17-multilingual.sh -P tools
wget https://raw.githubusercontent.com/awslabs/sockeye/main/docs/tutorials/multilingual/add_tag_to_lines.py -P tools
wget https://raw.githubusercontent.com/awslabs/sockeye/main/docs/tutorials/multilingual/remove_tag_from_translations.py -P tools
```


## Data

We will use data provided by the [IWSLT 2017 multilingual shared task](https://sites.google.com/site/iwsltevaluation2017/TED-tasks).

We limit ourselves to using the training data of just 3 languages (DE, EN and IT), but in principle you could include many more
language pairs, for instance NL and RO which are also part of this IWSLT data set.

## Preprocessing

The preprocessing consists of the following steps:

- Extract raw texts from input files.
- Tokenize the text and split with a learned BPE model.
- Prefix the source sentences with a special target language indicator token.

Run the following script to obtain IWSLT17 data in a convenient format,
the code is adapted from the [Fairseq example for preparing IWSLT17 data](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh).

```bash
bash tools/prepare-iwslt17-multilingual.sh
```

After executing this script, all original files will be in `iwslt_orig` and extracted text files will be
in `data`.

```bash
MOSES=tools/moses-scripts/scripts
DATA=data

TRAIN_PAIRS=(
    "de en"
    "en de"
    "it en"
    "en it"
)

TRAIN_SOURCES=(
    "de"
     "it"
)

TEST_PAIRS=(
    "de en"
    "en de"
    "it en"
    "en it"
    "de it"
    "it de"
)
```

We first create symlinks for the reverse training directions, i.e. EN-DE and EN-IT:

```bash
for SRC in "${TRAIN_SOURCES[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        for corpus in train valid; do
            ln -s $corpus.${SRC}-${TGT}.${LANG} $DATA/$corpus.${TGT}-${SRC}.${LANG}
        done
    done
done
```

We then normalize and tokenize all texts:

```bash
for PAIR in "${TRAIN_PAIRS[@]}"; do
    PAIR=($PAIR)
    SRC=${PAIR[0]}
    TGT=${PAIR[1]}

    for LANG in "${SRC}" "${TGT}"; do
        for corpus in train valid; do
            cat "$DATA/${corpus}.${SRC}-${TGT}.${LANG}" | perl $MOSES/tokenizer/normalize-punctuation.perl | perl $MOSES/tokenizer/tokenizer.perl -a -q -l $LANG  > "$DATA/${corpus}.${SRC}-${TGT}.tok.${LANG}"
        done
    done
done

for PAIR in "${TEST_PAIRS[@]}"; do
    PAIR=($PAIR)
    SRC=${PAIR[0]}
    TGT=${PAIR[1]}

    for LANG in "${SRC}" "${TGT}"; do
        cat "$DATA/test.${SRC}-${TGT}.${LANG}" | perl $MOSES/tokenizer/normalize-punctuation.perl | perl $MOSES/tokenizer/tokenizer.perl -a -q -l $LANG  > "$DATA/test.${SRC}-${TGT}.tok.${LANG}"
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
Next, we apply the Byte Pair Encoding to our training and development data:

```bash
for PAIR in "${TRAIN_PAIRS[@]}"; do
    PAIR=($PAIR)
    SRC=${PAIR[0]}
    TGT=${PAIR[1]}

    for LANG in "${SRC}" "${TGT}"; do
        for corpus in train valid; do
            subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab --vocabulary-threshold 50 < "$DATA/${corpus}.${SRC}-${TGT}.tok.${LANG}" > "$DATA/${corpus}.${SRC}-${TGT}.bpe.${LANG}"
        done
    done
done

for PAIR in "${TEST_PAIRS[@]}"; do
    PAIR=($PAIR)
    SRC=${PAIR[0]}
    TGT=${PAIR[1]}

    for LANG in "${SRC}" "${TGT}"; do
        subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab --vocabulary-threshold 50 < "$DATA/test.${SRC}-${TGT}.tok.${LANG}" > "$DATA/test.${SRC}-${TGT}.bpe.${LANG}"
    done
done
```

We also need to prefix the source sentences with a special tag to indicate target language:

```bash
for PAIR in "${TRAIN_PAIRS[@]}"; do
    PAIR=($PAIR)
    SRC=${PAIR[0]}
    TGT=${PAIR[1]}

    for corpus in train valid; do
        cat $DATA/$corpus.${SRC}-${TGT}.bpe.${SRC} | python tools/add_tag_to_lines.py --tag "<2${TGT}>" > $DATA/$corpus.${SRC}-${TGT}.tag.${SRC}
        cat $DATA/$corpus.${SRC}-${TGT}.bpe.${TGT} | python tools/add_tag_to_lines.py --tag "<2${SRC}>" > $DATA/$corpus.${SRC}-${TGT}.tag.${TGT}
    done
done

for PAIR in "${TEST_PAIRS[@]}"; do
    PAIR=($PAIR)
    SRC=${PAIR[0]}
    TGT=${PAIR[1]}

     cat $DATA/test.${SRC}-${TGT}.bpe.${SRC} | python tools/add_tag_to_lines.py --tag "<2${TGT}>" > $DATA/test.${SRC}-${TGT}.tag.${SRC}
     cat $DATA/test.${SRC}-${TGT}.bpe.${TGT} | python tools/add_tag_to_lines.py --tag "<2${SRC}>" > $DATA/test.${SRC}-${TGT}.tag.${TGT}
done
```

Concatenate all individual files to obtain final training and development files:

```bash
for corpus in train valid; do
    touch $DATA/$corpus.tag.src
    touch $DATA/$corpus.tag.trg

    # be specific here, to be safe

    cat $DATA/$corpus.de-en.tag.de $DATA/$corpus.en-de.tag.en $DATA/$corpus.it-en.tag.it $DATA/$corpus.en-it.tag.en > $DATA/$corpus.tag.src
    cat $DATA/$corpus.de-en.tag.en $DATA/$corpus.en-de.tag.de $DATA/$corpus.it-en.tag.en $DATA/$corpus.en-it.tag.it > $DATA/$corpus.tag.trg
done
```

As our test data, we need both the raw text and the preprocessed, tagged version: the tagged file as input for translation, the raw text for evaluation,
to compute detokenized BLEU.

As a sanity check, compute number of lines in all files:

```bash
wc -l $DATA/*
```

Sanity checks to perform at this point:
- Parallel files should still have the same number of lines.
- Most file endings indicate a language, language suffixes should be correct.
- Importantly, corresponding lines in the preprocessed training and validation files should be parallel.

## Training

Before we start training we will prepare the training data by splitting it into shards and serializing it in matrix format:
```bash
python -m sockeye.prepare_data \
                        -s $DATA/train.tag.src \
                        -t $DATA/train.tag.trg \
                        -o train_data \
                        --shared-vocab
```

We can now kick off the training process:
```bash
python -m sockeye.train -d train_data \
                        -vs $DATA/valid.tag.src \
                        -vt $DATA/valid.tag.trg \
                        --shared-vocab \
                        --weight-tying-type src_trg_softmax \
                        --device-ids 0 \
                        --decode-and-evaluate-device-id 0 \
                        -o iwslt_model
```

## Translation and Evaluation including Zero-Shot Directions

An interesting outcome of multilingual training is that a trained model is (to some extent) capable of translating between language pairs
that is has not seen training examples for.

To test the zero-shot condition, we translate not only the trained directions, but also
from German to Italian and vice versa. Both of those pairs are unknown to the model.

Let's first try this for a single sentence in German. Remember to preprocess input text in exactly the same way as the
training data.

```bash
echo "Was für ein schöner Tag!" | \
    perl $MOSES/tokenizer/normalize-punctuation.perl | \
    perl $MOSES/tokenizer/tokenizer.perl -a -q -l de | \
    subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab --vocabulary-threshold 50 | \
    python tools/add_tag_to_lines.py --tag "<2it>" | \
    python -m sockeye.translate \
                            -m iwslt_model \
                            --beam-size 10 \
                            --length-penalty-alpha 1.0 \
                            --device-ids 1
```

If you trained your model for at least several hours, the output should be similar to:

```bash
<2en> Era un bel giorno !
```

Which is a reasonable enough translation! Note that a well-trained model always generates a special language tag as the first token.
In this case it's `<2en>` since Italian data was always paired with English data in our training set.

Now let's translate all of our test sets to evaluate performance in all translation directions:

```bash
mkdir -p translations

for TEST_PAIR in "${TEST_PAIRS[@]}"; do
    TEST_PAIR=($TEST_PAIR)
    SRC=${TEST_PAIR[0]}
    TGT=${TEST_PAIR[1]}

    python -m sockeye.translate \
                            -i $DATA/test.${SRC}-${TGT}.tag.${SRC} \
                            -o translations/test.${SRC}-${TGT}.tag.${TGT} \
                            -m iwslt_model \
                            --beam-size 10 \
                            --length-penalty-alpha 1.0 \
                            --device-ids 0 \
                            --batch-size 64
done
```

Next we post-process the translations, first removing the special target language tag, then removing BPE,
then detokenizing:

```bash

for TEST_PAIR in "${TEST_PAIRS[@]}"; do
    TEST_PAIR=($TEST_PAIR)
    SRC=${TEST_PAIR[0]}
    TGT=${TEST_PAIR[1]}

    # remove target language tag

    cat translations/test.${SRC}-${TGT}.tag.${TGT} | \
        python tools/remove_tag_from_translations.py --verbose \
        > translations/test.${SRC}-${TGT}.bpe.${TGT}

    # remove BPE encoding

    cat translations/test.${SRC}-${TGT}.bpe.${TGT} | sed -r 's/@@( |$)//g' > translations/test.${SRC}-${TGT}.tok.${TGT}

    # remove tokenization

    cat translations/test.${SRC}-${TGT}.tok.${TGT} | $MOSES/tokenizer/detokenizer.perl -l "${TGT}" > translations/test.${SRC}-${TGT}.${TGT}
done
```

Finally, we compute BLEU scores for both zero-shot directions with [sacreBLEU](https://github.com/mjpost/sacreBLEU):

```bash
for TEST_PAIR in "${TEST_PAIRS[@]}"; do
    TEST_PAIR=($TEST_PAIR)
    SRC=${TEST_PAIR[0]}
    TGT=${TEST_PAIR[1]}

    echo "translations/test.${SRC}-${TGT}.${TGT}"
    cat translations/test.${SRC}-${TGT}.${TGT} | sacrebleu $DATA/test.${SRC}-${TGT}.${TGT}
done
```

## Summary

In this tutorial you trained a multilingual Sockeye model that can translate between several languages,
including zero-shot pairs that did not occur in the training data.

You now know how to modify the training
data to include special target language tags and how to translate and evaluate zero-shot directions.

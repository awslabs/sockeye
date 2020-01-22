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
pip install matplotlib mxboard

# install BPE library

pip install subword-nmt

# install sacrebleu for evaluation

pip install sacrebleu

# install Moses scripts for preprocessing

mkdir -p tools

git clone https://github.com/bricksdont/moses-scripts $tools/moses-scripts

# download helper scripts

wget https://raw.githubusercontent.com/ZurichNLP/sockeye/multilingual-tutorial/docs/tutorials/multilingual/prepare-iwslt17-multilingual.sh -P tools
wget https://raw.githubusercontent.com/ZurichNLP/sockeye/multilingual-tutorial/docs/tutorials/multilingual/add_tag_to_lines.py -P tools
wget https://raw.githubusercontent.com/ZurichNLP/sockeye/multilingual-tutorial/docs/tutorials/multilingual/remove_tag_from_translations.py -P tools
```


## Data

We will use data provided by the [IWSLT 2017 multilingual shared task](https://sites.google.com/site/iwsltevaluation2017/TED-tasks).

We limit ourselves to using the training data of just 3 languages (DE, EN and FR), but in principle you could include many more
language pairs.

## Preprocessing

The preprocessing consists of the following steps:

- Extract raw texts from input files.
- Tokenize the text and split with a learned BPE model.
- Prefix the source sentences with a special target language indicator token.

Run the following script to obtain IWSLT17 data in a convenient format,
the code is taken from the [Fairseq example for preparing IWSLT17 data](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh)
and adapted slightly.

```bash
./tools/prepare-iwslt17-multilingual.sh
```

After executing this script, all original files will be in `iwslt_orig` and extracted text files will be
in `data`.

We then normalize and tokenize all texts:

```bash
MOSES=tools/moses-scripts/scripts
DATA=data

for SRC in de fr; do
    for LANG in "${SRC}" en; do
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
for SRC in de fr; do
    for LANG in "${SRC}" en; do
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
    cat $DATA/$corpus.de-en.bpe.de | python tools/add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.de-en.tag.de
    cat $DATA/$corpus.de-en.bpe.en | python tools/add_tag_to_lines.py --tag "<2de>" > $DATA/$corpus.de-en.tag.en

    cat $DATA/$corpus.en-de.bpe.de | python tools/add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.en-de.tag.de
    cat $DATA/$corpus.en-de.bpe.en | python tools/add_tag_to_lines.py --tag "<2de>" > $DATA/$corpus.en-de.tag.en

    cat $DATA/$corpus.fr-en.bpe.fr | python tools/add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.fr-en.tag.fr
    cat $DATA/$corpus.fr-en.bpe.en | python tools/add_tag_to_lines.py --tag "<2fr>" > $DATA/$corpus.fr-en.tag.en

    cat $DATA/$corpus.en-fr.bpe.fr | python tools/add_tag_to_lines.py --tag "<2en>" > $DATA/$corpus.en-fr.tag.fr
    cat $DATA/$corpus.en-fr.bpe.en | python tools/add_tag_to_lines.py --tag "<2fr>" > $DATA/$corpus.en-fr.tag.en
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

cat $DATA/test.de-fr.bpe.de | python tools/add_tag_to_lines.py --tag "<2fr" > $DATA/test.de-fr.tag.de
cat $DATA/test.de-fr.bpe.fr | python tools/add_tag_to_lines.py --tag "<2de" > $DATA/test.de-fr.tag.fr

cat $DATA/test.fr-de.bpe.de | python tools/add_tag_to_lines.py --tag "<2fr" > $DATA/test.fr-de.tag.de
cat $DATA/test.fr-de.bpe.fr | python tools/add_tag_to_lines.py --tag "<2de" > $DATA/test.fr-de.tag.fr
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
cat translations/test.de-fr.tag.fr | python tools/remove_tag_from_translations.py --tag "<2en>" > translations/test.de-fr.bpe.fr
cat translations/test.de-fr.bpe.fr | sed -r 's/@@( |$)//g' > translations/test.de-fr.tok.fr
cat translations/test.de-fr.tok.fr | $MOSES/tokenizer/detokenizer.perl -l "fr" > translations/test.de-fr.fr

cat translations/test.fr-de.tag.de | python tools/remove_tag_from_translations.py --tag "<2en>" > translations/test.fr-de.bpe.de
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

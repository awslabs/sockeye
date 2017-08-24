# WMT German to English news translation

In this tutorial we will train a German to English Sockeye model on a dataset from the
[Conference on Machine Translation (WMT) 2017](http://www.statmt.org/wmt17/).

## Setup

Sockeye expects already tokenized data as the input.
For this tutorial we use data that has already been tokenized for us.
However, keep this in mind for any other data set you want to use with Sockeye.
In addition to tokenization we will splits words into subwords using Byte Pair Encoding (BPE).
In order to do so we use a tool called [subword-nmt](https://github.com/rsennrich/subword-nmt).
Run the following commands to set the tool up:

```bash
git clone https://github.com/rsennrich/subword-nmt.git
export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
```

For visualizating alignments we will need `matplotlib`.
If you haven't installed the library yet you can do so by running:
```bash
pip install matplotlib
```

We will visualize training progress using `tensorboard`. Install it using:
```bash
pip install tensorboard
```

## Data

We will use the data provided by the WMT 2017 news translation shared task.
Download the data using the following commands:

```bash
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz
gunzip corpus.tc.de.gz
gunzip corpus.tc.en.gz
curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf - 
```

## Preprocessing

The data has already been tokenized. Additionally, we will split words into subwords.
First we need to build our BPE vocabulary:
```bash
python -m learn_joint_bpe_and_vocab --input corpus.tc.de corpus.tc.en \
                                    -s 30000 \
                                    -o bpe.codes \
                                    --write-vocabulary bpe.vocab.de bpe.vocab.en

```

This will create a joint source and target BPE vocabulary.
Next, we use apply the Byte Pair Encoding to our training and development data:

```bash
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < corpus.tc.de > corpus.tc.BPE.de
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < corpus.tc.en > corpus.tc.BPE.en

python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.tc.BPE.de
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.tc.BPE.en
```

## Training

Parameters to discuss:
* early stopping (also: how to change the metric used for early stopping)
* batch size
* Optimizers: adam etc
* changing the optimized metric to BLEU

### Model variations
* Discuss the different model options provided

### Monitoring training with tensorboard

* show how we can use tensorboard for tracking progress
* add a screenshot of how this looks like


### Checkpoint averaging

* run this after training has finished

### Model ensembling

* maybe just use the same model three times ?

## Translation

* Discuss the different beam search parameters...

## Alignment visualization




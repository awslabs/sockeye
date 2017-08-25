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

All of the commands below assume you're running on a CPU.
If you have a GPU available you can simply remove `--use-cpu`.

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

Looking at the data you can see how words are split into subwords separated by the special sequence `@@`:
```
Globaldarlehen sind Kreditlinien an zwischengeschaltete Institute -> Glob@@ al@@ dar@@ lehen sind Kredit@@ linien an zwischen@@ gesch@@ al@@ tete Institute
```

## Training

Having preprocessed our data we can start training.
Note that Sockeye will load all training data into memory in order to be able to easily reshuffle after every epoch.
Depending on the amount of RAM you have available you might want to reduce size of the training corpus for this tutorial:
```bash
# (Optional: run this if you have limited RAM on the training machine) 
head -n 200000 corpus.tc.BPE.de > corpus.tc.BPE.de.tmp && mv corpus.tc.BPE.de.tmp corpus.tc.BPE.de
head -n 200000 corpus.tc.BPE.en > corpus.tc.BPE.en.tmp && mv corpus.tc.BPE.en.tmp corpus.tc.BPE.en
```
That said, we can how kick off the training process:
```bash
python -m sockeye.train -s corpus.tc.BPE.de \
                        -t corpus.tc.BPE.en \
                        -vs newstest2016.tc.BPE.de \
                        -vt newstest2016.tc.BPE.en \
                        --num-embed 256 \
                        --rnn-num-hidden 512 \
                        --attention-type dot \
                        --max-seq-len 60 \
                        --monitor-bleu 500 \
                        --use-tensorboard \
                        --use-cpu \
                        -o wmt_model
```

This will train a 1-layer bi-LSTM encoder, 1 layer LSTM decoder with dot attention.
Sockeye offers a whole variety of different options regarding the model architecture,
such as stacked RNNs with residual connections (`--num-layers`, `--rnn-residual-connections`),
[Transformer](https://arxiv.org/abs/1706.03762) encoder and decoder (`--encoder transformer`, `--decoder transformer`),
various RNN (`--rnn-cell-type`) and attention (`--attention-type`) types and more.  

There are also several parameters controlling training itself.
Unless you specify a different optimizer (`--optimizer`) [Adam](https://arxiv.org/abs/1412.6980) will be used.
Additionally, you can control the batch size (`--batch-size`), the learning rate schedule (`--learning-rate-schedule`)
and other parameters relevant for training.

Training will run until the validation perplexity stops improving.
Alternatively, you can track BLEU on the validation set (`--optimized-metric bleu`).
Sockeye will then at every checkpoint start a decoder in a separate process running on the same device as training.
To make sure the decoder finishes before the next checkpoint one can subsample the validation set for
BLEU score calculation.
For example `--monitor-bleu 500` will calculate BLEU on a random subset of 500 sentences.
Perplexity will not be affected by this and still be calculated on the full validation set.

Training a model on this data set is going to take a while.
In the next section we discuss how you can monitor the training progress.

### Monitoring training progress

log/metrics file and tensorboard

* training is going to take a while...
* show how we can use tensorboard for tracking progress
* add a screenshot of how this looks like

```bash
tensorboard --logdir .
```

[http://localhost:6006](http://localhost:6006)

### Checkpoint averaging

* run this after training has finished

```bash
cp -r wmt_model wmt_model_avg
python -m sockeye.average -i wmt_model -o wmt_model_avg/param.best
```

### Model ensembling

* maybe just use the same model three times ?

## Translation

* Discuss the different beam search parameters...

## Alignment visualization




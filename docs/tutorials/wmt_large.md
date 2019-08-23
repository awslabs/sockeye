# WMT German-English News Translation (Large Data)

In this tutorial, we train a German to English Sockeye model on data provided by the [2018 Conference on Machine Translation (WMT 2018)](http://www.statmt.org/wmt18/).
This is a larger scale build that uses Sockeye's multi-GPU training.

## Setup

**NOTE**: This build assumes that 4 GPUs are available.

Install Sockeye or use the included Docker files to build an image.
If using the Docker image, all Sockeye commands can be prefixed with:

```bash
nvidia-docker run --rm -i -v $(pwd):/work -w /work sockeye:TAG
```

where TAG is the current commit (run `docker images`).

We use one external piece of software, the [subword-nmt](https://github.com/rsennrich/subword-nmt) tool that implements byte-pair encoding (BPE):

```bash
git clone https://github.com/rsennrich/subword-nmt.git
export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
```

## Data

We use the preprocessed data provided by the WMT 2018 news translation shared task.
Download and extract the data using the following commands:

```bash
wget http://data.statmt.org/wmt18/translation-task/preprocessed/de-en/corpus.gz
wget http://data.statmt.org/wmt18/translation-task/preprocessed/de-en/dev.tgz
zcat corpus.gz |cut -f1 >corpus.tc.de
zcat corpus.gz |cut -f2 >corpus.tc.en
tar xvzf dev.tgz
```

## Preprocessing

The data has already been tokenized and true-cased.
We next learn a joint sub-word vocabulary using BPE.
To speed up this step, we use a random sample of the corpus (note that the German and English samples will not be parallel--BPE training data does not require parallel data).

```bash
shuf -n 1000000 corpus.tc.de >corpus.tc.de.sample
shuf -n 1000000 corpus.tc.en >corpus.tc.en.sample
python -m subword_nmt.learn_joint_bpe_and_vocab \
    --input corpus.tc.de.sample corpus.tc.en.sample \
    -s 32000 \
    -o bpe.codes \
    --write-vocabulary bpe.vocab.de bpe.vocab.en
```

We use this sub-word vocabulary to encode our training, validation, and test data.
For simplicity, we use the 2016 data for validation and 2017 data for test.

```bash
python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 <corpus.tc.de >corpus.tc.de.bpe
python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 <corpus.tc.en >corpus.tc.en.bpe

python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 <newstest2016.tc.de >newstest2016.tc.de.bpe
python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 <newstest2016.tc.en >newstest2016.tc.en.bpe

python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 <newstest2017.tc.de >newstest2017.tc.de.bpe
python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 <newstest2017.tc.en >newstest2017.tc.en.bpe
```

## Training

Prior to training, we prepare the training data by splitting it into shards and serializing it in MXNet's NDArray format:

```bash
python -m sockeye.prepare_data \
    -s corpus.tc.de.bpe \
    -t corpus.tc.en.bpe \
    -o prepared_data \
    --shared-vocab \
    --word-min-count 2:2 \
    --max-seq-len 99 \
    --num-samples-per-shard 10000000 \
    --seed 1
```

We can now start Sockeye training:

```bash
python -m sockeye.train \
    -d prepared_data \
    -vs newstest2016.tc.de.bpe \
    -vt newstest2016.tc.en.bpe \
    -o model \
    --num-layers 6 \
    --transformer-model-size 512 \
    --transformer-attention-heads 8 \
    --transformer-feed-forward-num-hidden 2048 \
    --weight-tying \
    --weight-tying-type src_trg_softmax \
    --optimizer adam \
    --batch-size 8192 \
    --checkpoint-interval 4000 \
    --initial-learning-rate 0.0002 \
    --learning-rate-reduce-factor 0.9 \
    --learning-rate-reduce-num-not-improved 8 \
    --max-num-checkpoint-not-improved 60 \
    --decode-and-evaluate 500 \
    --device-ids -4 \
    --seed 1
```

This trains a "base" [Transformer](https://arxiv.org/abs/1706.03762) model using the [Adam](https://arxiv.org/abs/1412.6980) optimizer.
The learning rate will automatically be reduced when validation perplexity does not improve for 8 checkpoints (4000 batches per checkpoint) and training will conclude when validation perplexity does not improve for 60 checkpoints.
Sockeye also starts a separate decoder process at every checkpoint to evaluate metrics such as BLEU on a sample of the validation data (500 sentences).
Note that these scores are calculated on the tokens provided to Sockeye, e.g. in this tutorial BLEU will be calculated on the sub-words we created above.

Training a model on this scale of data takes around TODO hours on 4 NVIDIA Tesla V100-SXM2-16GB GPUs.

## Evaluation

TODO

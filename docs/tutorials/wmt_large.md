# Large Data: WMT 2018 German-English

This tutorial covers training a Sockeye model using an arbitrarily large amount of data.
We use the data provided for the [WMT 2018](http://www.statmt.org/wmt18/translation-task.html) German-English news task (41 million parallel sentences), though similar settings could be used for even larger data sets.

## Setup

**NOTE**: This build assumes that 4 local GPUs are available.

For this tutorial, we use the Sockeye Docker image.

1. Follow the linked instructions to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

2. Build the Docker image and record the commit used as the tag:

```bash
python3 sockeye_contrib/docker/build.py

export TAG=$(git rev-parse --short HEAD)
```

3. This tutorial uses two external pieces of software, the [subword-nmt](https://github.com/rsennrich/subword-nmt) tool that implements byte-pair encoding (BPE) and the [langid.py](https://github.com/saffsd/langid.py) tool that performs language identification:

```bash
git clone https://github.com/rsennrich/subword-nmt.git
export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH

git clone https://github.com/saffsd/langid.py.git
export PYTHONPATH=$(pwd)/langid.py:$PYTHONPATH
```

4. We also recommend installing [GNU Parallel](https://www.gnu.org/software/parallel/) to speed up preprocessing steps (run `apt-get install parallel` or `yum install parallel`).

## Data

We use the preprocessed data provided for the WMT 2018 news translation shared task.
Download and extract the data using the following commands:

```bash
wget http://data.statmt.org/wmt18/translation-task/preprocessed/de-en/corpus.gz
wget http://data.statmt.org/wmt18/translation-task/preprocessed/de-en/dev.tgz
zcat corpus.gz |cut -f1 >corpus.de
zcat corpus.gz |cut -f2 >corpus.en
tar xvzf dev.tgz '*.en' '*.de'
```

## Preprocessing

The data has already been tokenized and true-cased, however no significant corpus cleaning is applied.
The majority of the data is taken from inherently noisy web-crawls (sentence pairs are not always in the correct language, or even natural language text).
If we were participating in the WMT evaluation, we would spend a substantial amount of effort selecting clean training data from the noisy corpus.
For this tutorial, we run a simple cleaning step that retains sentence pairs for which a language identification model classifies the target side as English.
The use of GNU Parallel is optional, but makes this step much faster:

```bash
parallel --pipe --keep-order \
    python -m langid.langid --line -l en,de <corpus.en >corpus.en.langid

paste corpus.en.langid corpus.de |grep "^('en" |cut -f2 >corpus.de.clean
paste corpus.en.langid corpus.en |grep "^('en" |cut -f2 >corpus.en.clean
```

We next use BPE to learn a joint sub-word vocabulary from the clean training data.
To speed up this step, we use random samples of the source and target data (note that these samples will not be parallel, but BPE training does not require parallel data).

```bash
shuf -n 1000000 corpus.de.clean >corpus.de.clean.sample
shuf -n 1000000 corpus.en.clean >corpus.en.clean.sample

python -m subword_nmt.learn_joint_bpe_and_vocab \
    --input corpus.de.clean.sample corpus.en.clean.sample \
    -s 32000 \
    -o bpe.codes \
    --write-vocabulary bpe.vocab.de bpe.vocab.en
```

We use this vocabulary to encode our training, validation, and test data.
For simplicity, we use the 2016 data for validation and 2017 data for test.
GNU parallel can also significantly speed up this step.

```bash
parallel --pipe --keep-order \
    python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 <corpus.de.clean >corpus.de.clean.bpe
parallel --pipe --keep-order \
    python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 <corpus.en.clean >corpus.en.clean.bpe

python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 <newstest2016.tc.de >newstest2016.tc.de.bpe
python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 <newstest2016.tc.en >newstest2016.tc.en.bpe

python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 <newstest2017.tc.de >newstest2017.tc.de.bpe
python -m subword_nmt.apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 <newstest2017.tc.en >newstest2017.tc.en.bpe
```

## Training

Now that our data is cleaned and sub-word encoded, we are almost ready to start model training.
We first run a data preparation step that splits the training data into shards and serializes it in MXNet's NDArray format.
This allows us to train on data of any size by efficiently loading and unloading different pieces during training:

```bash
nvidia-docker run --rm -i -v $(pwd):/work -w /work sockeye:$TAG \
    python -m sockeye.prepare_data \
        -s corpus.de.clean.bpe \
        -t corpus.en.clean.bpe \
        -o prepared_data \
        --shared-vocab \
        --word-min-count 2 \
        --pad-vocab-to-multiple-of 8 \
        --bucket-width 8 \
        --no-bucket-scaling \
        --max-seq-len 95 \
        --num-samples-per-shard 10000000 \
        --seed 1
```

We then start Sockeye training:

```bash
nvidia-docker run --rm -i -v $(pwd):/work -w /work -e OMP_NUM_THREADS=4 sockeye:$TAG \
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
        --update-interval 4 \
        --round-batch-sizes-to-multiple-of 8 \
        --checkpoint-interval 1000 \
        --initial-learning-rate 0.0004 \
        --learning-rate-reduce-factor 0.9 \
        --learning-rate-reduce-num-not-improved 8 \
        --max-num-checkpoint-not-improved 60 \
        --decode-and-evaluate 500 \
        --device-ids -4 \
        --seed 1
```

**Faster training**:

- To run FP16 training using a fixed loss scaling factor, add `--dtype float16`.
- To use MXNet's Automatic Mixed Precision, add `--amp`.

This trains a "base" [Transformer](https://arxiv.org/abs/1706.03762) model using the [Adam](https://arxiv.org/abs/1412.6980) optimizer with a batch size of 32,768 (8192 x 4) tokens.
The learning rate will automatically reduce when validation perplexity does not improve for 8 checkpoints (1000 updates per checkpoint) and training will conclude when validation perplexity does not improve for 60 checkpoints.
At each checkpoint, Sockeye runs a separate decoder process to evaluate metrics such as BLEU on a sample of the validation data (500 sentences).
Note that these scores are calculated on the tokens provided to Sockeye, e.g. in this tutorial BLEU will be calculated on the sub-words we created above.

## Evaluation

Now the model is ready to translate data.
Input should be preprocessed identically to the training data, including sub-word encoding (BPE).
Run the following to translate the test set that we've already preprocessed:

```bash
nvidia-docker run --rm -i -v $(pwd):/work -w /work sockeye:$TAG \
    python -m sockeye.translate \
        -i newstest2017.tc.de.bpe \
        -o newstest2017.tc.hyp.bpe \
        -m model \
        --beam-size 5 \
        --batch-size 64 \
        --device-ids -1
```

To evaluate the translations, reverse the BPE sub-word encoding and run [sacreBLEU](https://github.com/mjpost/sacreBLEU) to compute the BLEU score:

```bash
sed -re 's/(@@ |@@$)//g' <newstest2017.tc.hyp.bpe >newstest2017.tc.hyp

nvidia-docker run --rm -i -v $(pwd):/work -w /work sockeye:$TAG \
    sacrebleu newstest2017.tc.en -tok none -i newstest2017.tc.hyp
```

Note that this is tokenized, normalized, and true-cased data.
If we were actually participating in WMT, the translations would need to be recased and detokenized for human evaluation.

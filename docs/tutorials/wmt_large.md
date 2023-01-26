# WMT 2014 English-German

This tutorial covers training a standard big transformer on data of any size.
We start with relatively small data where model training converges quickly (WMT14 En-De).
The same settings can be used for arbitrarily large data.
The training recipe is optimized for 8 local GPUs, but can be scaled down to 4 or 1 for testing.

## Setup

Install Sockeye:
```bash
git clone https://github.com/awslabs/sockeye.git
cd sockeye && pip3 install --editable .
```

Install Subword-NMT:
```bash
pip3 install subword-nmt
```

## Data

We use the WMT 2014 English-German data pre-processed by the [Stanford NLP Group](https://nlp.stanford.edu/projects/nmt/) (4.5M parallel sentences):

```bash
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en'
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de'
for YEAR in 2012 2013 2014; do
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.en"
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.de"
done
cat newstest{2012,2013}.en >dev.en
cat newstest{2012,2013}.de >dev.de
cp newstest2014.en test.en
cp newstest2014.de test.de
```

## Preprocessing

The data is already tokenized, so we only need to apply byte-pair encoding ([Sennrich et al., 2016](https://aclanthology.org/P16-1162/)):

```bash
cat train.de train.en |subword-nmt learn-bpe -s 32000 >codes
for SET in train dev test; do
  subword-nmt apply-bpe -c codes <${SET}.en >${SET}.en.bpe
  subword-nmt apply-bpe -c codes <${SET}.de >${SET}.de.bpe
done
```

## Training

We first split the byte-pair encoded training data into shards and serialize it in PyTorch's tensor format.
This allows us to train on data of any size by loading and unloading different pieces throughout training:

```bash
sockeye-prepare-data \
    --source train.en.bpe --target train.de.bpe --shared-vocab \
    --word-min-count 2 --pad-vocab-to-multiple-of 8 --max-seq-len 95 \
    --num-samples-per-shard 10000000 --output prepared --max-processes $(nproc)
```

We then launch distributed training on 8 GPUs.
The following command trains a big transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) using the large batch recipe described by Ott et al. ([2018](https://arxiv.org/abs/1806.00187)):

```bash
torchrun --no_python --nproc_per_node 8 sockeye-train \
    --prepared-data prepared --validation-source dev.en.bpe \
    --validation-target dev.de.bpe --output model --num-layers 6 \
    --transformer-model-size 1024 --transformer-attention-heads 16 \
    --transformer-feed-forward-num-hidden 4096 --amp --batch-type max-word \
    --batch-size 5000 --update-interval 10 --checkpoint-interval 500 \
    --max-updates 15000 --optimizer-betas 0.9:0.98 --dist \
    --initial-learning-rate 0.06325 \
    --learning-rate-scheduler-type inv-sqrt-decay --learning-rate-warmup 4000 \
    --seed 1 --quiet-secondary-workers
```

Alternate command for 4 GPUs:

```bash
torchrun --no_python --nproc_per_node 4 sockeye-train \
    --prepared-data prepared --validation-source dev.en.bpe \
    --validation-target dev.de.bpe --output model --num-layers 6 \
    --transformer-model-size 1024 --transformer-attention-heads 16 \
    --transformer-feed-forward-num-hidden 4096 --amp --batch-type max-word \
    --batch-size 5000 --update-interval 20 --checkpoint-interval 500 \
    --max-updates 15000 --optimizer-betas 0.9:0.98 --dist \
    --initial-learning-rate 0.06325 \
    --learning-rate-scheduler-type inv-sqrt-decay --learning-rate-warmup 4000 \
    --seed 1 --quiet-secondary-workers
```

Alternate command for 1 GPU:

```bash
sockeye-train \
    --prepared-data prepared --validation-source dev.en.bpe \
    --validation-target dev.de.bpe --output model --num-layers 6 \
    --transformer-model-size 1024 --transformer-attention-heads 16 \
    --transformer-feed-forward-num-hidden 4096 --amp --batch-type max-word \
    --batch-size 5000 --update-interval 80 --checkpoint-interval 500 \
    --max-updates 15000 --optimizer-betas 0.9:0.98 \
    --initial-learning-rate 0.06325 \
    --learning-rate-scheduler-type inv-sqrt-decay --learning-rate-warmup 4000 \
    --seed 1
```

Training on larger data typically requires more updates for the model to reach a perplexity plateau.
When using the above recipe with larger data sets, increase the number of updates (`--max-updates`) or train until the model does not improve over many checkpoints (specify `--max-num-checkpoint-not-improved X` instead of `--max-updates Y`).

## Evaluation

When training is complete, we translate the preprocessed test set:

```bash
sockeye-translate \
    --input test.en.bpe \
    --output out.bpe \
    --model model \
    --dtype float16 \
    --beam-size 5 \
    --batch-size 64
```

We then reverse BPE and score the translations against the reference using [sacreBLEU](https://github.com/mjpost/sacreBLEU):

```bash
sed -re 's/(@@ |@@$)//g' <out.bpe >out.tok
sacrebleu test.de -tok none -i out.tok
```

Note that this is still tokenized, normalized, and true-cased data.
If we were actually participating in WMT, we would recase and detokenize the translations for human evaluation.

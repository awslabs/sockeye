# Training

## Data preparation

Sockeye can read the raw data at training time in two sentence-parallel files via the `--source` and `--target` command-line options.
You can also prepare the data ahead of time and dump it to disk as MXNet NDArrays.
This eliminates the data loading time when running training (since three passes over the raw data are required), and also reduces memory consumption,
since prepared data is also placed into random shards (which have one million lines each, by default).
To run data preparation, you can use the following command:

```bash
> python -m sockeye.prepare_data
usage: prepare_data.py [-h] --source SOURCE [--source-factors SOURCE_FACTORS [SOURCE_FACTORS ...]]
                       [--source-factors-use-source-vocab SOURCE_FACTORS_USE_SOURCE_VOCAB [SOURCE_FACTORS_USE_SOURCE_VOCAB ...]]
                       [--target-factors TARGET_FACTORS [TARGET_FACTORS ...]]
                       [--target-factors-use-target-vocab TARGET_FACTORS_USE_TARGET_VOCAB [TARGET_FACTORS_USE_TARGET_VOCAB ...]] --target
                       TARGET [--source-vocab SOURCE_VOCAB] [--target-vocab TARGET_VOCAB]
                       [--source-factor-vocabs SOURCE_FACTOR_VOCABS [SOURCE_FACTOR_VOCABS ...]]
                       [--target-factor-vocabs TARGET_FACTOR_VOCABS [TARGET_FACTOR_VOCABS ...]] [--shared-vocab] [--num-words NUM_WORDS]
                       [--word-min-count WORD_MIN_COUNT] [--pad-vocab-to-multiple-of PAD_VOCAB_TO_MULTIPLE_OF] [--no-bucketing]
                       [--bucket-width BUCKET_WIDTH] [--bucket-scaling] [--no-bucket-scaling] [--max-seq-len MAX_SEQ_LEN]
                       [--num-samples-per-shard NUM_SAMPLES_PER_SHARD] [--min-num-shards MIN_NUM_SHARDS] [--seed SEED] --output OUTPUT
                       [--max-processes MAX_PROCESSES] [--quiet] [--quiet-secondary-workers] [--no-logfile] [--loglevel {INFO,DEBUG}]
prepare_data.py: error: the following arguments are required: --source/-s, --target/-t, --output/-o
```

The main arguments are the required ones above (`--source`, `--target`, and `--output` to specify the directory to write the prepared data to).
Some other important ones are:

- `--shared-vocab`: to produce a shared vocabulary between the source and target sides of the corpora.
- `--num-samples-per-shard`: to control the shard size.

At training time (see next section), you then specify `--prepared-data` instead of `--source` and `--target`.

## Training

Training is carried out by the `sockeye.train` module. Basic usage is given by

```bash
> python -m sockeye.train
usage: train.py [-h] --source SOURCE --target TARGET --validation-source
                VALIDATION_SOURCE --validation-target VALIDATION_TARGET
                --output OUTPUT [...]
```

Training requires 5 arguments:
* `--source`, `--target`: give the training data files. Gzipped files are supported, provided that their filenames end with .gz.
* `--validation-source`, `--validation-target`: give the validation data files, gzip supported as above.
* `--output`: gives the output directory where the intermediate and final results will be written to.
Intermediate directories will be created if needed.
Logging will be written to `<model_dir>/log` as well as being echoed on the console.

For a complete list of supported options use the `--help` option.

### Data format

All input files files should be UTF-8 encoded, tokenized with standard whitespaces.
Each line should contain a single sentence and the source and target files should have the same number of lines.
Vocabularies will automatically be created from the training data and vocabulary coverage on the validation set during initialization will be reported.

### Checkpointing and early-stopping

Training is governed by the concept of "checkpoints", rather than epochs.
You can specify the checkpoint interval in terms of updates/batches with `--checkpoint-interval`.
Training performs early-stopping to prevent overfitting, i.e., training is stopped once a defined evaluation metric computed on the held-out validation data does not improve for a number of checkpoints given by the parameter `--max-num-checkpoint-not-improved`.
You can specify a maximum number of updates/batches using `--max-updates`.

Perplexity is the default metric to be considered for early-stopping, but you
can also choose to optimize accuracy or BLEU using the `--optimized-metric`
argument. In case of optimizing with respect to BLEU, you will need to set `--decode-and-evaluate` > 0
to decode validation at every checkpoint.

Note that evaluation metrics for training data and held-out validation data are
written in a tab-separated file called `metrics`.

At each checkpoint, the internal state of the training process is stored to
disk. If the training is interrupted (e.g. due to a hardware failure), you can
start sockeye again, with the same parameters as for the initial call, and
training will resume from the last checkpoint. Note that this is different to
using the `--params` argument. This argument is used only to initialize the
training with pre-computed values for the parameters of the model, but the
parameters of the optimizer and other parts of the system are initialized from
scratch.

### Monitoring training progress with Tensorboard

Sockeye can write all evaluation metrics in a Tensorboard compatible format.
This way you can monitor the training progress in the browser.
To visualize logged events, install Tensorboard:
```bash
> pip install tensorboard
```

Start tensorboard and point it to the model directory (or any parent directory):
```bash
> tensorboard --logdir model_dir
```

### CPU/GPU training

By default, training is carried out on the first GPU device of your machine.
You can specify an alternative GPU device with the `--device-id` option.
If you do not have or do not want to use a GPU, specify `--use-cpu`.
In this case a drop in training throughput is expected.

#### Multi-GPU training

Training can be carried out on multiple GPUs. See the
[WMT 2014 English-German tutorial](tutorials/wmt_large.md) for more information.


### Checkpoint averaging

A common technique for improving model performance is to average the weights for the last checkpoints.
This can be done as follows:
```bash
> python -m sockeye.average <model_dir> -o <model_dir>/model.best.avg.params
```

## Source factors

Sockeye supports source factors, which are described in:

> Rico Sennrich and Barry Haddow. 2016.
> [Linguistic Input Features Improve Neural Machine Translation](http://www.aclweb.org/anthology/W16-2209)
> Proceedings of the First Conference on Machine Translation: Volume 1, Research Papers.

Factors are enabled with two flags: `--source-factors` and `--source-factors-num-embed`.
The `--source-factors` argument takes one or more files that are token-parallel to the source.
This means that each line has the exact same number of whitespace-delimited tokens as the source file (`--source`).
For example, if you have the following line of a source sentence:

> the boy ate the waff@@ le .

You need a corresponding feature line of the following form:

> O O O O B E O

(Same number of tokens).

This flag can also be supplied to [the data preparation step](#data-preparation).

Each source factor has its own vocabulary and learned embedding.
The source factors can be combined with the word embeddings in two ways: concatenation and summing.
For concatenation (`--source-factors-combine concat`, the default), you need to specifiy the embedding sizes for each factor.
This is done with `--source-factors-num-embed X1 X2 ...`.
Since these embeddings concatenated to those of the word embeddings, the total source embedding size will be the sum of the word embeddings and all source factor embeddings.
You can also sum the embeddings (`--source-factors-combine sum`).
In this case, you do not need to specify `--source-factors-num-embed`, since they are automatically all set to the size of the word embeddings (`--num-embed`).

You then also have to apply factors for the source side [at inference time](inference.md#source-factors).

## Target factors

Sockeye supports target factors, i.e. alternative tokens/features to be predicted alongside the main decoding output.
Similar to source factors, the target factor files at training time need to be token-parallel to the target side.
For example, if you have the following line of a target sentence:

> der junge aß die waff@@ el .

A POS target factor could look like like this:

> DET N V DET N N PUNC

Internally, Sockeye will shift all target factors to the right by 1 to condition the prediction of the factors on the previously generated target word.
During training, Sockeye will optimize multiple losses in a multi-task setting, one for each target factor. The weight of the losses can be controlled by `--target-factors-weight`.

To receive the target factor predictions at inference time, use `--output-type translation_with_factors`.
Target factors do not participate in beam search, i.e. each target factor prediction is the argmax of the corresponding output layer distribution.

## Length ratio prediction

Sockeye supports an auxiliary training objective that predicts length ratio (|reference|/|input|) or the reference length for each input,
that can be enabled by setting `--length-task`, respectively, to `ratio` or to `length`.
Specify `--length-task-layers` to set the number of layers in the prediction MLP.
The weight of the loss in the global training objective is controlled with `--length-task-weight` (standard cross-entropy loss has weight 1.0).
During inference the predictions can be used to reward longer translations by enabling `--brevity-penalty-type`.

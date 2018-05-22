# Sockeye Autopilot

This module provides automated end-to-end system building for popular model types on public data sets.
These capabilities can also be used independently: users can provide their own data for model training or use Autopilot to download and pre-process public data for other use.
All intermediate files are preserved as plain text and commands are recorded, letting users take over at any point for further experimentation.

## Quick Start

If Sockeye is installed via pip or source, Autopilot can be run directly:

```bash
> sockeye-autopilot
```

This is equivalent to:

```bash
> python -m contrib.autopilot.autopilot
```

With a single command, Autopilot can download and pre-process training data, then train and evaluate a translation model.
For example, to build a transformer model on the WMT14 English-German benchmark, run:

```bash
> sockeye-autopilot --task wmt14_en_de --model transformer
```

By default, systems are built under `$HOME/sockeye_autopilot`.
The `--workspace` argument can specify a different location.
Also by default, a single GPU is used for training and decoding.
The `--gpus` argument can specify a larger number of GPUs for parallel training or `0` for CPU mode only.

Autopilot populates the following sub-directories in a workspace:

- cache: raw downloaded files from public data sets.
- third_party: downloaded third party tools for data pre-processing (currently [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer) and [subword-nmt](https://github.com/rsennrich/subword-nmt))
- logs: log files for various steps.
- systems: contains a single directory for each task, such as "wmt14_en_de".  Task directories contain (after a successful build):
  - data: raw, tokenized, and byte-pair encoded data for train, dev, and test sets.
  - model.bpe: byte-pair encoding model
  - model.*: directory for each Sockeye model built, such as "model.transformer"
  - results: decoding output and BLEU scores.  When starting with raw data, the .sacrebleu file contains a score that can be compared against official WMT results.

### Custom Data

Models can be built using custom data with any level of pre-processing.
For example, to use custom German-English raw data, run:

```bash
> sockeye-autopilot --model transformer \
    --custom-task my_task \
    --custom-text-type raw \
    --custom-lang de en \
    --custom-train train.de train.en \
    --custom-dev dev.de dev.en \
    --custom-test test.de test.en \
```

Pre-tokenized or byte-pair encoded data can be used with `--custom-text-type tok` and `--custom-text-type bpe`.
The `--custom-task` argument is used for directory naming.
A custom number of BPE operations can be specified with `--custom-bpe-op`.

### Data Preparation Only

To use Autopilot for data preparation only, simply provide `none` as the model type:

```bash
> sockeye-autopilot --task wmt14_en_de --model none
```

## Automation Steps

This section describes the steps Autopilot runs as part of each system build.
Builds can be stopped and re-started (CTRL+C).
Some steps are atomic while others (such as translation model training) can be resumed.
Each completed step records its success so a re-started build can pick up from the last finished step.

### Checkout Third Party Tools

If the task requires tokenization, check out the [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer).
If the task requires byte-pair encoding, check out the [subword-nmt](https://github.com/rsennrich/subword-nmt)) module.
Store git checkouts of these tools in the third_party directory for re-use with future tasks in the same workspace.

NOTE: These tools have different open source licenses than Sockeye.
See the included license files for more information.

### Download Data

Download to the cache directory all raw files referenced by the current task (if not already present).
See `RAW_FILES` and `TASKS` in `tasks.py` for examples of tasks referencing various publicly available data files.

### Populate Input Files

For known tasks, populate parallel train, dev, and test files under "data/raw" by extracting lines from raw files downloaded in the previous step.
For custom tasks, copy the user-provided data.
Train and dev files are concatenated while test sets are preserved as separate files.

This step includes Unicode whitespace normalization to ensure that only ASCII newlines are considered as line breaks (spurious Unicode newlines are a known issue in some noisy public data).

### Tokenize Data

If data is not pre-tokenized, run the Moses tokenizer and store the results in "data/tok".
For known tasks, use the listed `src_lang` and `trg_lang` (see `TASKS` in `tasks.py`).
For custom tasks, use the provided `--custom-lang` arguments.

### Byte-Pair Encode Data

If the data is not already byte-pair encoded, learn a BPE model "model.bpe" and apply it to the data, storing the results in "data/bpe".
For known tasks, use the listed number of operations `bpe_op`.
For custom tasks, use the provided `--custom-bpe-op` argument.

### Train Translation Model

Run `sockeye.train` and `sockeye.average` to learn a translation model on the byte-pair encoded data.
Use the arguments listed for the provided `--model` argument and specify "model.MODEL" (e.g., "model.transformer") as the model directory.
See `MODELS` in `models.py` for examples of training arguments.

This step can take several days and progress can be checked via the log file or tensorboard.
This step also supports resuming from a partially trained model.

### Translate Test Sets

Run `sockeye.translate` to decode each test set using the specified settings.
See `DECODE_ARGS` in `models.py` for decoding settings.

### Evaluate Translations

Provide the following outputs to the user under "results":

- test.N.MODEL.SETTINGS.bpe.bleu: BLEU score of raw decoder output against byte-pair encoded references
- test.N.MODEL.SETTINGS.tok.bleu: BLEU score of word-level decoder output against tokenized references
- test.N.MODEL.SETTINGS.detok.sacrebleu: BLEU score of detokenized decoder output against raw references using [SacreBLEU](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu).  These scores are directly comparable to those reported in WMT evaluations.

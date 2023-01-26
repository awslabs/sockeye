# Sequence copy model

This tutorial will show you the basic usage of Sockeye on a on a simple task: copying a sequence.
We will generate sequences consisting of digits of variable lengths.
The task is then to train a model that copies the sequence from the source to the target.
This task is on the one hand difficult enough to be interesting and on the other and allows for quickly training a model.

## Setup
For this tutorial we assume that you have successfully [installed](../setup.md) Sockeye.
We will be using scripts from the Sockeye repository, so you should either clone the repository or manually download the scripts.
Just as a reminder: Everything is run using Python 3, so depending on your setup you may have to replace `python` with `python3` below.
All of the commands below assume you are running on a CPU.
If you have a GPU available you can simply remove `--use-cpu`.

## 1. Generating the data
As a first step we will generate a synthetic data set consisting of random sequences of digits.
These sequences are then split into disjoint training and development sets.
Run the following command to create the data set:

```bash
wget https://raw.githubusercontent.com/awslabs/sockeye/main/docs/tutorials/seqcopy/genseqcopy.py
python genseqcopy.py
```

After running this script you have (under 'data/') a training (`train.source`, `train.target`) and a development data set (`dev.source`, `dev.target`).
The generated sequences will look like this:

```
2 3 5 5 4 6 7 0 3 8 10 9 3 6
9 9 1 5 3 0 5 4 0 8 8 5 7 7 8 7 3 1 0
9 1 9 7 9 1 9 9 9 3 9 3 2 8 0 1 6 10 4 3 1 9 2 7 1 5 7 7 5 5
2 1 4 10 7 7 7 2 10 9 4 9 9 7 8 4 10 6 8 2 6 7 5 3 2
4 6 0 7 8 8 6 3 4 10 2 10 6 9 5 3
8 0 5 4 1 8 0 8 7 4 4 0 0 9 5 8 9
```

## 2. Training

Now that we have some training data to play with we can train our model.
Start training by running the following command:

```bash
python3 -m sockeye.train -s data/train.source \
                         -t data/train.target \
                         -vs data/dev.source \
                         -vt data/dev.target \
                         --encoder transformer --decoder transformer \
                         --num-layers 1:1 \
                         --num-embed 32 \
                         --transformer-model-size 32 \
                         --transformer-feed-forward-num-hidden 64 \
                         --transformer-attention-heads 4 \
                         --use-cpu \
                         --max-num-checkpoint-not-improved 3 \
                         -o seqcopy_model
```

This will train a 1-layer Transformer model with 32 hidden units as the embedding size.
The Feed-Forward sublayers have 64 hidden units and attention mechanisms are using 4 heads.
Looking at the log we can see that our training data was assigned to buckets according to their lengths.
Additionally, Sockeye will take care of correctly padding sequences and masking relevant parts of the network,
in order to deal with sequences of variable length.


### Metrics and checkpointing
During training Sockeye will print relevant metrics on both the training and the validation data.
Validation metrics are evaluated every time we create a checkpoint.
During checkpointing the current model parameters are saved into the model directory and current validation scores are evaluated.
By default Sockeye will create a checkpoint every 4000 updates.
This can be adjusted through the `--checkpoint-interval` parameter.

From the log you can see that initially the perplexity is around `20.0`:
```bash
...
[INFO:sockeye.training] Early stopping by optimizing 'perplexity'
[INFO:sockeye.model] Saved model config to "seqcopy_model/config"
[INFO:sockeye.training] Training started.
[INFO:sockeye.training] Epoch[0] Batch [50]	Speed: 429.27 samples/sec 10879.00 tokens/sec 2.16 updates/sec	perplexity=20.074619
[INFO:sockeye.training] Epoch[0] Batch [100]	Speed: 534.38 samples/sec 13846.37 tokens/sec 2.76 updates/sec	perplexity=17.064554
...
```
As training progresses we see that after the first checkpoint (~7 epochs) the validation perplexity is at ~1.05.
Sockeye performs early stopping based on the validation metrics tracked when checkpointing.
Once the validation metrics have not improved for several checkpoints the training is stopped.
The number of tolerated non-improving checkpoints can be adjusted (`--max-num-checkpoint-not-improved`).

### Trained model

The trained model can be found in the folder `seqcopy_model`.
The folder contains everything necessary to run the model after training.
Most importantly `params.best` contains the parameters with the best validation score.
During training `param.best` will continously be updated to point to the currently best parameters.
This means that even while the model is still training you can use the model folder for translation, as described in the next section.

All other parameters can be found in files named `param.$NUM_CHECKPOINT`.
The `config` contains all model parameters as well as a reference to the data sets used during training.
`version` references the version of Sockeye used for training in order to check potential compatibility issues with the version used for decoding.

Additionally, we keep a copy of the `log` that you also saw printed on stdout.
The source and target vocabularies are stored in `vocab.src.json` and `vocab.trg.json`.
If you open the file you can see that in addition to the digits Sockeye also added special symbols indicating sentence boundaries, unknown words and padding symbols.


## 3. Translation

```bash
> echo "7 6 7 7 10 2 0 8 0 5 7 3 5 6 4 0 0 2 10 0" | \
  python -m sockeye.translate -m seqcopy_model --use-cpu

        7 6 7 7 10 2 0 8 0 5 7 3 5 6 4 0 0 2 10 0

```

Note that the model was trained on sequences consisting of between 10 and 30 digits.
Therefore, the model will most likely have some difficulties with sequences shorter than 10 digits.
By default Sockeye will read sentence from stdin and print the translations on stdout.

Internally Sockeye will run a beam search in order to (approximately) find the translation with the highest probability.

Instead of using the parameters with the best validation score we can also use other checkpoints using the `-c` parameter to use a checkpoint earlier in the training before the model converged:
```bash
> echo "7 6 7 7 10 2 0 8 0 5 7 3 5 6 4 0 0 2 10 0" | \
  python -m sockeye.translate -m seqcopy_model --use-cpu -c 3

        7 6 7 7 10 2 0 8 0 5 7 0 7 3 5 6 0 0 2 0 10
```
As the model has not converged yet it is still making a few mistakes when copying the sequence.

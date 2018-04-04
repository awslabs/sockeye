# User documentation

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
Vocabularies will automatically be created from the training data and vocabulary 
coverage on the validation set during initialization will be reported.

### Checkpointing and early-stopping

Training is governed by the concept of "checkpoints", rather than epochs. You
can specify the checkpoint frequency in terms of updates/batches with
`--checkpoint-frequency`.  Training performs early-stopping to prevent
overfitting, i.e. training is stopped once a defined evaluation metric computed
on the held-out validation data does not improve for a number of checkpoints
given by the parameter `--max-num-checkpoint-not-improved`.  You can specify a
maximum number of updates/batches using `--max-updates`.

Perplexity is the default metric to be considered for early-stopping, but you
can also choose to optimize accuracy or BLEU using the `--optimized-metric`
argument. In case of optimizing with respect to  BLEU, you will need to specify
`--monitor-bleu`. For efficiency reasons, sockeye spawns a sub-processes after each
checkpoint to decode the validation data and compute BLEU. This may introduce
some delay in the reporting of results, i.e. there may be checkpoints with no
BLEU results reported or with results corresponding to older checkpoints. This
is expected behaviour and sockeye internally keeps track of the results in the
correct order.

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
To enable this feature, install the MXNet-compatible interface, mxboard:
```bash
> pip install mxboard
```

For visualization, you still need the official tensorboard release (i.e. `pip install tensorboard`).
Start tensorboard and point it to the model directory (or any parent directory):
```bash
> tensorboard --logdir model_dir
```

### CPU/GPU training

By default, training is carried out on the first GPU device of your machine.
You can specify alternative GPU devices with the `--device-ids` option, with
which you can also activate multi-GPU training (see below). If
`--device-ids -1`, sockeye will try to find a free GPU on your machine and block
until one is available. The locking mechanism is based on files and therefore assumes all processes are running
on the same machine with the same file system.
If this is not the case there is a chance that two processes will be using the same GPU and you run out of GPU memory.
If you do not have or do not want to use a GPU, specify `--use-cpu`.
In this case a drop in performance is expected.

##### Multi-GPU training
Training can be carried out on multiple GPUs by either specifying multiple GPU device ids:
`--device-ids 0 1 2 3`, or specifying the number GPUs required: `--device-ids -n` attempts to acquire `n` GPUs through
the locking mechanism described above.
This will train using [Data Parallelism](https://github.com/dmlc/mxnet/blob/master/docs/how_to/multi_devices.md).
MXNet will divide the data in each batch and send it to the different devices.
Note that you should increase the batch size: for `k` GPUs use ``--batch-size k*<original_batch_size>``.
Also note that this will likely linearly increase your throughput in terms of sentences/second, but not necessarily
increase the model's convergence speed.


### Checkpoint averaging

A common technique for improving model performance is to average the weights for the last checkpoints.
This can be done as follows:
```bash
> python -m sockeye.average <model_dir> -o <model_dir>/model.best.avg.params
```

## Translation

Translating is handled by the `sockeye.translate` module:
```bash
> python -m sockeye.translate
```

The only required argument is `--models`, which should point to an `<model_dir>`
folder of trained models.  By default, sockeye chooses the parameters from the
best checkpoint and uses these for translation.  You can specify parameters
from a specific checkpoint by using `--checkpoints X`.

You can control the size of the beam using `--beam-size` and the maximum input
length by `--max-input-length`.  Sentences that are longer than
`max-input-length` are stripped.

Input is read from the standard input and the output is written to the standard
output.  The CLI will log translation speed once the input is consumed. Like in
the training module, the first GPU device is used by default. Note however that
multi-GPU translation is not currently supported. For CPU decoding use
`--use-cpu`.

Use the `--help` option to see a full list of options for translation.

### Ensemble Decoding
Sockeye supports ensemble decoding by specifying multiple model directories and
multiple checkpoints. The given lists must have the same length, such that the
first given checkpoint will be taken from the first model directory, the second
specified checkpoint from the second directory, etc.
```bash
> python -m sockeye.translate --models [<m1prefix> <m2prefix>] --checkpoints [<cp1> <cp2>]
```

### Visualization
The default mode of the translate CLI is to output translations to STDOUT. You
can also print out an ASCII matrix of the alignments using `--output-type
align_text`, or save the alignment matrix as a PNG plot using `--output-type
align_plot`. The PNG files will be written to files beginning with the prefix
given by the `--align-plot-prefix` option, one for each input sentence, indexed
by the sentence id.

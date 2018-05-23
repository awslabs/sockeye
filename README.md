# Sockeye

[![PyPI version](https://badge.fury.io/py/sockeye.svg)](https://badge.fury.io/py/sockeye)
[![GitHub license](https://img.shields.io/github/license/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/issues)
[![Build Status](https://travis-ci.org/awslabs/sockeye.svg?branch=master)](https://travis-ci.org/awslabs/sockeye)
[![Documentation Status](https://readthedocs.org/projects/sockeye/badge/?version=latest)](http://sockeye.readthedocs.io/en/latest/?badge=latest)

This package contains the Sockeye project,
a sequence-to-sequence framework for Neural Machine Translation based on Apache MXNet Incubating.
It implements state-of-the-art encoder-decoder architectures, such as
- Deep Recurrent Neural Networks with Attention [[Bahdanau, '14](https://arxiv.org/abs/1409.0473)]
- Transformer Models with self-attention [[Vaswani et al, '17](https://arxiv.org/abs/1706.03762)]
- Fully convolutional sequence-to-sequence models [[Gehring et al, '17](https://arxiv.org/abs/1705.03122)]

If you use Sockeye, please cite:

Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton and Matt Post (2017):
[Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/abs/1712.05690). In eprint arXiv:cs-CL/1712.05690.

```
@article{Sockeye:17,
   author = {Hieber, Felix and Domhan, Tobias and Denkowski, Michael
           and Vilar, David and Sokolov, Artem and Clifton, Ann and Post, Matt},
    title = "{Sockeye: A Toolkit for Neural Machine Translation}",
  journal = {arXiv preprint arXiv:1712.05690},
archivePrefix = "arXiv",
   eprint = {1712.05690},
 primaryClass = "cs.CL",
 keywords = {Computer Science - Computation and Language,
             Computer Science - Learning,
             Statistics - Machine Learning},
     year = 2017,
    month = dec,
      url = {https://arxiv.org/abs/1712.05690}
}
```

If you are interested in collaborating or have any questions, please submit a pull request or issue. [Click to find our developer guidelines](docs/development.md).
You can also send questions to *sockeye-dev-at-amazon-dot-com*.

Recent developments and changes are tracked in our [changelog](https://github.com/awslabs/sockeye/blob/master/CHANGELOG.md).

## Dependencies

Sockeye requires:
- **Python3**
- [MXNet-1.1.0](https://github.com/apache/incubator-mxnet/tree/1.1.0)
- numpy

## Installation

There are several options for installing Sockeye and it's dependencies. Below we list several alternatives and the
corresponding instructions.

### Either: AWS DeepLearning AMI

[AWS DeepLearning AMI](https://aws.amazon.com/amazon-ai/amis/) users only need to run the following line to install sockeye:

```bash
> sudo pip3 install sockeye --no-deps
```

For other environments, you can choose between installing via pip or directly from source. Note that for the
remaining instructions to work you will need to use `python3` instead of `python` and `pip3` instead of `pip`.


### Or: pip package

#### CPU

```bash
> pip install sockeye
```

#### GPU

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet Incubating contains the GPU
bindings.
Depending on your version of CUDA, you can do this by running the following:
```bash
> wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install sockeye --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
> rm requirements.gpu-cu${CUDA_VERSION}.txt
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), or `91` (9.1).

### Or: From Source

#### CPU

If you want to just use sockeye without extending it, simply install it via
```bash
> pip install -r requirements.txt
> pip install .
```
after cloning the repository from git.

#### GPU

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by
running the following:

```bash
> pip install -r requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), or `91` (9.1).

### Optional dependencies
In order to write training statistics to a Tensorboard event file for visualization, you can optionally install mxboard
 (````pip install mxboard````). To visualize these, run the Tensorboard tool (`pip install tensorboard tensorflow`) with
 the logging directory pointed to the training output folder: `tensorboard --logdir <model>`

If you want to create alignment plots you will need to install matplotlib (````pip install matplotlib````).

In general you can install all optional dependencies from the Sockeye source folder using:
```bash
> pip install '.[optional]'
```

### Running sockeye

After installation, command line tools such as *sockeye-train, sockeye-translate, sockeye-average* and *sockeye-embeddings* are available.
Alternatively, if the sockeye directory is on your`$PYTHONPATH` you can run the modules directly.
For example *sockeye-train* can also be invoked as
```bash
> python -m sockeye.train <args>
```

## First Steps

For easily training popular model types on known data sets, see the [Sockeye Autopilot documentation](https://github.com/awslabs/sockeye/tree/master/contrib/autopilot).
For manually training and running translation models on your data, read on.

### Train

In order to train your first Neural Machine Translation model you will need two sets of parallel files: one for training
and one for validation. The latter will be used for computing various metrics during training.
Each set should consist of two files: one with source sentences and one with target sentences (translations).
Both files should have the same number of lines, each line containing a single
sentence. Each sentence should be a whitespace delimited list of tokens.

Say you wanted to train a RNN German-to-English translation model, then you would call sockeye like this:
```bash
> python -m sockeye.train --source sentences.de \
                       --target sentences.en \
                       --validation-source sentences.dev.de \
                       --validation-target sentences.dev.en \
                       --use-cpu \
                       --output <model_dir>
```

After training the directory *<model_dir>* will contain all model artifacts such as parameters and model
configuration. The default setting is to train a 1-layer LSTM model with attention.


### Translate

Input data for translation should be in the same format as the training data (tokenization, preprocessing scheme).
You can translate as follows:

```bash
> python -m sockeye.translate --models <model_dir> --use-cpu
```

This will take the best set of parameters found during training and then translate strings from STDIN and
write translations to STDOUT.

For more detailed examples check out our user documentation.


## Step-by-step tutorial

More detailed step-by-step tutorials can be found in the
[tutorials directory](https://github.com/awslabs/sockeye/tree/master/tutorials).

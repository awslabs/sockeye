# Sockeye

[![Documentation Status](https://readthedocs.org/projects/sockeye/badge/?version=latest)](http://sockeye.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/awslabs/sockeye.svg?branch=master)](https://travis-ci.org/awslabs/sockeye)

This package contains the Sockeye project,
a sequence-to-sequence framework for Neural Machine Translation based on MXNet.
It implements the well-known encoder-decoder architecture with attention.

If you are interested in collaborating or have any questions, please submit a pull request or issue.
You can also send questions to *sockeye-dev-at-amazon-dot-com*.
 
## Dependencies

Sockeye requires:
- **Python3**
- [MXNet-0.10.0](https://github.com/dmlc/mxnet/tree/v0.10.0)
- numpy

## Installation
You have two options for installing sockeye: pip and directly from source.
### pip

#### CPU

```bash
> pip install sockeye
```

#### GPU

If you want to run sockeye on a GPU you need to make sure your version of MXNet contains the GPU code. Depending on your
version of CUDA you can do this by running the following for CUDA 8.0:

```bash
> pip install sockeye --no-deps -r requirements.gpu-cu80.txt
```
or the following for CUDA 7.5:
```bash
> pip install sockeye --no-deps -r requirements.gpu-cu75.txt
```

### From Source

#### CPU

If you want to just use sockeye without extending it, simply install it via
```bash
> python setup.py install
```
after cloning the repository from git.

#### GPU

If you want to run sockeye on a GPU you need to make sure your version of MXNet contains the GPU code. Depending on your
version of CUDA you can do this by running the following for CUDA 8.0:

```bash
> python setup.py install -r requirements.gpu-cu80.txt
```
or the following for CUDA 7.5:
```bash
> python setup.py install -r requirements.gpu-cu75.txt
```

### Optional dependencies
In order to track learning curves during training you can optionally install dmlc's tensorboard fork
 (````pip install tensorboard````).
If you want to create alignment plots you will need to install matplotlib (````pip install matplotlib````).

In general you can install all optional dependencies from the Sockeye source folder using:
```bash
> pip install -e '.[optional]'
```

### Running sockeye

After installation, command line tools such as *sockeye-train, sockeye-translate, sockeye-average* 
and *sockeye-embeddings* are available. Alternatively, if the sockeye directory is on your
PYTHONPATH you can run the modules 
directly. For example *sockeye-train* can also be invoked as
```bash
> python -m sockeye.train <args>
```

## First Steps

### Train

In order to train your first Neural Machine Translation model you will need two sets of parallel files: one for training 
and one for validation. The latter will be used for computing various metrics during training. 
Each set should consist of two files: one with source sentences and one with target sentences (translations). Both files should have the same number of lines, each line containing a single
sentence. Each sentence should be a whitespace delimited list of tokens.

Say you wanted to train a German to English translation model, then you would call sockeye like this:
```bash
> python -m sockeye.train --source sentences.de \
                       --target sentences.en \
                       --validation-source sentences.dev.de \
                       --validation-target sentences.dev.en \
                       --use-cpu \
                       --output <model_dir>
```

After training the directory *<model_dir>* will contain all model artifacts such as parameters and model 
configuration. 


### Translate

Input data for translation should be in the same format as the training data (tokenization, preprocessing scheme).
You can translate as follows: 
 
```bash
> python -m sockeye.translate --models <model_dir> --use-cpu
```

This will take the best set of parameters found during training and then translate strings from STDIN and 
write translations to STDOUT.

For more detailed examples check out our user documentation.

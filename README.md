# Sockeye

[![Documentation Status](https://readthedocs.org/projects/sockeye/badge/?version=master)](http://sockeye.readthedocs.io/en/master/?badge=master)

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

Install them with:
```bash
> pip install -r requirements.txt
```

Optionally, dmlc's tensorboard fork is supported to track learning curves (````pip install tensorboard````).

Full dependencies are listed in requirements.txt.

## Installation

If you want to just use sockeye without extending it, simply install it via
```bash
> python setup.py install
```
after cloning the repository from git. After installation, command line tools such as
*sockeye-train, sockeye-translate, sockeye-average* 
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

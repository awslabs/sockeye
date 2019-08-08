# Sockeye

[![PyPI version](https://badge.fury.io/py/sockeye.svg)](https://badge.fury.io/py/sockeye)
[![GitHub license](https://img.shields.io/github/license/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/issues)
[![Build Status](https://travis-ci.org/awslabs/sockeye.svg?branch=master)](https://travis-ci.org/awslabs/sockeye)
[![Documentation Status](https://readthedocs.org/projects/sockeye/badge/?version=latest)](http://sockeye.readthedocs.io/en/latest/?badge=latest)

This package contains the Sockeye project, a sequence-to-sequence framework for Neural Machine Translation based on Apache MXNet (Incubating).
It implements state-of-the-art encoder-decoder architectures, such as:

- Transformer Models with self-attention [[Vaswani et al, '17](https://arxiv.org/abs/1706.03762)]

Recent developments and changes are tracked in our [CHANGELOG](CHANGELOG.md).

If you have any questions or discover problems, please [file an issue](https://github.com/awslabs/sockeye/issues/new).
You can also send questions to *sockeye-dev-at-amazon-dot-com*.

## Installation

The easiest way to run Sockeye is with [Docker](https://www.docker.com) or [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
To build a Sockeye image with all features enabled, run the build script:

```bash
python3 sockeye_contrib/docker/build.py
```

See the [Dockerfile documentation](sockeye_contrib/docker) for more information.

## Documentation

For information on how to use Sockeye, please visit [our documentation](https://awslabs.github.io/sockeye/).
Developers may be interested in our [developer guidelines](https://awslabs.github.io/sockeye/development.html).

## Citation

For technical information about Sockeye, see our paper on the arXiv ([BibTeX](sockeye.bib)):

> Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton and Matt Post. 2017.
> [Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/abs/1712.05690). ArXiv e-prints.



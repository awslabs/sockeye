---
layout: default
---

# Sockeye

[![PyPI version](https://badge.fury.io/py/sockeye.svg)](https://badge.fury.io/py/sockeye)
[![GitHub license](https://img.shields.io/github/license/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/issues)
[![Build Status](https://travis-ci.org/awslabs/sockeye.svg?branch=master)](https://travis-ci.org/awslabs/sockeye)
[![Documentation Status](https://readthedocs.org/projects/sockeye/badge/?version=latest)](http://sockeye.readthedocs.io/en/latest/?badge=latest)

This is the documentation for Sockeye, a sequence-to-sequence framework for Neural Machine Translation based on Apache MXNet Incubating.
It implements state-of-the-art encoder-decoder architectures, such as

- Deep Recurrent Neural Networks with Attention [[Bahdanau, '14](https://arxiv.org/abs/1409.0473)]
- Transformer Models with self-attention [[Vaswani et al, '17](https://arxiv.org/abs/1706.03762)]
- Fully convolutional sequence-to-sequence models [[Gehring et al, '17](https://arxiv.org/abs/1705.03122)]

In addition, this framework provides an experimental [image-to-description module](https://github.com/awslabs/sockeye/tree/master/sockeye/image_captioning) that can be used for [image captioning](image_captioning.html).

Recent developments and changes are tracked in our [CHANGELOG](https://github.com/awslabs/sockeye/blob/master/CHANGELOG.md).

If you are interested in collaborating or have any questions, please submit a pull request or [issue](https://github.com/awslabs/sockeye/issues/new). 
You can also send questions to *sockeye-dev-at-amazon-dot-com*.
Developers may be interested in [our developer guidelines](development.html).

## Citation

For technical information about Sockeye, see our paper on the arXiv ([BibTeX](sockeye.bib)):

> Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton and Matt Post. 2017.
> [Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/abs/1712.05690). ArXiv e-prints.



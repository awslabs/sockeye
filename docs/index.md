---
layout: default
---

# Sockeye

[![PyPI version](https://badge.fury.io/py/sockeye.svg)](https://badge.fury.io/py/sockeye)
[![GitHub license](https://img.shields.io/github/license/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/issues)

This is the documentation for Sockeye, a sequence-to-sequence framework for Neural Machine Translation based on Apache MXNet Incubating.
It implements state-of-the-art encoder-decoder architectures, such as

- Transformer Models with self-attention [[Vaswani et al, '17](https://arxiv.org/abs/1706.03762)]

Recent developments and changes are tracked in our [CHANGELOG](https://github.com/awslabs/sockeye/blob/master/CHANGELOG.md).

If you are interested in collaborating or have any questions, please submit a pull request or [issue](https://github.com/awslabs/sockeye/issues/new).
You can also send questions to *sockeye-dev-at-amazon-dot-com*.
Developers may be interested in [our developer guidelines](development.html).

#### Version 2.0

With version 2.0, we have updated the usage of MXNet by moving to the [Gluon API](https://mxnet.incubator.apache.org/api/python/docs/api/gluon/index.html) and adding support for several state-of-the-art features such as distributed training, low-precision training and decoding, as well as easier debugging of neural network architectures.
In the context of this rewrite, we also trimmed down the large feature set of version 1.18.x to concentrate on the most important types of models and features, to provide a maintainable framework that is suitable for fast prototyping, research, and production.
We welcome Pull Requests if you would like to help with adding back features when needed.

## Citation

For more information about Sockeye, see our papers ([BibTeX](sockeye.bib)).

##### Sockeye 2.x

> Tobias Domhan, Michael Denkowski, David Vilar, Xing Niu, Felix Hieber, Kenneth Heafield.
> [The Sockeye 2 Neural Machine Translation Toolkit at AMTA 2020](https://www.aclweb.org/anthology/2020.amta-research.10/). Proceedings of the 14th Conference of the Association for Machine Translation in the Americas (AMTA'20).

> Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar.
> [Sockeye 2: A Toolkit for Neural Machine Translation](https://www.amazon.science/publications/sockeye-2-a-toolkit-for-neural-machine-translation). Proceedings of the 22nd Annual Conference of the European Association for Machine Translation, Project Track (EAMT'20).

##### Sockeye 1.x

> Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton, Matt Post.
> [The Sockeye Neural Machine Translation Toolkit at AMTA 2018](https://www.aclweb.org/anthology/W18-1820/). Proceedings of the 13th Conference of the Association for Machine Translation in the Americas  (AMTA'18).
>
> Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton and Matt Post. 2017.
> [Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/abs/1712.05690). ArXiv e-prints.




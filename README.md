# Sockeye

[![PyPI version](https://badge.fury.io/py/sockeye.svg)](https://badge.fury.io/py/sockeye)
[![GitHub license](https://img.shields.io/github/license/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/issues)
[![Documentation Status](https://readthedocs.org/projects/sockeye/badge/?version=latest)](http://sockeye.readthedocs.io/en/latest/?badge=latest)

Sockeye is an open-source sequence-to-sequence framework for Neural Machine Translation built on [PyTorch](https://pytorch.org/). It implements distributed training and optimized inference for state-of-the-art models, powering [Amazon Translate](https://aws.amazon.com/translate/) and other MT applications. Recent developments and changes are tracked in our [CHANGELOG](https://github.com/awslabs/sockeye/blob/master/CHANGELOG.md).

For a quickstart guide to training a standard NMT model on any size of data, see the [WMT 2014 English-German tutorial](docs/tutorials/wmt_large.md).

For questions and issue reports, please [file an issue](https://github.com/awslabs/sockeye/issues/new) on GitHub.

### Version 3.0.0 & Backwards Compatibility
With version 3.0.0, Sockeye is based on PyTorch. We maintain backwards compatibility with
MXNet models in version 2.3.x a little bit longer. If MXNet 2.x is installed, Sockeye can run both with PyTorch or MXNet.

All models trained with 2.3.x (using MXNet)
can be converted to models running with PyTorch using the converter CLI (`sockeye.mx_to_pt`). This will
create a PyTorch parameter file (`<model>/params.best`) and backup the existing MXNet parameter
file to `<model>/params.best.mx`. Note that this only applies to fully-trained models that are to be used
for inference. Continued training of an MXNet model with PyTorch is not supported
(because we do not convert training and optimizer states).
`sockeye.mx_to_pt` requires MXNet to be installed into the environment.

All CLIs of Version 3.0.0 now use PyTorch by default, e.g. `sockeye-{train,translate,score}`.
MXNet-based CLIs/modules are still operational and accessible via `sockeye-{train,translate,score}-mx`.

Sockeye 3 can be installed and run without MXNet, but if installed, an extended test suite is executed to ensure
equivalence between PyTorch and MXNet models. Note that running Sockeye 3.0.0 with MXNet requires MXNet 2.x to be
installed (`pip install --pre -f https://dist.mxnet.io/python 'mxnet>=2.0.0b2021'`)

## Installation

Download the current version of Sockeye:
```bash
git clone https://github.com/awslabs/sockeye.git
```

Install the sockeye module and its dependencies:
```bash
cd sockeye && pip3 install --editable .
```

For faster GPU training, install [NVIDIA Apex](https://github.com/NVIDIA/apex). NVIDIA also provides [PyTorch Docker containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) that include Apex.

## Documentation

- For information on how to use Sockeye, please visit [our documentation](https://awslabs.github.io/sockeye/).
- Developers may be interested in our [developer guidelines](https://awslabs.github.io/sockeye/development.html).

### Older versions

- Sockeye 2.x, based on the MXNet Gluon API, is available in the `sockeye_2` branch.
- Sockeye 1.x, based on the MXNet Module API, is available in the `sockeye_1` branch.

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

## Research with Sockeye

Sockeye has been used for both academic and industrial research. A list of known publications that use Sockeye is shown below.
If you know more, please let us know or submit a pull request (last updated: August 2021).

### 2021

* Bergmanis, Toms, Mārcis Pinnis. "Facilitating Terminology Translation with Target Lemma Annotations". arXiv preprint arXiv:2101.10035 (2021)
* Briakou, Eleftheria, Marine Carpuat. "Beyond Noise: Mitigating the Impact of Fine-grained Semantic Divergences on Neural Machine Translation". arXiv preprint arXiv:2105.15087 (2021)
* Tang, Gongbo, Philipp Rönchen, Rico Sennrich, Joakim Nivre. "Revisiting Negation in Neural Machine Translation". Transactions of the Association for Computation Linguistics 9 (2021)
* Vu, Thuy, Alessandro Moschitti. "Machine Translation Customization via Automatic Training Data Selection from the Web". arXiv preprint arXiv:2102.1024 (2021)
* Xu, Weijia, Marine Carpuat. "EDITOR: An Edit-Based Transformer with Repositioning for Neural Machine Translation with Soft Lexical Constraints." Transactions of the Association for Computation Linguistics 9 (2021)
* Müller, Mathias, Rico Sennrich. "Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation". Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (2021)
* Popović, Maja, Alberto Poncelas. "On Machine Translation of User Reviews." Proceedings of RANLP (2021)
* Popović, Maja. "On nature and causes of observed MT errors." Proceedings of the 18th MT Summit (Volume 1: Research Track) (2021)
* Jain, Nishtha, Maja Popović, Declan Groves, Eva Vanmassenhove. "Generating Gender Augmented Data for NLP." Proceedings of the 3rd Workshop on Gender Bias in Natural Language Processing (2021)
* Vilar, David, Marcello Federico. "A Statistical Extension of Byte-Pair Encoding." Proceedings of IWSLT (2021)

### 2020

* Dinu, Georgiana, Prashant Mathur, Marcello Federico, Stanislas Lauly, Yaser Al-Onaizan. "Joint translation and unit conversion for end-to-end localization." Proceedings of IWSLT (2020)
* Exel, Miriam, Bianka Buschbeck, Lauritz Brandt, Simona Doneva. "Terminology-Constrained Neural Machine Translation at SAP". Proceedings of EAMT (2020).
* Hisamoto, Sorami, Matt Post, Kevin Duh. "Membership Inference Attacks on Sequence-to-Sequence Models: Is My Data In Your Machine Translation System?" Transactions of the Association for Computational Linguistics, Volume 8 (2020)
* Naradowsky, Jason, Xuan Zhan, Kevin Duh. "Machine Translation System Selection from Bandit Feedback." arXiv preprint arXiv:2002.09646 (2020)
* Niu, Xing, Prashant Mathur, Georgiana Dinu, Yaser Al-Onaizan. "Evaluating Robustness to Input Perturbations for Neural Machine Translation". arXiv preprint 	arXiv:2005.00580 (2020)
* Niu, Xing, Marine Carpuat. "Controlling Neural Machine Translation Formality with Synthetic Supervision." Proceedings of AAAI (2020)
* Keung, Phillip, Julian Salazar, Yichao Liu, Noah A. Smith. "Unsupervised Bitext Mining and Translation
via Self-Trained Contextual Embeddings." arXiv preprint arXiv:2010.07761 (2020).
* Sokolov, Alex, Tracy Rohlin, Ariya Rastrow. "Neural Machine Translation for Multilingual Grapheme-to-Phoneme Conversion." arXiv preprint arXiv:2006.14194 (2020)
* Stafanovičs, Artūrs, Toms Bergmanis, Mārcis Pinnis. "Mitigating Gender Bias in Machine Translation with Target Gender
Annotations." arXiv preprint arXiv:2010.06203 (2020)
* Stojanovski, Dario, Alexander Fraser. "Addressing Zero-Resource Domains Using Document-Level Context in Neural Machine Translation." arXiv preprint arXiv preprint arXiv:2004.14927 (2020)
* Stojanovski, Dario, Benno Krojer, Denis Peskov, Alexander Fraser. "ContraCAT: Contrastive Coreference Analytical Templates for Machine Translation". Proceedings of COLING (2020)
* Zhang, Xuan, Kevin Duh. "Reproducible and Efficient Benchmarks for Hyperparameter Optimization of Neural Machine Translation Systems." Transactions of the Association for Computational Linguistics, Volume 8 (2020)
* Swe Zin Moe, Ye Kyaw Thu, Hnin Aye Thant, Nandar Win Min, and Thepchai Supnithi, "Unsupervised Neural Machine Translation between Myanmar Sign Language and Myanmar Language", Journal of Intelligent Informatics and Smart Technology, April 1st Issue, 2020, pp. 53-61. (Submitted December 21, 2019; accepted March 6, 2020; revised March 16, 2020; published online April 30, 2020)
* Thazin Myint Oo, Ye Kyaw Thu, Khin Mar Soe and Thepchai Supnithi, "Neural Machine Translation between Myanmar (Burmese) and Dawei (Tavoyan)", In Proceedings of the 18th International Conference on Computer Applications (ICCA 2020), Feb 27-28, 2020, Yangon, Myanmar, pp. 219-227
* Müller, Mathias, Annette Rios, Rico Sennrich. "Domain Robustness in Neural Machine Translation." Proceedings of AMTA (2020)
* Rios, Annette, Mathias Müller, Rico Sennrich. "Subword Segmentation and a Single Bridge Language Affect Zero-Shot Neural Machine Translation." Proceedings of the 5th WMT: Research Papers (2020)
* Popović, Maja, Alberto Poncelas. "Neural Machine Translation between similar South-Slavic languages." Proceedings of the 5th WMT: Research Papers (2020)
* Popović, Maja, Alberto Poncelas. "Extracting correctly aligned segments from unclean parallel data using character n-gram matching." Proceedings of Conference on Language Technologies & Digital Humanities (JTDH 2020).
* Popović, Maja, Alberto Poncelas, Marija Brkic, Andy Way. "Neural Machine Translation for translating into Croatian and Serbian." Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects (2020)

### 2019

* Agrawal, Sweta, Marine Carpuat. "Controlling Text Complexity in Neural Machine Translation." Proceedings of EMNLP (2019)
* Beck, Daniel, Trevor Cohn, Gholamreza Haffari. "Neural Speech Translation using Lattice Transformations and Graph Networks." Proceedings of TextGraphs-13 (EMNLP 2019)
* Currey, Anna, Kenneth Heafield. "Zero-Resource Neural Machine Translation with Monolingual Pivot Data." Proceedings of EMNLP (2019)
* Gupta, Prabhakar, Mayank Sharma. "Unsupervised Translation Quality Estimation for Digital Entertainment Content Subtitles." IEEE International Journal of Semantic Computing (2019)
* Hu, J. Edward, Huda Khayrallah, Ryan Culkin, Patrick Xia, Tongfei Chen, Matt Post, and Benjamin Van Durme. "Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting." Proceedings of NAACL-HLT (2019)
* Rosendahl, Jan, Christian Herold, Yunsu Kim, Miguel Graça,Weiyue Wang, Parnia Bahar, Yingbo Gao and Hermann Ney “The RWTH Aachen University Machine Translation Systems for WMT 2019” Proceedings of the 4th WMT: Research Papers (2019)
* Thompson, Brian, Jeremy Gwinnup, Huda Khayrallah, Kevin Duh, and Philipp Koehn. "Overcoming catastrophic forgetting during domain adaptation of neural machine translation." Proceedings of NAACL-HLT 2019 (2019)
* Tättar, Andre, Elizaveta Korotkova, Mark Fishel “University of Tartu’s Multilingual Multi-domain WMT19 News Translation Shared Task Submission” Proceedings of 4th WMT: Research Papers (2019)
* Thazin Myint Oo, Ye Kyaw Thu and Khin Mar Soe, "Neural Machine Translation between Myanmar (Burmese) and Rakhine (Arakanese)", In Proceedings of the Sixth Workshop on NLP for Similar Languages, Varieties and Dialects, NAACL-2019, June 7th 2019, Minneapolis, United States, pp. 80-88

### 2018

* Domhan, Tobias. "How Much Attention Do You Need? A Granular Analysis of Neural Machine Translation Architectures". Proceedings of 56th ACL (2018)
* Kim, Yunsu, Yingbo Gao, and Hermann Ney. "Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies." arXiv preprint arXiv:1905.05475 (2019)
* Korotkova, Elizaveta, Maksym Del, and Mark Fishel. "Monolingual and Cross-lingual Zero-shot Style Transfer." arXiv preprint arXiv:1808.00179 (2018)
* Niu, Xing, Michael Denkowski, and Marine Carpuat. "Bi-directional neural machine translation with synthetic parallel data." arXiv preprint arXiv:1805.11213 (2018)
* Niu, Xing, Sudha Rao, and Marine Carpuat. "Multi-Task Neural Models for Translating Between Styles Within and Across Languages." COLING (2018)
* Post, Matt and David Vilar. "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation." Proceedings of NAACL-HLT (2018)
* Schamper, Julian, Jan Rosendahl, Parnia Bahar, Yunsu Kim, Arne Nix, and Hermann Ney. "The RWTH Aachen University Supervised Machine Translation Systems for WMT 2018." Proceedings of the 3rd WMT: Shared Task Papers (2018)
* Schulz, Philip, Wilker Aziz, and Trevor Cohn. "A stochastic decoder for neural machine translation." arXiv preprint arXiv:1805.10844 (2018)
* Tamer, Alkouli, Gabriel Bretschner, and Hermann Ney. "On The Alignment Problem In Multi-Head Attention-Based Neural Machine Translation." Proceedings of the 3rd WMT: Research Papers (2018)
* Tang, Gongbo, Rico Sennrich, and Joakim Nivre. "An Analysis of Attention Mechanisms: The Case of Word Sense Disambiguation in Neural Machine Translation." Proceedings of 3rd WMT: Research Papers (2018)
* Thompson, Brian, Huda Khayrallah, Antonios Anastasopoulos, Arya McCarthy, Kevin Duh, Rebecca Marvin, Paul McNamee, Jeremy Gwinnup, Tim Anderson, and Philipp Koehn. "Freezing Subnetworks to Analyze Domain Adaptation in Neural Machine Translation." arXiv preprint arXiv:1809.05218 (2018)
* Vilar, David. "Learning Hidden Unit Contribution for Adapting Neural Machine Translation Models." Proceedings of NAACL-HLT (2018)
* Vyas, Yogarshi, Xing Niu and Marine Carpuat “Identifying Semantic Divergences in Parallel Text without Annotations”. Proceedings of NAACL-HLT (2018)
* Wang, Weiyue, Derui Zhu, Tamer Alkhouli, Zixuan Gan, and Hermann Ney. "Neural Hidden Markov Model for Machine Translation". Proceedings of 56th ACL (2018)
* Zhang, Xuan, Gaurav Kumar, Huda Khayrallah, Kenton Murray, Jeremy Gwinnup, Marianna J Martindale, Paul McNamee, Kevin Duh, and Marine Carpuat. "An Empirical Exploration of Curriculum Learning for Neural Machine Translation." arXiv preprint arXiv:1811.00739 (2018)
* Swe Zin Moe, Ye Kyaw Thu, Hnin Aye Thant and Nandar Win Min, "Neural Machine Translation between Myanmar Sign Language and Myanmar Written Text", In the second Regional Conference on Optical character recognition and Natural language processing technologies for ASEAN languages 2018 (ONA 2018), December 13-14, 2018, Phnom Penh, Cambodia.
* Tang, Gongbo, Mathias Müller, Annette Rios and Rico Sennrich. "Why Self-attention? A Targeted Evaluation of Neural Machine Translation Architectures." Proceedings of EMNLP (2018)

### 2017

* Domhan, Tobias and Felix Hieber. "Using target-side monolingual data for neural machine translation through multi-task learning." Proceedings of EMNLP (2017).

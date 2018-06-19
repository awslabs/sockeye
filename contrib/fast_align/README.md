# fast_align

[[Dyer et al., 2013](http://www.aclweb.org/anthology/N13-1073)] [[GitHub Page](https://github.com/clab/fast_align)] [[Apache-2.0](https://github.com/clab/fast_align/blob/master/LICENSE.txt)]

`fast_align` is a fast and reliable word aligner that can be used to generate probabilistic lexicons.
These lexicons can then be used with Sockeye to speed up decoding by limiting each step's output layer to the most likely word-level translations of the source.

This directory contains a Dockerfile and convenience script for generating `fast_align` lexical tables from the data used to train Sockeye models.

## Running

Install [Docker](https://www.docker.com/) and run the following script to build the `fast_align` image:

```bash
./build.sh
```

Given parallel training data for Sockeye (for instance, sentences.de and sentences.en), run the following script to generate a lexical table:

```bash
 ./lex_table.sh sentences.de sentences.en lex.out
```

This file can then be converted to Sockeye lexicon using:

```bash
python -m sockeye.lexicon create --input lex.out ...
```

This requires a trained Sockeye model and the resulting lexicon is specific to that model.

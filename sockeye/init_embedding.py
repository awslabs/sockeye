#!/usr/bin/env python3

# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Initializing Sockeye embedding weights with pretrained word representations.
It also supports updating vocabulary-sized weights for a new vocabulary.

Quick usage:

    python3 -m sockeye.init_embedding        \
            -w embed-in-src.npy embed-in-tgt.npy    \
            -i vocab-in-src.json vocab-in-tgt.json   \
            -o vocab-out-src.json vocab-out-tgt.json  \
            -n source_embed_weight target_embed_weight \
            -f params.init

Optional arguments:

    --weight-files, -w
        list of input weight files in .npy, .npz or Sockeye parameter format
        .npy: a single array with shape=(vocab-in-size, embedding-size/hidden-size)
        .npz: a dictionary of {parameter_name: array}
              parameter_name is given by "--names"
        Sockeye parameter: the parameter name is given by "--names"

    --vocabularies-in, -i
        list of input vocabularies as token-index dictionaries in .json format

    --vocabularies-out, -o
        list of output vocabularies as token-index dictionaries in .json format
        They can be generated using sockeye.vocab before actual Sockeye training.

    --names, -n
        list of Sockeye parameter names for embedding weights (or other vocabulary-sized weights)
        Most common ones are source_embed_weight, target_embed_weight, source_target_embed_weight,
        target_output_weight and target_output_bias.

    Sizes of above 4 lists should be exactly the same - they are vertically aligned.

    --file, -f
        file to write initialized parameters

    --encoding, -c
        open input vocabularies with specified encoding (default: utf-8)
"""

import argparse
import sys
import logging
from typing import Dict

import numpy as np
import mxnet as mx

from sockeye.log import setup_main_logger, log_sockeye_version
from . import arguments
from . import utils
from . import vocab

logger = logging.getLogger(__name__)


def init_weight(weight: np.ndarray,
                vocab_in: Dict[str, int],
                vocab_out: Dict[str, int],
                initializer: mx.initializer.Initializer=mx.init.Constant(value=0.0)) -> mx.nd.NDArray:
    """
    Initialize vocabulary-sized weight by existing values given input and output vocabularies.

    :param weight: Input weight.
    :param vocab_in: Input vocabulary.
    :param vocab_out: Output vocabulary.
    :param initializer: MXNet initializer.
    :return: Initialized output weight.
    """
    shape = list(weight.shape)
    shape[0] = len(vocab_out)
    weight_init = mx.nd.empty(tuple(shape), dtype='float32')
    weight_desc = mx.init.InitDesc("vocabulary_sized_weight")
    initializer(weight_desc, weight_init)
    for token in vocab_out:
        if token in vocab_in:
            weight_init[vocab_out[token]] = weight[vocab_in[token]]
    return weight_init


def load_weight(weight_file: str,
                weight_name: str,
                weight_file_cache: Dict[str, Dict]) -> mx.nd.NDArray:
    """
    Load wight fron a file or the cache if it was loaded before.

    :param weight_file: Weight file.
    :param weight_name: Weight name.
    :param weight_file_cache: Cache of loaded files.
    :return: Loaded weight.
    """
    logger.info('Loading input weight file: %s', weight_file)
    if weight_file.endswith(".npy"):
        return np.load(weight_file)
    elif weight_file.endswith(".npz"):
        if weight_file not in weight_file_cache:
            weight_file_cache[weight_file] = np.load(weight_file)
        return weight_file_cache[weight_file][weight_name]
    else:
        if weight_file not in weight_file_cache:
            weight_file_cache[weight_file] = mx.nd.load(weight_file)
        return weight_file_cache[weight_file]['arg:%s' % weight_name].asnumpy()


def main():
    """
    Commandline interface to initialize Sockeye embedding weights with pretrained word representations.
    """
    raise NotImplementedError()  # TODO: re-implement for sockeye 2.0 / Gluon
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description='Quick usage: python3 -m sockeye.init_embedding '
                                                 '-w embed-in-src.npy embed-in-tgt.npy '
                                                 '-i vocab-in-src.json vocab-in-tgt.json '
                                                 '-o vocab-out-src.json vocab-out-tgt.json '
                                                 '-n source_embed_weight target_embed_weight '
                                                 '-f params.init')
    arguments.add_init_embedding_args(params)
    args = params.parse_args()
    init_embeddings(args)


def init_embeddings(args: argparse.Namespace):
    log_sockeye_version(logger)

    if len(args.weight_files) != len(args.vocabularies_in) or \
            len(args.weight_files) != len(args.vocabularies_out) or \
            len(args.weight_files) != len(args.names):
        logger.error("Exactly the same number of 'input weight files', 'input vocabularies', "
                     "'output vocabularies' and 'Sockeye parameter names' should be provided.")
        sys.exit(1)

    params = {}  # type: Dict[str, mx.nd.NDArray]
    weight_file_cache = {}  # type: Dict[str, np.ndarray]
    for weight_file, vocab_in_file, vocab_out_file, name in zip(args.weight_files, args.vocabularies_in,
                                                                args.vocabularies_out, args.names):
        weight = load_weight(weight_file, name, weight_file_cache)
        logger.info('Loading input/output vocabularies: %s %s', vocab_in_file, vocab_out_file)
        vocab_in = vocab.vocab_from_json(vocab_in_file, encoding=args.encoding)
        vocab_out = vocab.vocab_from_json(vocab_out_file)
        logger.info('Initializing parameter: %s', name)
        initializer = mx.init.Normal(sigma=np.std(weight))
        params[name] = init_weight(weight, vocab_in, vocab_out, initializer)

    logger.info('Saving initialized parameters to %s', args.file)
    #utils.save_params(params, args.file)


if __name__ == '__main__':
    main()

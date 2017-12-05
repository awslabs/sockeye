#!/usr/bin/env python3

# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

Quick usage:

    python3 -m contrib.utils.init_embedding        \
            -e embed-in-src.npy embed-in-tgt.npy    \
            -i vocab-in-src.json vocab-in-tgt.json   \
            -o vocab-out-src.json vocab-out-tgt.json  \
            -n source_embed_weight target_embed_weight \
            -f params.init

Optional arguments:

    --embeddings, -e
        list of input embedding weights in .npy format
        shape=(vocab-in-size, embedding-size)

    --vocabularies-in, -i
        list of input vocabularies as token-index dictionaries in .json format

    --vocabularies-out, -o
        list of output vocabularies as token-index dictionaries in .json format
        They can be generated using sockeye.vocab before actual Sockeye training.

    --names, -n
        list of Sockeye parameter names for embedding weights
        Most common ones are source_embed_weight, target_embed_weight and source_target_embed_weight.

    Sizes of above 4 lists should be exactly the same - they are vertically aligned.

    --file, -f
        file to write initialized parameters

    --encoding, -c
        open input vocabularies with specified encoding (default: utf-8)
"""

import argparse
import sys
from typing import Dict

import numpy as np
import mxnet as mx

from sockeye.log import setup_main_logger, log_sockeye_version
from . import arguments
from . import utils
from . import vocab

logger = setup_main_logger(__name__, console=True, file_logging=False)

def init_embedding(embed: np.ndarray,
                   vocab_in: Dict[str, int],
                   vocab_out: Dict[str, int],
                   initializer: mx.initializer.Initializer=mx.init.Constant(value=0.0)) -> mx.nd.NDArray:
    """
    Initialize embedding weight by existing word representations given input and output vocabularies.

    :param embed: Input embedding weight.
    :param vocab_in: Input vocabulary.
    :param vocab_out: Output vocabulary.
    :param initializer: MXNet initializer.
    :return: Initialized output embedding weight.
    """
    embed_init = mx.nd.empty((len(vocab_out), embed.shape[1]), dtype='float32')
    embed_desc = mx.init.InitDesc("embed_weight")
    initializer(embed_desc, embed_init)
    for token in vocab_out:
        if token in vocab_in:
            embed_init[vocab_out[token]] = embed[vocab_in[token]]
    return embed_init


def main():
    """
    Commandline interface to initialize Sockeye embedding weights with pretrained word representations.
    """
    log_sockeye_version(logger)
    params = argparse.ArgumentParser(description='Quick usage: python3 -m contrib.utils.init_embedding '
                                                 '-e embed-in-src.npy embed-in-tgt.npy '
                                                 '-i vocab-in-src.json vocab-in-tgt.json '
                                                 '-o vocab-out-src.json vocab-out-tgt.json '
                                                 '-n source_embed_weight target_embed_weight '
                                                 '-f params.init')
    arguments.add_init_embedding_args(params)
    args = params.parse_args()

    if len(args.embeddings) != len(args.vocabularies_in) or \
       len(args.embeddings) != len(args.vocabularies_out) or \
       len(args.embeddings) != len(args.names):
           logger.error("Exactly the same number of 'input embedding weights', 'input vocabularies', "
                        "'output vocabularies' and 'Sockeye parameter names' should be provided.")
           sys.exit(1)

    params = {} # type: Dict[str, mx.nd.NDArray]
    for embed_file, vocab_in_file, vocab_out_file, name in zip(args.embeddings, args.vocabularies_in, \
                                                               args.vocabularies_out, args.names):
        logger.info('Loading input embedding weight: %s', embed_file)
        embed = np.load(embed_file)
        logger.info('Loading input/output vocabularies: %s %s', vocab_in_file, vocab_out_file)
        vocab_in = vocab.vocab_from_json(vocab_in_file, encoding=args.encoding)
        vocab_out = vocab.vocab_from_json(vocab_out_file)
        logger.info('Initializing parameter: %s', name)
        initializer = mx.init.Normal(sigma=np.std(embed))
        params[name] = init_embedding(embed, vocab_in, vocab_out, initializer)

    logger.info('Saving initialized parameters to %s', args.file)
    utils.save_params(params, args.file)


if __name__ == '__main__':
    main()

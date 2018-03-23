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
Command-line tool to inspect model embeddings.
"""
import argparse
import os
import sys
from typing import Iterable, Tuple

import mxnet as mx
import numpy as np

from . import constants as C
from . import model
from . import utils
from .data_io import tokens2ids
from .log import setup_main_logger
from .utils import check_condition
from .vocab import load_source_vocabs, load_target_vocab, reverse_vocab

logger = setup_main_logger(__name__, file_logging=False)


def compute_sims(inputs: mx.nd.NDArray, normalize: bool) -> mx.nd.NDArray:
    """
    Returns a matrix with pair-wise similarity scores between inputs.
    Similarity score is (normalized) Euclidean distance. 'Similarity with self' is masked
    to large negative value.

    :param inputs: NDArray of inputs.
    :param normalize: Whether to normalize to unit-length.
    :return: NDArray with pairwise similarities of same shape as inputs.
    """
    if normalize:
        logger.info("Normalizing embeddings to unit length")
        inputs = mx.nd.L2Normalization(inputs, mode='instance')
    sims = mx.nd.dot(inputs, inputs, transpose_b=True)
    sims_np = sims.asnumpy()
    np.fill_diagonal(sims_np, -9999999.)
    sims = mx.nd.array(sims_np)
    return sims


def nearest_k(similarity_matrix: mx.nd.NDArray,
              query_word_id: int,
              k: int,
              gamma: float = 1.0) -> Iterable[Tuple[int, float]]:
    """
    Returns values and indices of k items with largest similarity.

    :param similarity_matrix: Similarity matrix.
    :param query_word_id: Query word id.
    :param k: Number of closest items to retrieve.
    :param gamma: Parameter to control distribution steepness.
    :return: List of indices and values of k nearest elements.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    values, indices = mx.nd.topk(mx.nd.softmax(similarity_matrix[query_word_id] / gamma), k=k, ret_typ='both')
    return zip(indices.asnumpy(), values.asnumpy())


def get_embedding_parameter_names(config: model.ModelConfig) -> Tuple[str, str]:
    if config.weight_tying and C.WEIGHT_TYING_SRC in config.weight_tying_type and \
            C.WEIGHT_TYING_SRC_TRG_SOFTMAX in config.weight_tying_type:
        name = "%sweight" % C.SHARED_EMBEDDING_PREFIX
        return name, name
    else:
        return "%sweight" % C.SOURCE_EMBEDDING_PREFIX, "%sweight" % C.TARGET_EMBEDDING_PREFIX


def main():
    """
    Command-line tool to inspect model embeddings.
    """
    params = argparse.ArgumentParser(description='Shows nearest neighbours of input tokens in the embedding space.')
    params.add_argument('--model', '-m', required=True,
                        help='Model folder to load config from.')
    params.add_argument('--checkpoint', '-c', required=False, type=int, default=None,
                        help='Optional specific checkpoint to load parameters from. Best params otherwise.')
    params.add_argument('--side', '-s', required=True, choices=['source', 'target'], help='what embeddings to look at')
    params.add_argument('--norm', '-n', action='store_true', help='normalize embeddings to unit length')
    params.add_argument('-k', type=int, default=5, help='Number of neighbours to print')
    params.add_argument('--gamma', '-g', type=float, default=1.0, help='Softmax distribution steepness.')
    args = params.parse_args()

    logger.info("Arguments: %s", args)

    config = model.SockeyeModel.load_config(os.path.join(args.model, C.CONFIG_NAME))
    source_embedding_name, target_embedding_name = get_embedding_parameter_names(config)

    if args.side == "source":
        vocab = load_source_vocabs(args.model)[0]
    else:
        vocab = load_target_vocab(args.model)
    vocab_inv = reverse_vocab(vocab)

    params_fname = C.PARAMS_BEST_NAME
    if args.checkpoint is not None:
        params_fname = C.PARAMS_NAME % args.checkpoint
    params, _ = utils.load_params(os.path.join(args.model, params_fname))
    if args.side == "source":
        logger.info("Loading %s", source_embedding_name)
        weights = params[source_embedding_name]
    else:
        logger.info("Loading %s", target_embedding_name)
        weights = params[target_embedding_name]
    logger.info("Embedding size: %d", weights.shape[1])

    logger.info("Computing pairwise similarities...")
    sims = compute_sims(weights, args.norm)

    # weights (vocab, num_target_embed)
    check_condition(weights.shape[0] == len(vocab),
                    "vocab and embeddings matrix do not match: %d vs. %d" % (weights.shape[0], len(vocab)))

    logger.info("Reading from STDin...")
    for line in sys.stdin:
        tokens = list(utils.get_tokens(line))
        if not tokens:
            continue
        print("Input:", line.rstrip())
        ids = tokens2ids(tokens, vocab)
        for token, token_id in zip(tokens, ids):
            print("%s id=%d" % (token, token_id))
            neighbours = nearest_k(sims, token_id, args.k, args.gamma)
            for i, (token_id, score) in enumerate(neighbours, 1):
                print("  %s id=%d sim=%.4f" % (vocab_inv[token_id], token_id, score))
        print()


if __name__ == '__main__':
    main()

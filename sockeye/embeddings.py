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
import sys
from typing import List, Tuple

import mxnet as mx
import numpy as np

import sockeye.constants as C
import sockeye.translate
import sockeye.utils
import sockeye.vocab
from sockeye.log import setup_main_logger

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
              gamma: float = 1.0) -> List[Tuple[int, float]]:
    """
    Returns values and indices of k items with largest similarity.

    :param similarity_matrix: Similarity matrix.
    :param query_word_id: Query word id.
    :param k: Number of closest items to retrieve.
    :param gamma: Parameter to control distribution steepness.
    :return: List of indices and values of k nearest elements.
    """
    values, indices = mx.nd.topk(mx.nd.softmax(similarity_matrix[query_word_id] / gamma), k=k, ret_typ='both')
    return zip(indices.asnumpy(), values.asnumpy())


def main():
    """
    Command-line tool to inspect model embeddings.
    """
    params = argparse.ArgumentParser(description='Shows nearest neighbours of input tokens in the embedding space.')
    params.add_argument('--params', '-p', required=True, help='params file to read parameters from')
    params.add_argument('--vocab', '-v', required=True, help='vocab file')
    params.add_argument('--side', '-s', required=True, choices=['source', 'target'], help='what embeddings to look at')
    params.add_argument('--norm', '-n', action='store_true', help='normalize embeddings to unit length')
    params.add_argument('-k', type=int, default=5, help='Number of neighbours to print')
    params.add_argument('--gamma', '-g', type=float, default=1.0, help='Softmax distribution steepness.')
    args = params.parse_args()

    logger.info("Arguments: %s", args)

    vocab = sockeye.vocab.vocab_from_pickle(args.vocab)
    vocab_inv = sockeye.vocab.reverse_vocab(vocab)

    params, _ = sockeye.utils.load_params(args.params)
    weights = params[C.SOURCE_EMBEDDING_PREFIX + "weight"]
    if args.side == 'target':
        weights = params[C.TARGET_EMBEDDING_PREFIX + "weight"]
    logger.info("Embedding size: %d", weights.shape[1])

    sims = compute_sims(weights, args.norm)

    # weights (vocab, num_target_embed)
    assert weights.shape[0] == len(vocab), "vocab and embeddings matrix do not match: %d vs. %d" % (
        weights.shape[0], len(vocab))

    for line in sys.stdin:
        line = line.rstrip()
        for token in line.split():
            if token not in vocab:
                sys.stdout.write("\n")
                logger.error("'%s' not in vocab", token)
                continue
            sys.stdout.write("Token: %s [%d]: " % (token, vocab[token]))
            neighbours = nearest_k(sims, vocab[token], args.k, args.gamma)
            for i, (wid, score) in enumerate(neighbours, 1):
                sys.stdout.write("%d. %s[%d] %.4f\t" % (i, vocab_inv[wid], wid, score))
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == '__main__':
    main()

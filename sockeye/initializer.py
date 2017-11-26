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

import logging
from typing import Optional

import mxnet as mx
import numpy as np

import sockeye.constants as C

logger = logging.getLogger(__name__)


def get_initializer(default_init_type: str, default_init_scale: float, default_init_xavier_factor_type: str,
                    embed_init_type: str, embed_init_sigma: float,
                    rnn_init_type: str) -> mx.initializer.Initializer:
    """
    Returns a mixed MXNet initializer.

    :param default_init_type: The default weight initializer type.
    :param default_init_scale: The scale used for default weight initialization (only used with uniform initialization).
    :param default_init_xavier_factor_type: Xavier factor type.
    :param embed_init_type: Embedding matrix initialization type.
    :param embed_init_sigma: Sigma for normal initialization of embedding matrix.
    :param rnn_init_type: Initialization type for RNN h2h matrices.
    :return: Mixed initializer.
    """
    # default initializer
    if default_init_type == C.INIT_XAVIER:
        default_init = [(C.DEFAULT_INIT_PATTERN,
                         mx.init.Xavier(rnd_type="uniform",
                                        factor_type=default_init_xavier_factor_type,
                                        magnitude=default_init_scale))]
    elif default_init_type == C.INIT_UNIFORM:
        default_init = [(C.DEFAULT_INIT_PATTERN, mx.init.Uniform(scale=default_init_scale))]
    else:
        raise ValueError("Unknown default initializer %s." % default_init_type)

    # embedding initializer
    if embed_init_type == C.EMBED_INIT_NORMAL:
        embed_init = [(C.EMBED_INIT_PATTERN, mx.init.Normal(sigma=embed_init_sigma))]
    elif embed_init_type == C.EMBED_INIT_DEFAULT:
        embed_init = []
    else:
        raise ValueError('Unknown embedding initializer: %s' % embed_init_type)

    # rnn initializer
    if rnn_init_type == C.RNN_INIT_ORTHOGONAL:
        rnn_init = [(C.RNN_INIT_PATTERN, mx.initializer.Orthogonal())]
    elif rnn_init_type == C.RNN_INIT_ORTHOGONAL_STACKED:
        rnn_init = [(C.RNN_INIT_PATTERN, StackedOrthogonalInit(scale=1.0, rand_type="eye"))]
    elif rnn_init_type == C.RNN_INIT_DEFAULT:
        rnn_init = []
    else:
        raise ValueError('Unknown RNN initializer: %s' % rnn_init_type)

    params_init_pairs = embed_init + rnn_init + default_init
    return mx.initializer.Mixed(*zip(*params_init_pairs))


@mx.init.register
class StackedOrthogonalInit(mx.initializer.Initializer):
    """
    Initializes weight as Orthogonal matrix. Here we assume that the weight consists of stacked square matrices of
    the same size.
    For example one could have 3 (2,2) matrices resulting in a (6,2) matrix. This situation arises in RNNs when one
    wants to perform multiple h2h transformations in a single matrix multiplication.

    Reference:
    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    arXiv preprint arXiv:1312.6120 (2013).

    :param scale: Scaling factor of weight.
    :param rand_type: use "uniform" or "normal" random number to initialize weight.
           "eye" simply sets the matrix to an identity matrix.

    """

    def __init__(self, scale=1.414, rand_type="uniform"):
        super().__init__()
        self.scale = scale
        self.rand_type = rand_type

    def _init_weight(self, sym_name, arr):
        assert len(arr.shape) == 2, "Only 2d weight matrices supported."
        base_dim = arr.shape[1]
        stacked_dim = arr.shape[0]  # base_dim * num_sub_matrices
        assert stacked_dim % base_dim == 0, \
            "Dim1 must be a multiple of dim2 (as weight = stacked square matrices)."

        num_sub_matrices = stacked_dim // base_dim
        logger.info("Initializing weight %s (shape=%s, num_sub_matrices=%d) with an orthogonal weight matrix.",
                    sym_name, arr.shape, num_sub_matrices)

        for mat_idx in range(0, num_sub_matrices):
            if self.rand_type == "uniform":
                tmp = np.random.uniform(-1.0, 1.0, (base_dim, base_dim))
                _, __, q = np.linalg.svd(tmp)
            elif self.rand_type == "normal":
                tmp = np.random.normal(0.0, 1.0, (base_dim, base_dim))
                _, __, q = np.linalg.svd(tmp)
            elif self.rand_type == "eye":
                q = np.eye(base_dim)
            else:
                raise ValueError("unknown rand_type %s" % self.rand_type)
            q = self.scale * q
            arr[mat_idx * base_dim:mat_idx * base_dim + base_dim] = q

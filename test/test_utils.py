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

import sockeye.utils
import numpy as np
import mxnet as mx
import numpy as np


def test_get_alignments():
    attention_matrix = np.asarray([[0.1, 0.4, 0.5],
                                   [0.2, 0.8, 0.0],
                                   [0.4, 0.4, 0.2]])
    test_cases = [(0.5, [(1, 1)]),
                  (0.8, []),
                  (0.1, [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)])]

    for threshold, expected_alignment in test_cases:
        alignment = list(sockeye.utils.get_alignments(attention_matrix, threshold=threshold))
        assert alignment == expected_alignment


def gaussian_vector(shape, return_symbol=False):
    """
    Generates random normal tensors (diagonal covariance)
    
    :param shape: shape of the tensor.
    :param return_symbol: True if the result should be a Symbol, False if it should be an Numpy array.
    :return: A gaussian tensor.
    """
    return mx.sym.random_normal(shape=shape) if return_symbol else np.random.normal(size=shape)


def integer_vector(shape, max_value, return_symbol=False):
    """
    Generates a random positive integer tensor
    
    :param shape: shape of the tensor.
    :param max_value: maximum integer value.
    :param return_symbol: True if the result should be a Symbol, False if it should be an Numpy array.
    :return: A random integer tensor.
    """
    return mx.sym.round(mx.sym.random_uniform(shape=shape) * max_value) if return_symbol \
        else np.round(np.random.uniform(size=shape) * max_value)


def uniform_vector(shape, min_value=0, max_value=1, return_symbol=False):
    """
    Generates a uniformly random tensor
    
    :param shape: shape of the tensor
    :param min_value: minimum possible value
    :param max_value: maximum possible value (exclusive)
    :param return_symbol: True if the result should be a mx.sym.Symbol, False if it should be a Numpy array
    :return: 
    """
    return mx.sym.random_uniform(low=min_value, high=max_value, shape=shape) if return_symbol \
        else np.random.uniform(low=min_value, high=max_value, size=shape)

# Copyright 2018--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest

import mxnet as mx
import numpy as np

import sockeye.transformer
import sockeye.constants as C


def test_auto_regressive_bias_dtype():
    block = sockeye.transformer.AutoRegressiveBias()
    block.initialize()
    length = 10
    dtype = 'float32'
    data = mx.nd.ones((2, length, 10), dtype=dtype)
    bias = block(data)
    assert bias.dtype == np.float32

    dtype = 'float16'
    block.cast(dtype)
    bias = block(data.astype(dtype))
    assert bias.dtype == np.float16
    assert bias.min().asscalar() == -C.LARGE_VALUES[dtype]


def test_auto_regressive_bias_output():
    block = sockeye.transformer.AutoRegressiveBias()
    block.initialize()
    length = 2
    data = mx.nd.ones((2, length, 10), dtype='float32')
    bias = block(data)

    expected = np.array([[0.0, -1.0e8], [0.0, 0.0]]).reshape((1, 2, 2))
    np.testing.assert_array_equal(bias.asnumpy(), expected)


@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_transformer_feed_forward(use_glu):
    block = sockeye.transformer.TransformerFeedForward(num_hidden=2,
                                                       num_model=2,
                                                       act_type=C.RELU,
                                                       dropout=0.1,
                                                       dtype=C.DTYPE_FP32,
                                                       prefix='ff_',
                                                       use_glu=use_glu)
    block.initialize()
    block.hybridize()

    data = mx.nd.ones((1, 10, 2), dtype=C.DTYPE_FP32)
    block(data)

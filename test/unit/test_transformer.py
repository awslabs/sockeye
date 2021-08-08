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

from mxnet import np
import numpy as onp

import sockeye.transformer
import sockeye.constants as C


def test_auto_regressive_bias_dtype():
    block = sockeye.transformer.AutoRegressiveBias()
    block.initialize()
    length = 10
    dtype = 'float32'
    data = np.ones((2, length, 10), dtype=dtype)
    bias = block(data)
    assert bias.dtype == np.float32

    dtype = 'float16'
    block.cast(dtype)
    bias = block(data.astype(dtype))
    assert bias.dtype == np.float16
    assert bias.min().item() == -C.LARGE_VALUES[dtype]


def test_auto_regressive_bias_output():
    block = sockeye.transformer.AutoRegressiveBias()
    block.initialize()
    length = 2
    data = np.ones((2, length, 10), dtype='float32')
    bias = block(data)

    expected = np.array([[0.0, -1.0e8], [0.0, 0.0]]).reshape((1, 2, 2))
    onp.testing.assert_array_equal(bias, expected)


@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_transformer_feed_forward(use_glu):
    block = sockeye.transformer.TransformerFeedForward(num_hidden=2,
                                                       num_model=2,
                                                       act_type=C.RELU,
                                                       dropout=0.1,
                                                       dtype=C.DTYPE_FP32,
                                                       use_glu=use_glu)
    block.initialize()
    block.hybridize()

    data = np.ones((1, 10, 2), dtype=C.DTYPE_FP32)
    block(data)


import torch as pt

@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_pt_mx_eq_transformer_feed_forward(use_glu):
    b_mx = sockeye.transformer.TransformerFeedForward(num_hidden=4,
                                                       num_model=2,
                                                       act_type=C.RELU,
                                                      dropout=0.0,
                                                      dtype=C.DTYPE_FP32,
                                                      use_glu=use_glu)
    b_mx.initialize()

    b_pt = sockeye.transformer.PyTorchFeedForward(num_hidden=4,
                                                  num_model=2,
                                                  act_type=C.RELU,
                                                  dropout=0.0,
                                                  dtype=C.DTYPE_FP32,
                                                  use_glu=use_glu)

    # use mxnet parameter initializations for pytorch block
    b_pt.ff1.weight[:] = pt.as_tensor(b_mx.ff1.weight.data().asnumpy())
    b_pt.ff2.weight[:] = pt.as_tensor(b_mx.ff2.weight.data().asnumpy())
    b_pt.ff1.bias[:] = pt.as_tensor(b_mx.ff1.bias.data().asnumpy())
    b_pt.ff2.bias[:] = pt.as_tensor(b_mx.ff2.bias.data().asnumpy())
    if use_glu:
        b_pt.linear.weight[:] = pt.as_tensor(b_mx.linear.weight.data().asnumpy())
        b_pt.linear.bias[:] = pt.as_tensor(b_mx.linear.bias.data().asnumpy())

    result_mx = b_mx(np.ones((2, 2, 2))).asnumpy()
    result_pt = b_pt(pt.ones(2, 2, 2)).detach().numpy()

    assert np.allclose(result_mx, result_pt)


@pytest.mark.parametrize('length', [1, 10, 100])
def test_pt_mx_eq_autoregressive_bias(length):
    x_mx = np.zeros((2, length, 32))
    x_pt = pt.zeros(2, length, 32)

    b_mx = sockeye.transformer.AutoRegressiveBias()
    b_mx.initialize()
    b_pt = sockeye.transformer.PyTorchAutoRegressiveBias()

    result_mx = b_mx(x_mx).asnumpy()
    result_pt = b_pt(x_pt).detach().numpy()

    assert np.allclose(result_mx, result_pt)


@pytest.mark.parametrize('sequence', ['rn', 'nr', 'r', 'n', ''])  # not testing dropout
def test_pt_mx_eq_transformer_process_block(sequence):
    num_hidden = 32
    x_mx = np.zeros((2, 10, num_hidden))
    prev_mx = np.ones((2, 10, num_hidden))
    x_pt = pt.zeros(2, 10, num_hidden)
    prev_pt = pt.ones(2, 10, num_hidden)

    sequence = 'nr'
    b_mx = sockeye.transformer.TransformerProcessBlock(sequence, 0.0, num_hidden)
    b_mx.initialize()
    b_pt = sockeye.transformer.PyTorchTransformerProcessBlock(sequence, 0.0, num_hidden)

    result_mx = b_mx(x_mx, prev_mx).asnumpy()
    result_pt = b_pt(x_pt, prev_pt).detach().numpy()

    assert np.allclose(result_mx, result_pt)
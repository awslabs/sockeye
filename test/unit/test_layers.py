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

import mxnet as mx
import numpy as np

import sockeye.layers
import sockeye.rnn


def test_layer_normalization():
    batch_size = 32
    other_dim = 10
    num_hidden = 64
    x_nd = mx.nd.uniform(0, 10, (batch_size, other_dim, num_hidden))
    x_np = x_nd.asnumpy()

    ln = sockeye.layers.LayerNormalization(prefix="")
    ln.initialize()

    expected_mean = np.mean(x_np, axis=-1, keepdims=True)
    expected_var = np.var(x_np, axis=-1, keepdims=True)
    expected_norm = (x_np - expected_mean) / np.sqrt(expected_var)

    norm = ln(x_nd)
    assert np.isclose(norm.asnumpy(), expected_norm, atol=1.e-6).all()
    ln.hybridize()
    norm = ln(x_nd)
    assert np.isclose(norm.asnumpy(), expected_norm, atol=1.e-6).all()


def test_lhuc():
    num_hidden = 50
    batch_size = 10

    inp = mx.sym.Variable("inp")
    params = mx.sym.Variable("params")
    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden, weight=params)
    with_lhuc = lhuc(inp)

    inp_nd = mx.nd.random_uniform(shape=(batch_size, num_hidden))
    params_same_nd = mx.nd.zeros(shape=(num_hidden,))
    params_double_nd = mx.nd.ones(shape=(num_hidden,)) * 20

    out_same = with_lhuc.eval(inp=inp_nd, params=params_same_nd)[0]
    assert np.isclose(inp_nd.asnumpy(), out_same.asnumpy()).all()

    out_double = with_lhuc.eval(inp=inp_nd, params=params_double_nd)[0]
    assert np.isclose(2 * inp_nd.asnumpy(), out_double.asnumpy()).all()


def test_weight_normalization():
    # The norm after the operation should be equal to the scale factor.
    expected_norm = np.asarray([1., 2.])
    scale_factor = mx.nd.array([[1.], [2.]])
    weight = mx.sym.Variable("weight")
    weight_norm = sockeye.layers.WeightNormalization(weight,
                                                     num_hidden=2)
    norm_weight = weight_norm()
    nd_norm_weight = norm_weight.eval(weight=mx.nd.array([[1., 2.],
                                                          [3., 4.]]),
                                      wn_scale=scale_factor)
    assert np.isclose(np.linalg.norm(nd_norm_weight[0].asnumpy(), axis=1), expected_norm).all()


def test_length_ratio_average_sources():
    # sources: (n=3, length=5, hidden_size=2)
    sources = mx.nd.array([[[1, 5],
                            [2, 6],
                            [3, 7],
                            [4, 8],
                            [0, 9]],
                          [[10, 0],
                            [9, 1],
                            [8, 3],
                            [7, 5],
                            [0, 7]],
                          [[-1, 0],
                           [-1, 0],
                           [-1, 0],
                           [0, -1],
                           [0, -1]]])
    lengths = mx.nd.array([3, 4, 5])
    expected_averages = np.array([[2., 6.], [8.5, 2.25], [-0.6, -0.4]])

    average = sockeye.layers.LengthRatio.average_sources(mx.sym.Variable('sources'),
                                                         mx.sym.Variable('lengths'))
    average = average.eval(sources=sources, lengths=lengths)[0]
    assert np.isclose(average.asnumpy(), expected_averages).all()


def test_length_ratio():
    # sources: (n=3, length=5, hidden_size=2)
    sources = mx.nd.array([[[1, 6],
                            [2, 7],
                            [3, 8],
                            [4, 9],
                            [5, 10]],
                          [[10, 5],
                            [9, 4],
                            [8, 3],
                            [7, 2],
                            [6, 1]],
                          [[-1, 1],
                           [-1, 0],
                           [-1, 2],
                           [-1, -2],
                           [1, 1]]])
    lengths = mx.nd.array([5, 5, 4])
    expected_averages = np.array([[3., 8.], [8., 3.], [-1., 0.25]])
    weight = mx.nd.array([[1.1, 1.3]])
    bias = mx.nd.array([8])

    length_ratio = sockeye.layers.LengthRatio(hidden_size=2, num_layers=1, prefix="lr_")

    data = length_ratio(mx.sym.Variable('sources'), mx.sym.Variable('lengths'))
    ratio = data.eval(sources=sources, lengths=lengths,
                      lr_dense0_weight=weight, lr_dense0_bias=bias)[0]

    average = sockeye.layers.LengthRatio.average_sources(mx.sym.Variable('sources'),
                                                         mx.sym.Variable('lengths')).eval(sources=sources,
                                                                                          lengths=lengths)[0]
    assert np.isclose(average.asnumpy(), expected_averages).all()

    softrelu = lambda x: np.log(1 + np.exp(x))
    expected_softrelu = softrelu(np.dot(expected_averages, weight.asnumpy().T) + bias.asnumpy())

    assert np.isclose(ratio.asnumpy(), expected_softrelu).all()

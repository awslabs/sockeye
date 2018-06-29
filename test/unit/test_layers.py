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
    x = mx.sym.Variable('x')
    x_nd = mx.nd.uniform(0, 10, (batch_size, other_dim, num_hidden))
    x_np = x_nd.asnumpy()

    ln = sockeye.layers.LayerNormalization(prefix="")

    expected_mean = np.mean(x_np, axis=-1, keepdims=True)
    expected_var = np.var(x_np, axis=-1, keepdims=True)
    expected_norm = (x_np - expected_mean) / np.sqrt(expected_var)

    norm = ln(x).eval(x=x_nd,
                    _gamma=mx.nd.ones((num_hidden,)),
                    _beta=mx.nd.zeros((num_hidden,)))[0]

    assert np.isclose(norm.asnumpy(), expected_norm, atol=1.e-6).all()


def test_lhuc():
    num_hidden = 50
    batch_size = 10

    inp = mx.sym.Variable("inp")
    params = mx.sym.Variable("params")
    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden, weight=params)
    with_lhuc = lhuc(inputs=inp)

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

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


def test_compare_dot_atts():
    batch = 3
    lq = 4
    lk = 5
    depth = 32
    heads = 8
    depth_per_head = depth // heads

    def sym_old():
        q = mx.sym.Variable('q')
        k = mx.sym.Variable('k')
        v = mx.sym.Variable('v')

        q = sockeye.layers.split_heads(q, depth_per_head, heads, fold_heads=True)
        k = sockeye.layers.split_heads(k, depth_per_head, heads, fold_heads=True)
        v = sockeye.layers.split_heads(v, depth_per_head, heads, fold_heads=True)

        contexts = sockeye.layers.dot_attention(q, k, v)
        # (batch, query_max_length, depth)
        contexts = sockeye.layers.combine_heads(contexts, depth_per_head, heads, folded_heads=True)
        return contexts

    def sym_new():
        q = mx.sym.Variable('q')
        k = mx.sym.Variable('k')
        v = mx.sym.Variable('v')

        q = sockeye.layers.split_heads(q, depth_per_head, heads, fold_heads=False)
        k = sockeye.layers.split_heads(k, depth_per_head, heads, fold_heads=False)
        v = sockeye.layers.split_heads(v, depth_per_head, heads, fold_heads=False)

        contexts = sockeye.layers.multi_head_dot_attention(q, k, v)
        # (batch, query_max_length, depth)
        contexts = sockeye.layers.combine_heads(contexts, depth_per_head, heads, folded_heads=False)
        return contexts

    data_q = mx.nd.random.uniform(0, 1, (batch, lq, depth))
    data_k = mx.nd.random.uniform(0, 1, (batch, lk, depth))
    data_v = data_k.copy()
    data_lk = mx.nd.array(np.random.randint(1, lk, (batch,)))
    print(data_lk)

    # att
    # batch, lk
    new = sym_new().eval(q=data_q, k=data_k, v=data_v, l=data_lk)[0].asnumpy()
    old = sym_old().eval(q=data_q, k=data_k, v=data_v, l=data_lk)[0].asnumpy()
    assert np.allclose(new, old)

    # self-att autoregressive
    new = sym_new().eval(q=data_q, k=data_q, v=data_q)[0].asnumpy()
    old = sym_old().eval(q=data_q, k=data_q, v=data_q)[0].asnumpy()
    assert np.allclose(new, old)

    # self-att variable length
    new = sym_new().eval(q=data_k, k=data_k, v=data_k)[0].asnumpy()
    old = sym_old().eval(q=data_k, k=data_k, v=data_k)[0].asnumpy()
    assert np.allclose(new, old)

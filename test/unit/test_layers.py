# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from mxnet import np
import numpy as onp

import sockeye.layers


def test_lhuc():
    num_hidden = 50
    batch_size = 10
    inp = np.random.uniform(0, 1, size=(batch_size, num_hidden))

    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden, weight_init='zeros')
    lhuc.initialize()
    out = lhuc(inp)
    assert onp.allclose(inp, out)

    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden, weight_init=mx.init.Constant(value=20.0))
    lhuc.initialize()
    out = lhuc(inp)
    assert onp.allclose(2 * inp, out)


def test_weight_normalization():
    expected_norm = np.array([1., 1.])
    weight = np.array([[1., 2.],
                       [3., 4.]])
    weight_norm = sockeye.layers.WeightNormalization(num_hidden=2)
    weight_norm.initialize()
    norm_weight = weight_norm(weight)
    assert onp.allclose(np.linalg.norm(norm_weight, axis=1), expected_norm)


def test_positional_embeddings():
    num_embed = 32
    max_seq_len = 10
    scale_up_input = False
    scale_down_positions = False
    data_len = 5
    data = np.zeros((2, data_len, num_embed))

    # fixed embeddings
    expected_fixed_embedding = sockeye.layers.get_positional_embeddings(data_len, num_embed)
    b = sockeye.layers.PositionalEmbeddings(weight_type='fixed',
                                            num_embed=num_embed,
                                            max_seq_len=max_seq_len,
                                            scale_up_input=scale_up_input,
                                            scale_down_positions=scale_down_positions,
                                            weight_init=None)
    b.initialize()
    # no steps
    out = b(data, None)
    assert np.allclose(out[0], expected_fixed_embedding)
    assert np.allclose(out[1], expected_fixed_embedding)

    # steps
    steps = np.expand_dims(np.array([2, 3]), axis=1)
    out = b(data, steps)
    assert onp.allclose(out[0], expected_fixed_embedding[2])
    assert onp.allclose(out[1], expected_fixed_embedding[3])

    # learned embeddings
    b = sockeye.layers.PositionalEmbeddings(weight_type='learned',
                                            num_embed=num_embed,
                                            max_seq_len=max_seq_len,
                                            scale_up_input=scale_up_input,
                                            scale_down_positions=scale_down_positions,
                                            weight_init='ones')
    b.initialize()
    expected_learned_embeddings = np.ones((data_len, num_embed))
    out = b(data, None)
    assert onp.allclose(out[0], expected_learned_embeddings)


def test_output_layer():
    num_hidden = 32
    vocab_size = 64
    data = np.ones((2, 10, num_hidden))
    vocab_slice_ids = np.array([4, 7, 23])

    b = sockeye.layers.OutputLayer(num_hidden, vocab_size)
    b.initialize()
    assert b.weight.data().shape == (vocab_size, num_hidden)

    output = b(data, None)
    assert output.shape == (2, 10, vocab_size)
    reduced_output = output.take(vocab_slice_ids, axis=-1)

    output_restricted = b(data, vocab_slice_ids)
    assert output_restricted.shape == (2, 10, len(vocab_slice_ids))

    assert onp.allclose(output_restricted, reduced_output)

    b.hybridize()
    output = b(data, None)
    assert output.shape == (2, 10, vocab_size)
    reduced_output = output.take(vocab_slice_ids, axis=-1)

    output_restricted = b(data, vocab_slice_ids)
    assert output_restricted.shape == (2, 10, len(vocab_slice_ids))

    assert onp.allclose(output_restricted, reduced_output)

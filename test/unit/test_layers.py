# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from math import pow, sqrt
import numpy as np
import pytest
import torch as pt

import sockeye.constants as C
import sockeye.layers
import sockeye.transformer
from sockeye.knn import KNNConfig, FaissIndexBuilder

# Only run certain tests in this file if faiss is installed
try:
    import faiss  # pylint: disable=E0401
    faiss_installed = True
except:
    faiss_installed = False


def test_lhuc():
    num_hidden = 50
    batch_size = 10
    inp = pt.rand(batch_size, num_hidden)

    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden)
    pt.nn.init.zeros_(lhuc.weight)
    out = lhuc(inp)
    pt.testing.assert_close(inp, out)

    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden)
    pt.nn.init.constant_(lhuc.weight, 20.0)
    out = lhuc(inp)
    pt.testing.assert_close(2 * inp, out)


def test_positional_embeddings():
    num_embed = 32
    max_seq_len = 10
    scale_up_input = False
    scale_down_positions = False
    data_len = 5
    data = pt.zeros(2, data_len, num_embed)

    # fixed embeddings
    expected_fixed_embedding = sockeye.layers.get_positional_embeddings(data_len, num_embed)
    b = sockeye.layers.PositionalEmbeddings(weight_type='fixed',
                                            num_embed=num_embed,
                                            max_seq_len=max_seq_len,
                                            scale_up_input=scale_up_input,
                                            scale_down_positions=scale_down_positions)
    # no steps
    out = b(data, None)
    pt.testing.assert_close(out[0], expected_fixed_embedding)
    pt.testing.assert_close(out[1], expected_fixed_embedding)

    # steps
    steps = pt.tensor([2, 3, 1, 1, 1]).unsqueeze(0)
    out = b(data, steps)
    pt.testing.assert_close(out[0, 0], expected_fixed_embedding[2])
    pt.testing.assert_close(out[1, 0], expected_fixed_embedding[2])
    pt.testing.assert_close(out[0, 1], expected_fixed_embedding[3])
    pt.testing.assert_close(out[1, 1], expected_fixed_embedding[3])
    pt.testing.assert_close(out[0, 2], expected_fixed_embedding[1])
    pt.testing.assert_close(out[1, 2], expected_fixed_embedding[1])

    # learned embeddings
    b = sockeye.layers.PositionalEmbeddings(weight_type='learned',
                                            num_embed=num_embed,
                                            max_seq_len=max_seq_len,
                                            scale_up_input=scale_up_input,
                                            scale_down_positions=scale_down_positions)
    pt.nn.init.constant_(b.weight, val=1.0)
    expected_learned_embeddings = pt.ones(data_len, num_embed)
    out = b(data, None)
    pt.testing.assert_close(out[0], expected_learned_embeddings)


def test_output_layer():
    num_hidden = 32
    vocab_size = 64
    data = pt.ones(2, 10, num_hidden)
    vocab_slice_ids = pt.tensor([4, 7, 23])

    b = sockeye.layers.OutputLayer(num_hidden, vocab_size)
    assert b.weight.data.shape == (vocab_size, num_hidden)

    output = b(data, None)
    assert output.shape == (2, 10, vocab_size)
    reduced_output = output.index_select(-1, vocab_slice_ids)

    output_restricted = b(data, vocab_slice_ids)
    assert output_restricted.shape == (2, 10, len(vocab_slice_ids))

    pt.testing.assert_close(output_restricted, reduced_output, equal_nan=True)


@pytest.mark.parametrize('qlen, kvlen, batch_size, hidden, heads',
                         [(10, 9, 1, 12, 4), (1, 1, 2, 4, 1), (3, 32, 15, 64, 8),
                          (10, 32, 15, 32, 8), (1, 1, 1, 1, 1)])
def test_interleaved_multihead_attention(qlen, kvlen, batch_size, hidden, heads):
    queries_pt = pt.rand((qlen, batch_size, hidden))
    memory_pt = pt.rand((kvlen, batch_size, hidden))

    # test without mask
    mha = sockeye.layers.MultiHeadAttention(hidden, heads, hidden, dropout=0.0, depth_key_value=hidden)
    mha.train()
    assert not mha.kv_interleaved
    r_train = mha(queries_pt, memory_pt, mask=None, projected_memory_kv=None)
    mha.eval()
    assert mha.kv_interleaved
    r_test = mha(queries_pt, memory_pt, mask=None, projected_memory_kv=None)
    assert pt.allclose(r_train, r_test, atol=1e-06)

    # test with mask
    valid_length = pt.randint(1, kvlen + 1, (batch_size,))
    mask = sockeye.layers.prepare_source_length_mask(valid_length, heads, kvlen)
    mask = mask.repeat(1, qlen, 1)  # Shape: (batch *h heads, qlen, kvlen)
    mha.train()
    assert not mha.kv_interleaved
    r_train = mha(queries_pt, memory_pt, mask=mask, projected_memory_kv=None)
    mha.eval()
    assert mha.kv_interleaved
    r_test = mha(queries_pt, memory_pt, mask=mask, projected_memory_kv=None)
    assert pt.allclose(r_train, r_test, atol=1e-06)


@pytest.mark.parametrize('seq_len, batch_size, hidden, heads, side',
                         [(10, 1, 12, 4, 'decoder'), (1, 2, 4, 1, 'decoder'), (3, 15, 64, 8, 'decoder'),
                          (10, 1, 12, 4, 'encoder'), (1, 2, 4, 1, 'encoder'), (3, 15, 64, 8, 'encoder'),
                          (96, 32, 32, 8, 'encoder'), (96, 32, 32, 8, 'decoder')])
def test_interleaved_multihead_self_attention(seq_len, batch_size, hidden, heads, side):
    inputs = pt.rand((seq_len, batch_size, hidden))

    # test without attention masking
    mha = sockeye.layers.MultiHeadSelfAttention(hidden, heads, hidden, dropout=0.0)
    mha.train()
    assert not mha.kv_interleaved
    r_train, _ = mha(inputs, previous_states=None, mask=None)
    mha.eval()
    assert mha.kv_interleaved
    r_test, _ = mha(inputs, previous_states=None, mask=None)
    assert pt.allclose(r_train, r_test, atol=1e-06)

    # test with two types of attention masks (autoregressive, and valid_length based)
    if side == 'decoder':
        # autoregressive mask. Shape: (len, len)
        mask = sockeye.transformer.AutoRegressiveMask()(inputs.transpose(0, 1))
        mha.train()
        assert not mha.kv_interleaved
        r_train, _ = mha(inputs, previous_states=None, mask=mask)
        mha.eval()
        assert mha.kv_interleaved
        r_test, _ = mha(inputs, previous_states=None, mask=mask)
        assert pt.allclose(r_train, r_test, atol=1e-06)
    elif side == 'encoder':
        valid_length = pt.randint(1, seq_len + 1, (batch_size,))
        # source attention mask. Shape: (batch * heads, 1, seq_len)
        mask = sockeye.layers.prepare_source_length_mask(valid_length, heads, seq_len)
        mask = mask.repeat(1, seq_len, 1)  # Shape: (batch * heads, seq_len, seq_len)
        mha.train()
        assert not mha.kv_interleaved
        r_train, _ = mha(inputs, previous_states=None, mask=mask)
        mha.eval()
        assert mha.kv_interleaved
        r_test, _ = mha(inputs, previous_states=None,
                        mask=mask)  # Note: can also handle the mask repated on the qlen axis
        assert pt.allclose(r_train, r_test, atol=1e-06)


@pytest.mark.skipif(not faiss_installed, reason='Faiss is not installed')
def test_knn_layer():
    num_data_points = 16
    num_dimensions = 16
    assert num_dimensions > 4  # there are at least 4 items in a vocabulary

    config = KNNConfig(num_data_points, num_dimensions, 'float32', 'int32', "Flat", -1)
    builder = FaissIndexBuilder(config)
    index = builder.init_faiss_index()

    # build data
    states = np.outer(np.arange(num_data_points, dtype=np.float32), np.ones(num_dimensions, dtype=np.float32))
    words = np.arange(num_data_points + 1, dtype=np.int32) - 1  # need to prepend a <s> at the beginning
    words[0] = 0
    words = np.expand_dims(words, axis=1)
    builder.add_items(index, states)

    # in case BOS and/or EOS ID are changed, the test should be revisited to make sure no overflow/underflow occurs
    assert C.BOS_ID + 2 < num_data_points - 1
    assert C.EOS_ID < C.BOS_ID + 2 or C.EOS_ID > num_data_points

    def build_gld_probs(offset):
        gld_dists = pt.sqrt(pt.FloatTensor([pow(1 + offset, 2) * num_dimensions,
                                            pow(offset, 2) * num_dimensions,
                                            pow(1 - offset, 2) * num_dimensions]))
        gld_logits = pt.exp(-gld_dists)
        gld_probs = gld_logits.div_(pt.sum(gld_logits))
        return gld_probs

    def query_test(knn_layer, offset):
        for i in range(C.BOS_ID + 2, num_data_points - 1):
            query = np.expand_dims(states[i], axis=0) + offset
            query = pt.from_numpy(query)
            probs = knn_layer(query)

            logits_idxs = pt.LongTensor(list(range(i-1, i+2)))
            gld_probs = pt.zeros(1, num_dimensions)
            gld_probs[0, logits_idxs] = build_gld_probs(offset)
            assert pt.allclose(probs, gld_probs)

    # test when inexact distances are used
    knn_layer = sockeye.layers.KNN(index, words, 16, 3, 1)
    query_test(knn_layer, 0.1)

    # test when exact distances are used
    knn_layer = sockeye.layers.KNN(index, words, 16, 3, 1, states)
    query_test(knn_layer, 0.1)

    # test BOS case & scatter_add
    assert C.BOS_ID == 2
    assert C.EOS_ID == 3
    offset = 0.1
    query = np.expand_dims(states[C.BOS_ID], axis=0) + offset
    query = pt.from_numpy(query)
    probs = knn_layer(query)
    gld_probs_unscattered = build_gld_probs(offset)
    gld_probs = pt.zeros(1, num_dimensions)
    gld_probs[0, 1] = gld_probs_unscattered[0]
    gld_probs[0, 3] = gld_probs_unscattered[1] + gld_probs_unscattered[2]
    gld_probs = gld_probs.div_(pt.sum(gld_probs))

    assert pt.allclose(probs, gld_probs)

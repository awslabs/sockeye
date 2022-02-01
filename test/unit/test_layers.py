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

import pytest
import torch as pt

import sockeye.layers
import sockeye.transformer


def test_lhuc():
    num_hidden = 50
    batch_size = 10
    inp = pt.rand(batch_size, num_hidden)

    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden)
    pt.nn.init.zeros_(lhuc.weight)
    out = lhuc(inp)
    pt.testing.assert_allclose(inp, out)

    lhuc = sockeye.layers.LHUC(num_hidden=num_hidden)
    pt.nn.init.constant_(lhuc.weight, 20.0)
    out = lhuc(inp)
    pt.testing.assert_allclose(2 * inp, out)


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
    pt.testing.assert_allclose(out[0], expected_fixed_embedding)
    pt.testing.assert_allclose(out[1], expected_fixed_embedding)

    # steps
    steps = pt.tensor([2, 3, 1, 1, 1]).unsqueeze(0)
    out = b(data, steps)
    pt.testing.assert_allclose(out[0, 0], expected_fixed_embedding[2])
    pt.testing.assert_allclose(out[1, 0], expected_fixed_embedding[2])
    pt.testing.assert_allclose(out[0, 1], expected_fixed_embedding[3])
    pt.testing.assert_allclose(out[1, 1], expected_fixed_embedding[3])
    pt.testing.assert_allclose(out[0, 2], expected_fixed_embedding[1])
    pt.testing.assert_allclose(out[1, 2], expected_fixed_embedding[1])

    # learned embeddings
    b = sockeye.layers.PositionalEmbeddings(weight_type='learned',
                                            num_embed=num_embed,
                                            max_seq_len=max_seq_len,
                                            scale_up_input=scale_up_input,
                                            scale_down_positions=scale_down_positions)
    pt.nn.init.constant_(b.weight, val=1.0)
    expected_learned_embeddings = pt.ones(data_len, num_embed)
    out = b(data, None)
    pt.testing.assert_allclose(out[0], expected_learned_embeddings)


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

    pt.testing.assert_allclose(output_restricted, reduced_output)


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

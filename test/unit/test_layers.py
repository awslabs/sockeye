# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as onp
import pytest
import torch as pt

import sockeye.layers_pt
import sockeye.transformer_pt


def test_lhuc():
    num_hidden = 50
    batch_size = 10
    inp = pt.rand(batch_size, num_hidden)

    lhuc = sockeye.layers_pt.PyTorchLHUC(num_hidden=num_hidden)
    pt.nn.init.zeros_(lhuc.weight)
    out = lhuc(inp)
    pt.testing.assert_allclose(inp, out)

    lhuc = sockeye.layers_pt.PyTorchLHUC(num_hidden=num_hidden)
    pt.nn.init.constant_(lhuc.weight, 20.0)
    out = lhuc(inp)
    pt.testing.assert_allclose(2 * inp, out)


def test_mx_pt_eq_lhuc():
    mxnet = pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    num_hidden = 50
    batch_size = 10
    inp_mx = np.random.uniform(0, 1, size=(batch_size, num_hidden))
    inp_pt = pt.as_tensor(inp_mx.asnumpy())
    b_mx = sockeye.layers.LHUC(num_hidden=num_hidden, weight_init='zeros')
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchLHUC(num_hidden=num_hidden)
    pt.nn.init.zeros_(b_pt.weight)

    out_mx = b_mx(inp_mx).asnumpy()
    out_pt = b_pt(inp_pt).detach().numpy()

    assert onp.allclose(out_mx, out_pt)

    b_mx = sockeye.layers.LHUC(num_hidden=num_hidden, weight_init=mxnet.init.Constant(value=20.0))
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchLHUC(num_hidden=num_hidden)
    pt.nn.init.constant_(b_pt.weight, 20.0)

    out_mx = b_mx(inp_mx).asnumpy()
    out_pt = b_pt(inp_pt).detach().numpy()

    assert onp.allclose(out_mx, out_pt)


def test_weight_normalization():
    expected_norm = onp.array([1., 1.])
    weight = pt.tensor([[1., 2.],
                        [3., 4.]])
    weight_norm = sockeye.layers_pt.PyTorchWeightNormalization(num_hidden=2)
    norm_weight = weight_norm(weight).detach().numpy()
    assert onp.allclose(onp.linalg.norm(norm_weight, axis=1), expected_norm)


def test_mx_pt_eq_weight_normalization():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    num_hidden = 3
    weight_mx = np.random.uniform(0, 1, size=(num_hidden, 4))
    weight_pt = pt.as_tensor(weight_mx.asnumpy())
    b_mx = sockeye.layers.WeightNormalization(num_hidden=num_hidden)
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchWeightNormalization(num_hidden=num_hidden)

    result_mx = b_mx(weight_mx).asnumpy()
    result_pt = b_pt(weight_pt).detach().numpy()

    assert np.allclose(result_mx, result_pt)


def test_positional_embeddings():
    num_embed = 32
    max_seq_len = 10
    scale_up_input = False
    scale_down_positions = False
    data_len = 5
    data = pt.zeros(2, data_len, num_embed)

    # fixed embeddings
    expected_fixed_embedding = sockeye.layers_pt.pytorch_get_positional_embeddings(data_len, num_embed)
    b = sockeye.layers_pt.PyTorchPositionalEmbeddings(weight_type='fixed',
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
    b = sockeye.layers_pt.PyTorchPositionalEmbeddings(weight_type='learned',
                                                      num_embed=num_embed,
                                                      max_seq_len=max_seq_len,
                                                      scale_up_input=scale_up_input,
                                                      scale_down_positions=scale_down_positions)
    pt.nn.init.constant_(b.weight, val=1.0)
    expected_learned_embeddings = pt.ones(data_len, num_embed)
    out = b(data, None)
    pt.testing.assert_allclose(out[0], expected_learned_embeddings)


@pytest.mark.parametrize('data_len, num_embed, scale_up_input, scale_down_positions, steps',
                         [(5, 32, False, False, None),
                          (5, 32, False, False, [2, 3]),
                          (10, 32, True, False, [1, 3]),
                          (4, 32, False, True, [2, 3])])
def test_mx_pt_eq_positional_embeddings(data_len, num_embed, scale_up_input, scale_down_positions, steps):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    max_seq_len = 10
    data_mx = np.random.uniform(0, 1, (2, data_len, num_embed))
    data_pt = pt.as_tensor(data_mx.asnumpy())
    if steps is None:
        steps_mx, steps_pt = None, None
    else:
        steps_mx = np.array(steps).reshape((-1, 1))
        steps_pt = pt.as_tensor(steps).unsqueeze(1)

    b_mx = sockeye.layers.PositionalEmbeddings(weight_type='fixed',
                                               num_embed=num_embed,
                                               max_seq_len=max_seq_len,
                                               scale_up_input=scale_up_input,
                                               scale_down_positions=scale_down_positions,
                                               weight_init=None)
    b_mx.initialize()
    r_mx = b_mx(data_mx, steps_mx).asnumpy()

    b_pt = sockeye.layers_pt.PyTorchPositionalEmbeddings(weight_type='fixed',
                                                         num_embed=num_embed,
                                                         max_seq_len=max_seq_len,
                                                         scale_up_input=scale_up_input,
                                                         scale_down_positions=scale_down_positions)
    b_pt.weights_from_mxnet_block(b_mx)
    r_pt = b_pt(data_pt, steps_pt).detach().numpy()

    np.allclose(r_mx, r_pt)


def test_mx_pt_eq_get_positional_embeddings():
    pytest.importorskip("mxnet")
    import sockeye.layers

    data_len = 5
    num_embed = 32

    embed_mx = sockeye.layers.get_positional_embeddings(data_len, num_embed).asnumpy()
    embed_pt = sockeye.layers_pt.pytorch_get_positional_embeddings(data_len, num_embed).detach().numpy()

    assert onp.allclose(embed_mx, embed_pt)


def test_output_layer():
    num_hidden = 32
    vocab_size = 64
    data = pt.ones(2, 10, num_hidden)
    vocab_slice_ids = pt.tensor([4, 7, 23])

    b = sockeye.layers_pt.PyTorchOutputLayer(num_hidden, vocab_size)
    assert b.weight.data.shape == (vocab_size, num_hidden)

    output = b(data, None)
    assert output.shape == (2, 10, vocab_size)
    reduced_output = output.index_select(-1, vocab_slice_ids)

    output_restricted = b(data, vocab_slice_ids)
    assert output_restricted.shape == (2, 10, len(vocab_slice_ids))

    pt.testing.assert_allclose(output_restricted, reduced_output)


def test_mx_pt_eq_output_layer():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    num_hidden = 32
    vocab_size = 64
    data_mx = np.random.uniform(0, 1, (2, 10, num_hidden))
    data_pt = pt.as_tensor(data_mx.asnumpy())
    vocab_slice_ids_mx = np.array([4, 7, 23])
    vocab_slice_ids_pt = pt.tensor([4, 7, 23])

    b_mx = sockeye.layers.OutputLayer(num_hidden, vocab_size)
    b_mx.initialize()

    b_pt = sockeye.layers_pt.PyTorchOutputLayer(num_hidden, vocab_size)
    b_pt.weights_from_mxnet_block(b_mx)
    assert b_pt.weight.size() == (vocab_size, num_hidden)

    out_mx = b_mx(data_mx, None)
    assert out_mx.shape == (2, 10, vocab_size)

    out_pt = b_pt(data_pt, None)
    assert out_pt.shape == (2, 10, vocab_size)

    assert np.allclose(out_mx.asnumpy(), out_pt.detach().numpy(), atol=1e-06)

    reduced_out_mx = out_mx.take(vocab_slice_ids_mx, axis=-1).asnumpy()
    reduced_out_pt = pt.index_select(out_pt, 2, vocab_slice_ids_pt).detach().numpy()
    assert np.allclose(reduced_out_mx, reduced_out_pt, atol=1e-06)

    out_restricted_mx = b_mx(data_mx, vocab_slice_ids_mx).asnumpy()
    out_restricted_pt = b_pt(data_pt, vocab_slice_ids_pt).detach().numpy()
    assert out_restricted_mx.shape == (2, 10, len(vocab_slice_ids_mx))
    assert out_restricted_pt.shape == (2, 10, len(vocab_slice_ids_pt))

    assert onp.allclose(out_restricted_mx, out_restricted_pt, atol=1e-06)


@pytest.mark.parametrize('qlen, kvlen, batch_size',
                         [(10, 9, 1), (1, 1, 1), (3, 32, 128)])
def test_mx_pt_eq_interleaved_matmul_encdec_qk(qlen, kvlen, batch_size):
    pytest.importorskip("mxnet")
    from mxnet import np, npx
    import sockeye.layers

    hidden = 32
    q_mx = np.random.uniform(0, 1, (qlen, batch_size, hidden))
    kv_mx = np.random.uniform(0, 1, (kvlen, batch_size, hidden * 2))
    heads = 4
    q_pt = pt.as_tensor(q_mx.asnumpy())
    kv_pt = pt.as_tensor(kv_mx.asnumpy())

    assert np.allclose(q_pt.numpy(), q_mx.asnumpy())
    assert np.allclose(kv_pt.numpy(), kv_mx.asnumpy())

    r0 = npx.interleaved_matmul_encdec_qk(q_mx, kv_mx, heads=heads).asnumpy()
    r1 = sockeye.layers_pt.pytorch_interleaved_matmul_encdec_qk(q_pt, kv_pt, heads=heads).detach().numpy()
    assert np.allclose(r0, r1)


@pytest.mark.parametrize('qlen, kvlen, batch_size',
                         [(10, 9, 1), (1, 1, 1), (3, 32, 128)])
def test_mx_pt_eq_interleaved_matmul_encdec_valatt(qlen, kvlen, batch_size):
    pytest.importorskip("mxnet")
    from mxnet import np, npx
    import sockeye.layers

    hidden = 32
    kv_mx = np.random.uniform(0, 1, (kvlen, batch_size, hidden * 2))
    heads = 4
    kv_pt = pt.as_tensor(kv_mx.asnumpy())
    att = np.random.uniform(0, 1, (batch_size * heads, qlen, kvlen))
    attpt = pt.as_tensor(att.asnumpy())
    r0 = npx.interleaved_matmul_encdec_valatt(kv_mx, att, heads=heads).asnumpy()
    r1 = sockeye.layers_pt.pytorch_interleaved_matmul_encdec_valatt(kv_pt, attpt, heads=heads).numpy()
    assert np.allclose(r0, r1)


@pytest.mark.parametrize('qlen, kvlen, batch_size, hidden, heads',
                         [
                             (10, 9, 1, 32, 8),
                             (1, 1, 1, 4, 1),
                             (13, 13, 20, 8, 2),
                             (3, 32, 64, 128, 2)
                         ])
def test_mx_pt_eq_dot_attention_cell(qlen, kvlen, batch_size, hidden, heads):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers
    import sockeye.transformer
    import sockeye.transformer_pt

    q_mx = np.random.uniform(0, 1, (qlen, batch_size, hidden))
    kv_mx = np.random.uniform(0, 1, (kvlen, batch_size, hidden * 2))
    q_pt = pt.as_tensor(q_mx.asnumpy())
    kv_pt = pt.as_tensor(kv_mx.asnumpy())

    if qlen == kvlen:  # self-attention case
        bias_mx = sockeye.transformer.AutoRegressiveBias()
        bias_mx.initialize()
        autoregr_mx = bias_mx(q_mx.transpose(1, 0, 2))
        mx_args = (None, autoregr_mx)  # no source mask, autoregr mask

        mask_pt = sockeye.transformer_pt.AutoRegressiveMask()
        att_mask_pt = mask_pt(q_pt.permute(1, 0, 2))

    else:  # cross-attention
        lengths_mx = np.random.randint(1, kvlen, (batch_size,), )
        valid_lengths_mx = sockeye.layers.prepare_source_valid_lengths(lengths_mx, q_mx.transpose(1, 0, 2), heads)
        mx_args = (valid_lengths_mx, None)  # source mask, no autoregr mask

        lengths_pt = pt.tensor(lengths_mx.asnumpy())
        att_mask_pt = sockeye.layers_pt.prepare_source_length_mask(lengths_pt, heads, kvlen)

    b_mx = sockeye.layers.DotAttentionCell(dropout=0.0)
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchDotAttentionCell(dropout=0.0, heads=heads)

    r_mx = b_mx(q_mx, kv_mx, heads, *mx_args).asnumpy()
    r_pt = b_pt(q_pt, kv_pt, mask=att_mask_pt).detach().numpy()

    assert np.allclose(r_mx, r_pt, atol=1e-06)


@pytest.mark.parametrize('qlen, kvlen, batch_size, hidden, heads',
                         [(10, 9, 1, 12, 4), (1, 1, 1, 4, 1), (3, 32, 15, 64, 2)])
def test_mx_pt_eq_multi_head_attention_base(qlen, kvlen, batch_size, hidden, heads):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    q_mx = np.random.uniform(0, 1, (qlen, batch_size, hidden))
    kv_mx = np.random.uniform(0, 1, (kvlen, batch_size, hidden * 2))
    q_pt = pt.as_tensor(q_mx.asnumpy())
    kv_pt = pt.as_tensor(kv_mx.asnumpy())

    b_mx = sockeye.layers.MultiHeadAttentionBase(hidden, heads, hidden)
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchMultiHeadAttentionBase(hidden, heads, hidden)
    # use mxnet parameter initializations for pytorch block
    b_pt.ff_out.weight.data[:] = pt.as_tensor(b_mx.ff_out.weight.data().asnumpy())

    r_mx = b_mx._attend(q_mx, kv_mx, None, None).asnumpy()
    r_pt = b_pt._attend(q_pt, kv_pt, mask=None).detach().numpy()

    assert np.allclose(r_mx, r_pt, atol=1e-06)


@pytest.mark.parametrize('seq_len, batch_size, hidden, heads',
                         [(10, 1, 12, 4), (1, 1, 4, 1), (3, 15, 64, 2)])
def test_mx_pt_eq_multi_head_self_attention(seq_len, batch_size, hidden, heads):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    inputs_mx = np.random.uniform(0, 1, (seq_len, batch_size, hidden))
    inputs_pt = pt.as_tensor(inputs_mx.asnumpy())

    b_mx = sockeye.layers.MultiHeadSelfAttention(hidden, heads, hidden, dropout=0.0)
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchMultiHeadSelfAttention(hidden, heads, hidden, dropout=0.0)
    b_pt.eval()
    b_pt.weights_from_mxnet_block(b_mx)

    r_mx, states_mx = b_mx(inputs_mx, None, None, None)
    r_pt, states_pt = b_pt(inputs_pt, previous_states=None, inputs_length_mask=None, bias=None)

    r_mx = r_mx.asnumpy()
    states_mx = states_mx.asnumpy()
    r_pt = r_pt.detach().numpy()
    states_pt = states_pt.detach().numpy()

    assert np.allclose(r_mx, r_pt, atol=1e-06)
    assert np.allclose(states_mx, states_pt, atol=1e-06)


@pytest.mark.parametrize('qlen, kvlen, batch_size, hidden, heads',
                         [(10, 9, 1, 12, 4), (1, 1, 2, 4, 1), (3, 32, 15, 64, 8),
                          (10, 32, 15, 32, 8), (1, 1, 1, 1, 1)])
def test_interleaved_multihead_attention(qlen, kvlen, batch_size, hidden, heads):
    queries_pt = pt.rand((qlen, batch_size, hidden))
    memory_pt = pt.rand((kvlen, batch_size, hidden))

    # test without mask
    mha = sockeye.layers_pt.PyTorchMultiHeadAttention(hidden, heads, hidden, dropout=0.0, depth_key_value=hidden)
    mha.train()
    assert not mha.kv_interleaved
    r_train = mha(queries_pt, memory_pt, mask=None, projected_memory_kv=None)
    mha.eval()
    assert mha.kv_interleaved
    r_test = mha(queries_pt, memory_pt, mask=None, projected_memory_kv=None)
    assert pt.allclose(r_train, r_test, atol=1e-06)

    # test with mask
    valid_length = pt.randint(1, kvlen + 1, (batch_size,))
    mask = sockeye.layers_pt.prepare_source_length_mask(valid_length, heads, kvlen)
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
    mha = sockeye.layers_pt.PyTorchMultiHeadSelfAttention(hidden, heads, hidden, dropout=0.0)
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
        mask = sockeye.transformer_pt.AutoRegressiveMask()(inputs.transpose(0, 1))
        mha.train()
        assert not mha.kv_interleaved
        r_train, _ = mha(inputs, previous_states=None, mask=mask)
        mha.eval()
        assert mha.kv_interleaved
        r_test, _ = mha(inputs, previous_states=None, mask=mask)
        assert pt.allclose(r_train, r_test, atol=1e-06)
    elif side == 'encoder':
        valid_length = pt.randint(1, seq_len+1, (batch_size,))
        # source attention mask. Shape: (batch * heads, 1, seq_len)
        mask = sockeye.layers_pt.prepare_source_length_mask(valid_length, heads, seq_len)
        mask = mask.repeat(1, seq_len, 1)  # Shape: (batch * heads, seq_len, seq_len)
        mha.train()
        assert not mha.kv_interleaved
        r_train, _ = mha(inputs, previous_states=None, mask=mask)
        mha.eval()
        assert mha.kv_interleaved
        r_test, _ = mha(inputs, previous_states=None, mask=mask)  # Note: can also handle the mask repated on the qlen axis
        assert pt.allclose(r_train, r_test, atol=1e-06)


@pytest.mark.parametrize('qlen, kvlen, batch_size, hidden, heads',
                         [(10, 9, 1, 12, 4), (1, 1, 1, 4, 1), (3, 32, 15, 64, 2)])
def test_mx_pt_eq_multi_head_attention(qlen, kvlen, batch_size, hidden, heads):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    queries_mx = np.random.uniform(0, 1, (qlen, batch_size, hidden))
    queries_pt = pt.as_tensor(queries_mx.asnumpy())
    memory_mx = np.random.uniform(0, 1, (kvlen, batch_size, hidden))
    memory_pt = pt.as_tensor(memory_mx.asnumpy())

    b_mx = sockeye.layers.MultiHeadAttention(hidden, heads, hidden, dropout=0.0)
    b_mx.initialize()
    r_mx = b_mx(queries_mx, memory_mx, None, None, None)

    b_pt = sockeye.layers_pt.PyTorchMultiHeadAttention(hidden, heads, hidden, dropout=0.0, depth_key_value=hidden)
    b_pt.weights_from_mxnet_block(b_mx)
    r_pt = b_pt(queries_pt, memory_pt, mask=None, projected_memory_kv=None)

    print(b_pt.ff_kv.weight[0])
    print(b_mx.ff_kv.weight.data()[0])

    r_mx = r_mx.asnumpy()
    r_pt = r_pt.detach().numpy()

    assert np.allclose(r_mx, r_pt, atol=1e-06)


@pytest.mark.parametrize('hidden, inference_only, seq_len, batch',
                         [(16, False, 10, 4),
                          (10, False, 2, 1),
                          (16, True, 1, 4), ])
def test_mx_pt_eq_ssru(hidden, inference_only, seq_len, batch):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    b_mx = sockeye.layers.SSRU(hidden, inference_only)
    b_mx.initialize()
    b_pt = sockeye.layers_pt.PyTorchSSRU(hidden, inference_only)
    b_pt.weights_from_mxnet_block(b_mx)

    inputs_mx = np.random.uniform(0, 1, (seq_len, batch, hidden))
    previous_states_mx = np.zeros((1, batch, hidden))
    inputs_pt = pt.as_tensor(inputs_mx.asnumpy())
    previous_states_pt = pt.as_tensor(previous_states_mx.asnumpy())

    r1_mx, r2_mx = b_mx(inputs_mx, previous_states_mx)
    r1_pt, r2_pt = b_pt(inputs_pt, previous_states_pt)

    r1_mx = r1_mx.asnumpy()
    r2_mx = r2_mx.asnumpy()
    r1_pt = r1_pt.detach().numpy()
    r2_pt = r2_pt.detach().numpy()

    assert np.allclose(r1_mx, r1_pt)
    assert np.allclose(r2_mx, r2_pt)


def test_mx_pt_eq_length_ratio():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.layers

    hidden_size = 32
    seq_len = 10
    batch_size = 8
    num_layers = 1  # more layers seems to be numerically unstable

    source_encoded_mx = np.random.uniform(0, 1, (batch_size, seq_len, hidden_size))
    source_encoded_pt = pt.as_tensor(source_encoded_mx.asnumpy())
    source_lengths_mx = np.random.randint(1, seq_len, (batch_size,), dtype='int32')
    source_lengths_pt = pt.as_tensor(source_lengths_mx.asnumpy())

    b_mx = sockeye.layers.LengthRatio(hidden_size=hidden_size, num_layers=num_layers)
    b_mx.initialize()
    r_mx = b_mx(source_encoded_mx, source_lengths_mx).asnumpy()

    b_pt = sockeye.layers_pt.PyTorchLengthRatio(hidden_size=hidden_size, num_layers=num_layers)
    b_pt.weights_from_mxnet_block(b_mx)
    r_pt = b_pt(source_encoded_pt, source_lengths_pt).detach().numpy()

    assert np.allclose(r_mx, r_pt)

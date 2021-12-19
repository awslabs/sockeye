# Copyright 2018--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sockeye.constants as C
import sockeye.transformer_pt
from sockeye.layers_pt import prepare_source_length_mask


def test_auto_regressive_bias_dtype():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer

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
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer

    block = sockeye.transformer.AutoRegressiveBias()
    block.initialize()
    length = 2
    data = np.ones((2, length, 10), dtype='float32')
    bias = block(data)

    expected = np.array([[0.0, -1.0e8], [0.0, 0.0]]).reshape((1, 2, 2))
    onp.testing.assert_array_equal(bias, expected)


@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_transformer_feed_forward(use_glu):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer
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


@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_pt_transformer_feed_forward(use_glu):
    block = sockeye.transformer_pt.PyTorchTransformerFeedForward(num_hidden=2,
                                                                 num_model=2,
                                                                 act_type=C.RELU,
                                                                 dropout=0.1,
                                                                 use_glu=use_glu)

    data = pt.ones(1, 10, 2)
    block(data)


@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_mx_pt_eq_transformer_feed_forward(use_glu):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer

    b_mx = sockeye.transformer.TransformerFeedForward(num_hidden=4,
                                                      num_model=2,
                                                      act_type=C.RELU,
                                                      dropout=0.0,
                                                      dtype=C.DTYPE_FP32,
                                                      use_glu=use_glu)
    b_mx.initialize()

    b_pt = sockeye.transformer_pt.PyTorchTransformerFeedForward(num_hidden=4,
                                                                num_model=2,
                                                                act_type=C.RELU,
                                                                dropout=0.0,
                                                                use_glu=use_glu)
    b_pt.weights_from_mxnet_block(b_mx)

    result_mx = b_mx(np.ones((2, 2, 2))).asnumpy()
    result_pt = b_pt(pt.ones(2, 2, 2)).detach().numpy()

    assert np.allclose(result_mx, result_pt)


@pytest.mark.parametrize('length', [1, 10, 100])
def test_pt_autoregressive_mask(length):
    x_pt = pt.zeros(2, length, 32)
    b_pt = sockeye.transformer_pt.AutoRegressiveMask()
    result_pt = b_pt(x_pt).detach()

    assert result_pt.dtype == pt.bool
    assert result_pt.size() == (length, length)


@pytest.mark.parametrize('sequence', ['rn', 'nr', 'r', 'n', ''])  # not testing dropout
def test_mx_pt_eq_transformer_process_block(sequence):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer

    num_hidden = 32
    x_mx = np.random.uniform(0, 1, (2, 10, num_hidden))
    prev_mx = np.random.uniform(0, 1, (2, 10, num_hidden))
    x_pt = pt.as_tensor(x_mx.asnumpy())
    prev_pt = pt.as_tensor(prev_mx.asnumpy())

    b_mx = sockeye.transformer.TransformerProcessBlock(sequence, 0.0, num_hidden)
    b_mx.initialize()
    b_pt = sockeye.transformer_pt.PyTorchTransformerProcessBlock(sequence, 0.0, num_hidden)
    b_pt.weights_from_mxnet_block(b_mx)

    result_mx = b_mx(x_mx, prev_mx).asnumpy()
    result_pt = b_pt(x_pt, prev_pt).detach().numpy()

    assert np.allclose(result_mx, result_pt, atol=1e-06)


@pytest.mark.parametrize('batch_size, input_len, model_size, heads, ff_hidden, num_layers, use_lhuc, '
                         'preprocess_sequence, postprocess_sequence, use_glu',
                         [
                             (1, 100, 512, 8, 1024, 1, False, 'n', 'r', False),
                             (5, 25, 128, 4, 32, 1, True, 'n', 'nr', True),
                         ])
def test_mx_pt_eq_transformer_encoder_block(batch_size, input_len, model_size, heads,
                                            ff_hidden, num_layers, use_lhuc, preprocess_sequence,
                                            postprocess_sequence, use_glu):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer

    config = sockeye.transformer.TransformerConfig(
        model_size=model_size,
        attention_heads=heads,
        feed_forward_num_hidden=ff_hidden,
        act_type=C.RELU,
        num_layers=num_layers,
        dropout_attention=0.0,
        dropout_act=0.0,
        dropout_prepost=0.0,
        positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
        preprocess_sequence=preprocess_sequence,
        postprocess_sequence=postprocess_sequence,
        max_seq_len_source=100,
        max_seq_len_target=100,
        decoder_type=C.TRANSFORMER_TYPE,
        use_lhuc=use_lhuc,
        depth_key_value=model_size,
        use_glu=use_glu)

    data_mx = np.random.uniform(0, 1, (batch_size, input_len, model_size))
    data_pt = pt.as_tensor(data_mx.asnumpy())
    valid_lengths_mx = np.random.randint(1, input_len, (batch_size,))
    valid_lengths_pt = pt.as_tensor(valid_lengths_mx.asnumpy())

    att_valid_lengths_mx = sockeye.layers.prepare_source_valid_lengths(valid_lengths_mx, data_mx, num_heads=heads)
    source_length_mask_pt = sockeye.layers_pt.prepare_source_length_mask(valid_lengths_pt,
                                                                         heads=heads, max_length=data_pt.size()[1])
    source_length_mask_pt = source_length_mask_pt.repeat(1, input_len, 1)

    # time-major as done in the Transformer encoder
    data_mx = np.transpose(data_mx, axes=(1, 0, 2))
    data_pt = data_pt.permute(1, 0, 2)

    b_mx = sockeye.transformer.TransformerEncoderBlock(config, dtype=C.DTYPE_FP32)
    b_mx.initialize()
    r_mx = b_mx(data_mx, att_valid_lengths_mx).asnumpy()

    b_pt = sockeye.transformer_pt.PyTorchTransformerEncoderBlock(config)
    b_pt.weights_from_mxnet_block(b_mx)

    r_pt = b_pt(data_pt, source_length_mask_pt).detach().numpy()

    assert np.allclose(r_mx, r_pt, atol=1e-05)


@pytest.mark.parametrize('batch_size, source_input_len, target_input_len, model_size, heads, ff_hidden, '
                         'num_layers, use_lhuc, preprocess_sequence, postprocess_sequence, use_glu, '
                         'inference_only, decoder_type',
                         [
                             (1, 100, 100, 512, 8, 1024, 1, False, 'n', 'r', False, False, C.TRANSFORMER_TYPE),
                             (5, 23, 25, 128, 4, 32, 1, True, 'n', 'nr', True, False, C.TRANSFORMER_TYPE),
                             (1, 55, 54, 32, 2, 64, 1, False, '', '', False, False, C.SSRU_TRANSFORMER),
                             (1, 55, 54, 32, 2, 64, 1, False, '', '', False, True, C.SSRU_TRANSFORMER)
                         ])
def test_mx_pt_eq_transformer_decoder_block(batch_size, source_input_len, target_input_len, model_size, heads,
                                            ff_hidden, num_layers, use_lhuc, preprocess_sequence,
                                            postprocess_sequence, use_glu, inference_only, decoder_type):
    mxnet = pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.transformer
    import sockeye.layers

    pt.manual_seed(13)
    mxnet.random.seed(13)
    config = sockeye.transformer.TransformerConfig(
        model_size=model_size,
        attention_heads=heads,
        feed_forward_num_hidden=ff_hidden,
        act_type=C.RELU,
        num_layers=num_layers,
        dropout_attention=0.0,
        dropout_act=0.0,
        dropout_prepost=0.0,
        positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
        preprocess_sequence=preprocess_sequence,
        postprocess_sequence=postprocess_sequence,
        max_seq_len_source=100,
        max_seq_len_target=100,
        decoder_type=decoder_type,
        use_lhuc=use_lhuc,
        depth_key_value=model_size,
        use_glu=use_glu)

    target_mx = np.random.uniform(0, 1, (batch_size, target_input_len, model_size))
    target_pt = pt.tensor(target_mx.asnumpy())

    autoregr_bias_mx = sockeye.transformer.AutoRegressiveBias()
    autoregr_bias_mx.initialize()
    autoregr_bias_pt = sockeye.transformer_pt.AutoRegressiveMask()
    target_bias_mx = autoregr_bias_mx(target_mx)
    target_bias_pt = autoregr_bias_pt(target_pt).detach()

    source_mx = np.random.uniform(0, 1, (batch_size, source_input_len, model_size))
    source_mx = np.transpose(source_mx, axes=(1, 0, 2))
    source_pt = pt.tensor(source_mx.asnumpy())

    source_lengths_mx = np.random.randint(0, source_input_len, (batch_size,))
    source_lengths_pt = pt.tensor(source_lengths_mx.asnumpy())
    source_lengths_mx = sockeye.layers.prepare_source_valid_lengths(source_lengths_mx, target_mx, heads)
    source_max_len = source_pt.size()[0]
    source_length_mask_pt = prepare_source_length_mask(source_lengths_pt, heads, source_max_len)

    target_mx = np.transpose(target_mx, axes=(1, 0, 2))
    target_pt = target_pt.permute(1, 0, 2)

    b_mx = sockeye.transformer.TransformerDecoderBlock(config, inference_only, C.DTYPE_FP32)
    b_mx.initialize()
    autoregr_states_mx = np.zeros(b_mx.get_states_shape(batch_size))
    new_target_mx, new_states_mx = b_mx(target_mx, target_bias_mx, source_mx, source_lengths_mx, autoregr_states_mx)
    new_target_mx = new_target_mx.asnumpy()
    new_states_mx = [s.asnumpy() for s in new_states_mx]

    b_pt = sockeye.transformer_pt.PyTorchTransformerDecoderBlock(config, inference_only)
    b_pt.eval()
    autoregr_states_pt = pt.zeros(*b_pt.get_states_shape(batch_size))
    b_pt.weights_from_mxnet_block(b_mx)
    new_target_pt, new_states_pt = b_pt(target_pt, target_bias_pt, source_pt, source_length_mask_pt, autoregr_states_pt)
    new_target_pt = new_target_pt.detach().numpy()
    new_states_pt = [s.detach().numpy() for s in new_states_pt]

    assert len(new_states_mx) == len(new_states_pt)
    assert np.allclose(new_target_mx, new_target_pt, atol=1e-05)
    assert np.allclose(new_states_mx, new_states_pt, atol=1e-05)

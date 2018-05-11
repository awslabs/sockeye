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

import pytest
import mxnet as mx
import numpy as np

import sockeye.constants as C
import sockeye.encoder


_BATCH_SIZE = 8
_SEQ_LEN = 10
_NUM_EMBED = 8
_DATA_LENGTH_ND = mx.nd.array([1, 2, 3, 4, 5, 6, 7, 8])


def test_get_recurrent_encoder_no_conv_config():
    rnn_config = sockeye.rnn.RNNConfig(cell_type=C.LSTM_TYPE,
                                       num_hidden=10,
                                       num_layers=20,
                                       dropout_inputs=1.0,
                                       dropout_states=2.0)
    config = sockeye.encoder.RecurrentEncoderConfig(rnn_config, conv_config=None, reverse_input=True, dtype='float16')
    encoder = sockeye.encoder.get_recurrent_encoder(config, prefix='test_')

    assert type(encoder) == sockeye.encoder.EncoderSequence
    assert len(encoder.encoders) == 5

    assert type(encoder.encoders[0]) == sockeye.encoder.ConvertLayout
    assert encoder.encoders[0].__dict__.items() >= dict(num_hidden=0, target_layout='TNC',
                                                        dtype='float16').items()

    assert type(encoder.encoders[1]) == sockeye.encoder.ReverseSequence
    assert encoder.encoders[1].__dict__.items() >= dict(num_hidden=0, dtype='float16').items()

    assert type(encoder.encoders[2]) == sockeye.encoder.BiDirectionalRNNEncoder
    assert encoder.encoders[2].__dict__.items() >= dict(layout='TNC', prefix='test_encoder_birnn_', dtype='float32').items()

    assert type(encoder.encoders[3]) == sockeye.encoder.RecurrentEncoder
    assert encoder.encoders[3].__dict__.items() >= dict(layout='TNC', dtype='float32').items()

    assert type(encoder.encoders[4]) == sockeye.encoder.ConvertLayout
    assert encoder.encoders[4].__dict__.items() >= dict(num_hidden=10, target_layout='NTC', dtype='float16').items()


def test_get_recurrent_encoder():
    rnn_config = sockeye.rnn.RNNConfig(cell_type=C.LSTM_TYPE,
                                       num_hidden=10,
                                       num_layers=20,
                                       dropout_inputs=1.0,
                                       dropout_states=2.0)
    conv_config = sockeye.encoder.ConvolutionalEmbeddingConfig(num_embed=6, add_positional_encoding=True)
    config = sockeye.encoder.RecurrentEncoderConfig(rnn_config, conv_config, reverse_input=True, dtype='float16')
    encoder = sockeye.encoder.get_recurrent_encoder(config, prefix='test_')

    assert type(encoder) == sockeye.encoder.EncoderSequence
    assert len(encoder.encoders) == 7

    assert type(encoder.encoders[0]) == sockeye.encoder.ConvolutionalEmbeddingEncoder
    assert encoder.encoders[0].__dict__.items() >= dict(num_embed=6, prefix='test_encoder_char_',
                                                        dtype='float32').items()

    assert type(encoder.encoders[1]) == sockeye.encoder.AddSinCosPositionalEmbeddings
    assert encoder.encoders[1].__dict__.items() >= dict(num_embed=6, prefix='test_encoder_char_add_positional_encodings',
                                                        scale_up_input=False,
                                                        scale_down_positions=False, dtype='float16').items()

    assert type(encoder.encoders[2]) == sockeye.encoder.ConvertLayout
    assert encoder.encoders[2].__dict__.items() >= dict(num_hidden=6, target_layout='TNC', dtype='float16').items()

    assert type(encoder.encoders[3]) == sockeye.encoder.ReverseSequence
    assert encoder.encoders[3].__dict__.items() >= dict(num_hidden=6, dtype='float16').items()

    assert type(encoder.encoders[4]) == sockeye.encoder.BiDirectionalRNNEncoder
    assert encoder.encoders[4].__dict__.items() >= dict(layout='TNC', prefix='test_encoder_birnn_', dtype='float32').items()

    assert type(encoder.encoders[5]) == sockeye.encoder.RecurrentEncoder
    assert encoder.encoders[5].__dict__.items() >= dict(layout='TNC', dtype='float32').items()

    assert type(encoder.encoders[6]) == sockeye.encoder.ConvertLayout
    assert encoder.encoders[6].__dict__.items() >= dict(num_hidden=10, target_layout='NTC', dtype='float16').items()


def test_get_transformer_encoder():
    conv_config = sockeye.encoder.ConvolutionalEmbeddingConfig(num_embed=6, add_positional_encoding=True)
    config = sockeye.transformer.TransformerConfig(model_size=20,
                                                   attention_heads=10,
                                                   feed_forward_num_hidden=30,
                                                   act_type='test_act',
                                                   num_layers=40,
                                                   dropout_attention=1.0,
                                                   dropout_act=2.0,
                                                   dropout_prepost=3.0,
                                                   positional_embedding_type=C.LEARNED_POSITIONAL_EMBEDDING,
                                                   preprocess_sequence='test_pre',
                                                   postprocess_sequence='test_post',
                                                   max_seq_len_source=50,
                                                   max_seq_len_target=60,
                                                   conv_config=conv_config, dtype='float16')
    encoder = sockeye.encoder.get_transformer_encoder(config, prefix='test_')

    assert type(encoder) == sockeye.encoder.EncoderSequence
    assert len(encoder.encoders) == 3

    assert type(encoder.encoders[0]) == sockeye.encoder.AddLearnedPositionalEmbeddings
    assert encoder.encoders[0].__dict__.items() >= dict(num_embed=20, max_seq_len=50, prefix='test_source_pos_embed_',
                                                        dtype='float16').items()

    assert type(encoder.encoders[1]) == sockeye.encoder.ConvolutionalEmbeddingEncoder
    assert encoder.encoders[1].__dict__.items() >= dict(num_embed=6, prefix='test_encoder_char_', dtype='float32').items()

    assert type(encoder.encoders[2]) == sockeye.encoder.TransformerEncoder
    assert encoder.encoders[2].__dict__.items() >= dict(prefix='test_encoder_transformer_', dtype='float16').items()


def test_get_convolutional_encoder():
    cnn_config = sockeye.convolution.ConvolutionConfig(kernel_width=5, num_hidden=10)
    config = sockeye.encoder.ConvolutionalEncoderConfig(num_embed=10,
                                                        max_seq_len_source=20,
                                                        cnn_config=cnn_config,
                                                        num_layers=30,
                                                        positional_embedding_type=C.NO_POSITIONAL_EMBEDDING,
                                                        dtype='float16')
    encoder = sockeye.encoder.get_convolutional_encoder(config, prefix='test_')

    assert type(encoder) == sockeye.encoder.EncoderSequence
    assert len(encoder.encoders) == 2

    assert type(encoder.encoders[0]) == sockeye.encoder.NoOpPositionalEmbeddings
    assert encoder.encoders[0].__dict__.items() >= dict(num_embed=10, dtype='float16').items()

    assert type(encoder.encoders[1]) == sockeye.encoder.ConvolutionalEncoder
    assert encoder.encoders[1].__dict__.items() >= dict(dtype='float16').items()


@pytest.mark.parametrize("config, out_data_shape, out_data_length, out_seq_len", [
    (sockeye.encoder.ConvolutionalEmbeddingConfig(num_embed=_NUM_EMBED,
                                                  output_dim=None,
                                                  max_filter_width=3,
                                                  num_filters=[8, 16, 16],
                                                  pool_stride=4,
                                                  num_highway_layers=2,
                                                  dropout=0,
                                                  add_positional_encoding=False),
     (8, 3, 40),
     [1, 1, 1, 1, 2, 2, 2, 2],
     3),
    (sockeye.encoder.ConvolutionalEmbeddingConfig(num_embed=_NUM_EMBED,
                                                  output_dim=32,
                                                  max_filter_width=2,
                                                  num_filters=[8, 16],
                                                  pool_stride=3,
                                                  num_highway_layers=0,
                                                  dropout=0.1,
                                                  add_positional_encoding=True),
     (8, 4, 32),
     [1, 1, 1, 2, 2, 2, 3, 3],
     4),
])
def test_convolutional_embedding_encoder(config, out_data_shape, out_data_length, out_seq_len):
    conv_embed = sockeye.encoder.ConvolutionalEmbeddingEncoder(config)

    data_nd = mx.nd.random_normal(shape=(_BATCH_SIZE, _SEQ_LEN, _NUM_EMBED))

    data = mx.sym.Variable("data", shape=data_nd.shape)
    data_length = mx.sym.Variable("data_length", shape=_DATA_LENGTH_ND.shape)

    (encoded_data,
     encoded_data_length,
     encoded_seq_len) = conv_embed.encode(data=data, data_length=data_length, seq_len=_SEQ_LEN)

    exe = encoded_data.simple_bind(mx.cpu(), data=data_nd.shape)
    exe.forward(data=data_nd)
    assert exe.outputs[0].shape == out_data_shape

    exe = encoded_data_length.simple_bind(mx.cpu(), data_length=_DATA_LENGTH_ND.shape)
    exe.forward(data_length=_DATA_LENGTH_ND)
    assert np.equal(exe.outputs[0].asnumpy(), np.asarray(out_data_length)).all()

    assert encoded_seq_len == out_seq_len


def test_sincos_positional_embeddings():
    # Test that .encode() and .encode_positions() return the same values:
    data = mx.sym.Variable("data")
    positions = mx.sym.Variable("positions")
    pos_encoder = sockeye.encoder.AddSinCosPositionalEmbeddings(num_embed=_NUM_EMBED,
                                                                scale_up_input=False,
                                                                scale_down_positions=False,
                                                                prefix="test")
    encoded, _, __ = pos_encoder.encode(data, None, _SEQ_LEN)
    nd_encoded = encoded.eval(data=mx.nd.zeros((_BATCH_SIZE, _SEQ_LEN, _NUM_EMBED)))[0]
    # Take the first element in the batch to get (seq_len, num_embed)
    nd_encoded = nd_encoded[0]

    encoded_positions = pos_encoder.encode_positions(positions, data)
    # Explicitly encode all positions from 0 to _SEQ_LEN
    nd_encoded_positions = encoded_positions.eval(positions=mx.nd.arange(0, _SEQ_LEN),
                                                  data=mx.nd.zeros((_SEQ_LEN, _NUM_EMBED)))[0]
    assert np.isclose(nd_encoded.asnumpy(), nd_encoded_positions.asnumpy()).all()


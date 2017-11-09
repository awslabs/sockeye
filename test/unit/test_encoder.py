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

import sockeye.encoder


_BATCH_SIZE = 8
_SEQ_LEN = 10
_NUM_EMBED = 8
_DATA_LENGTH_ND = mx.nd.array([1, 2, 3, 4, 5, 6, 7, 8])


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
                                                                scale_input=False,
                                                                scale_positions=False,
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


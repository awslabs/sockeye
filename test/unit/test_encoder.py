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
                                                   max_seq_len_target=60)
    encoder = sockeye.encoder.get_transformer_encoder(config, prefix='test_')

    assert type(encoder) == sockeye.encoder.TransformerEncoder

    assert type(encoder.encoders[0]) == sockeye.encoder.AddLearnedPositionalEmbeddings
    assert encoder.encoders[0].__dict__.items() >= dict(num_embed=20, max_seq_len=50, prefix='test_source_pos_embed_',
                                                        dtype='float16').items()

    assert type(encoder.encoders[2]) == sockeye.encoder.TransformerEncoder
    assert encoder.encoders[2].prefix == "test_encoder_transformer_"
    assert encoder.encoders[2].dtype == 'float16'


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


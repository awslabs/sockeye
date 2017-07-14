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
import sockeye.encoder

def test_convolutional_embedding_encoder():

    batch_size = 8
    seq_len = 10

    num_embed = 8
    max_filter_width = 3
    num_filters = [8, 16, 16]
    pool_stride = 5
    num_highway_layers = 2
    dropout = 0.1

    conv_embed = sockeye.encoder.ConvolutionalEmbeddingEncoder(num_embed=num_embed,
                                                               max_filter_width=max_filter_width,
                                                               num_filters=num_filters,
                                                               pool_stride=pool_stride,
                                                               num_highway_layers=num_highway_layers,
                                                               dropout=dropout)

    data_nd = mx.nd.random_normal(shape=(batch_size, seq_len, num_embed))
    data_length_nd = mx.nd.array([1, 2, 3, 4, 5, 6, 7, 8])

    data = mx.sym.Variable("data", shape=data_nd.shape)
    data_length = mx.sym.Variable("data_length", shape=data_length_nd.shape)

    (encoded_data,
     encoded_data_length,
     encoded_seq_len) = conv_embed.encode(data=data, data_length=data_length, seq_len=seq_len)

    exe = encoded_data.simple_bind(mx.cpu(), data=data_nd.shape)
    exe.forward(data=data_nd)
    assert exe.outputs[0].shape == (8, 2, 40)

    exe = encoded_data_length.simple_bind(mx.cpu(), data_length=data_length_nd.shape)
    exe.forward(data_length=data_length_nd)
    assert np.equal(exe.outputs[0].asnumpy(), np.asarray([1, 1, 1, 1, 1, 2, 2, 2])).all()

    assert encoded_seq_len == 2

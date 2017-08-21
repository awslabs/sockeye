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

from typing import Optional

import mxnet as mx
import numpy as np

from . import config
from . import layers


class TransformerConfig(config.Config):

    def __init__(self,
                 model_size: int,
                 attention_heads: int,
                 feed_forward_num_hidden: int,
                 num_layers: int,
                 vocab_size: int,
                 dropout_attention: float,
                 dropout_relu: float,
                 dropout_residual: float,
                 layer_normalization: bool,
                 weight_tying: bool,
                 positional_encodings: bool,
                 conv_config: Optional['ConvolutionalEmbeddingConfig'] = None) -> None:  # type: ignore
        super().__init__()
        self.model_size = model_size
        self.attention_heads = attention_heads
        self.feed_forward_num_hidden = feed_forward_num_hidden
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout_attention = dropout_attention
        self.dropout_relu = dropout_relu
        self.dropout_residual = dropout_residual
        self.layer_normalization = layer_normalization
        self.weight_tying = weight_tying
        self.positional_encodings = positional_encodings
        self.conv_config = conv_config


class TransformerEncoderBlock:
    """
    A transformer encoder block consists of the 4 following sublayers:
     1. self-attention
     2. residual connection (w/ optional layer normalization)
     3. feed-forward network
     4. residual connection (w/ optional layer normalization)
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:
        self.attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_" % prefix)
        self.residual1 = TransformerResidual(num_hidden=config.model_size,
                                             layer_normalization=config.layer_normalization,
                                             dropout=config.dropout_residual,
                                             prefix="%satt_res" % prefix)
        self.feed_forward = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                   num_model=config.model_size,
                                                   dropout=config.dropout_relu,
                                                   prefix="%sff_" % prefix)
        self.residual2 = TransformerResidual(num_hidden=config.model_size,
                                             layer_normalization=config.layer_normalization,
                                             dropout=config.dropout_residual,
                                             prefix="%sff_res" % prefix)

    def __call__(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        data = self.residual1(data,
                              self.attention(data, data_length, seq_len),
                              seq_len)
        data = self.residual2(data,
                              self.feed_forward(data, seq_len),
                              seq_len)
        return data


class TransformerDecoderBlock:
    """
    A transformer decoder block consists of the following sublayers:
     1. self-attention
     2. residual connection (w/ optional layer normalization)
     3. source-attention
     4. residual connection (w/ optional layer normalization)
     5. feed-forward network
     6. residual connection (w/ optional layer normalization)
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.residual_self = TransformerResidual(num_hidden=config.model_size,
                                                 layer_normalization=config.layer_normalization,
                                                 dropout=config.dropout_residual,
                                                 prefix="%satt_self_res" % prefix)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_enc_" % prefix)
        self.residual_enc = TransformerResidual(num_hidden=config.model_size,
                                                layer_normalization=config.layer_normalization,
                                                dropout=config.dropout_residual,
                                                prefix="%satt_enc_res" % prefix)
        self.feed_forward = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                   num_model=config.model_size,
                                                   dropout=config.dropout_relu,
                                                   prefix="%sff_" % prefix)
        self.residual_ff = TransformerResidual(num_hidden=config.model_size,
                                               layer_normalization=config.layer_normalization,
                                               dropout=config.dropout_residual,
                                               prefix="%sff_res" % prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_lengths: mx.sym.Symbol,
                 target_max_length: int,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_lengths: mx.sym.Symbol,
                 source_max_length: int) -> mx.sym.Symbol:
        target = self.residual_self(target,
                                    self.self_attention(target, target_lengths,
                                                        target_max_length, bias=target_bias),
                                    target_max_length)
        target = self.residual_enc(target,
                                   self.enc_attention(target, target_max_length,
                                                      source, source_lengths, source_max_length),
                                   target_max_length)
        target = self.residual_ff(target,
                                  self.feed_forward(target, target_max_length),
                                  target_max_length)
        return target


class TransformerResidual:
    """
    Residual connection with optional layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 layer_normalization: bool,
                 dropout: float,
                 prefix: str) -> None:
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.layer_norm = None
        self.prefix = prefix
        if layer_normalization:
            self.layer_norm = layers.LayerNormalization(num_hidden=self.num_hidden, prefix=self.prefix)

    def __call__(self, x: mx.sym.Symbol, y: mx.sym.Symbol, length: int) -> mx.sym.Symbol:
        """
        Apply residual connections with optional layer normalization and dropout.

        :param x: (batch, length, num_hidden).
        :param y: (batch, length, num_hidden).
        :param length: maximum sequence length.
        :return: (batch, length, num_hidden).
        """
        if self.dropout > 0.0:
            y = mx.sym.Dropout(y, p=self.dropout)
        z = x + y
        if self.layer_norm is not None:
            z = self._reshape_and_normalize(z, length)
        return z

    def _reshape_and_normalize(self, data: mx.sym.Symbol, length: int) -> mx.sym.Symbol:
        data = mx.sym.reshape(data, shape=(-3, self.num_hidden))
        data = self.layer_norm.normalize(data)
        data = mx.sym.reshape(data, shape=(-4, -1, length, self.num_hidden))
        return data


class TransformerFeedForward:
    """
    Position-wise feed-forward network with ReLU activation.
    """

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 dropout: float,
                 prefix: str) -> None:
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.prefix = prefix
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

    def __call__(self, x, length) -> mx.sym.Symbol:
        """
        Position-wise feed-forward network with ReLU activation.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :param length: sequence length
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        # TODO: use a convolution?
        x = mx.sym.reshape(x, shape=(-3, -1))
        h = mx.sym.FullyConnected(data=x, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h)
        h = mx.sym.Activation(h, act_type="relu")
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        y = mx.sym.FullyConnected(data=h, num_hidden=self.num_model, weight=self.w_h2o, bias=self.b_h2o)
        y = mx.sym.reshape(y, shape=(-1, length, self.num_model))
        return y


def get_autoregressive_bias(max_length: int, name: str) -> mx.sym.Symbol:
    """
    Returns bias/mask to ensure position i can only attend to positions <i.

    :param max_length: Sequence length.
    :param name: Name of symbol.
    :return: Bias symbol of shape (1, max_length, max_length).
    """
    return mx.sym.BlockGrad(mx.symbol.Custom(length=max_length,
                                             name=name,
                                             op_type='auto_regressive_bias'))


class AutoRegressiveBias(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, length, length) with cells above the main diagonal
    set to a large negative value, e.g.
    length=4

    0 1 1 1
    0 0 1 1   * -99999
    0 0 0 1
    0 0 0 0
    """

    def __init__(self, length: int) -> None:
        super().__init__()
        self.bias = self.get_bias(length)

    @staticmethod
    def get_bias(length: int):
        # matrix with lower triangle and main diagonal set to 0, upper triangle set to 1
        upper_triangle = np.triu(np.ones((length, length)), k=1)
        # (1, length, length)
        bias = -99999999. * np.reshape(upper_triangle, (1, length, length))
        return mx.nd.array(bias)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("auto_regressive_bias")
class AutoRegressiveBiasProp(mx.operator.CustomOpProp):

    def __init__(self, length: str) -> None:
        super().__init__()
        self.length = int(length)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.length)], []

    def infer_type(self, in_type):
        return [], [np.float32], []

    def create_operator(self, ctx, shapes, dtypes):
        return AutoRegressiveBias(length=self.length)

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

from typing import Optional, Tuple

import mxnet as mx

from . import initializer
from . import utils


class LayerNormalization:
    """
    Implements Ba et al, Layer Normalization (https://arxiv.org/abs/1607.06450).

    :param num_hidden: Number of hidden units of layer to be normalized.
    :param prefix: Optional prefix of layer name.
    :param scale: Optional variable for scaling of shape (num_hidden,). Will be created if None.
    :param shift: Optional variable for shifting of shape (num_hidden,). Will be created if None.
    :param scale_init: Initial value of scale variable if scale is None. Default 1.0.
    :param shift_init: Initial value of shift variable if shift is None. Default 0.0.
    """

    # TODO(fhieber): this should eventually go to MXNet

    def __init__(self,
                 num_hidden: int,
                 prefix: Optional[str] = None,
                 scale: Optional[mx.sym.Symbol] = None,
                 shift: Optional[mx.sym.Symbol] = None,
                 scale_init: float = 1.0,
                 shift_init: float = 0.0) -> None:
        utils.check_condition(num_hidden > 1,
                              "Layer normalization should only be applied to layers with more than 1 neuron.")
        self.prefix = prefix
        self.scale = scale if scale is not None else mx.sym.Variable('%s_gamma' % prefix, shape=(num_hidden,),
                                                                     init=mx.init.Constant(value=scale_init))
        self.shift = shift if shift is not None else mx.sym.Variable('%s_beta' % prefix, shape=(num_hidden,),
                                                                     init=mx.init.Constant(value=shift_init))

    @staticmethod
    def moments(inputs: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Computes mean and variance of a Symbol across axis 1.

        :param inputs: Shape(batch_size, hidden).
        :return: mean, var: Shape(batch_size, 1).
        """
        mean = mx.sym.mean(data=inputs, axis=1, keepdims=True)
        # TODO(fhieber): MXNet should have this.
        var = mx.sym.mean(mx.sym.square(mx.sym.broadcast_minus(inputs, mean)), axis=1, keepdims=True)
        return mean, var

    def normalize(self, inputs: mx.sym.Symbol, eps: float = 0.00001) -> mx.sym.Symbol:
        """
        Normalizes hidden units of inputs.
        inputs = scale * (inputs - mean) / sqrt(var + eps) + shift

        :param inputs: Inputs to normalize. Shape(batch_size, num_hidden).
        :param eps: Variance epsilon.
        :return: inputs_norm: Normalized inputs. Shape(batch_size, num_hidden).
        """
        mean, var = self.moments(inputs)
        inputs_norm = mx.sym.broadcast_minus(inputs, mean, name='%sinp_minus_mean' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, mx.sym.rsqrt(var + eps), name='%sinp_norm' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, self.scale, name='%sinp_norm_scaled' % self.prefix)
        inputs_norm = mx.sym.broadcast_add(inputs_norm, self.shift, name='%sinp_norm_scaled_shifted' % self.prefix)
        return inputs_norm


class TransformerResidual:

    def __init__(self, num_hidden: int, layer_normalization: bool, dropout: float, prefix: str) -> None:
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.layer_norm = None
        self.prefix = prefix
        if layer_normalization:
            self.layer_norm = LayerNormalization(num_hidden=self.num_hidden, prefix=self.prefix)

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
        if self.layer_norm:
            self._reshape_and_normalize(z, length)
        return z

    def _reshape_and_normalize(self, data: mx.sym.Symbol, length: int):
        data = mx.sym.reshape(data, shape=(-3, self.num_hidden))
        data = self.layer_norm.normalize(data)
        data = mx.sym.reshape(data, shape=(-4, -1, length, self.num_hidden))
        return data


def split_heads(x: mx.sym.Symbol, length: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with head dimension folded into batch and depth divided by the number of heads.

    :param x: Symbol of shape (batch, length, depth).
    :param length: Sequence length.
    :param heads: Number of heads.
    :return: Symbol of shape (batch * heads, length, depth_per_heads).
    """
    # (batch, length, heads, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(0, length, heads, -1))
    # (batch, heads, length, depth/heads)
    x = mx.sym.transpose(data=x, axes=(0, 2, 1, 3))
    # (batch * heads, length, depth/heads)
    return mx.sym.reshape(data=x, shape=(-3, length, -1))


def combine_heads(x: mx.sym.Symbol, length: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with both batch & length, and head & depth dimensions combined.

    :param x: Symbol of shape (batch * heads, length, depth_per_head).
    :param length: Sequence length.
    :param heads: Number of heads.
    :return: Symbol of shape (batch * length, depth).
    """
    # (batch, heads, length, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(-4, -1, heads, length, 0))
    # (batch, length, heads, depth_per_head)
    x = mx.sym.transpose(x, axes=(0, 2, 1, 3))
    # (batch, length, depth)
    return mx.sym.reshape(x, shape=(-1, length, -3))


def broadcast_lengths(x: mx.sym.Symbol, heads: int) -> mx.sym.Symbol:
    """
    Broadcasts the length information of each sequence to multiple heads.

    :param x: Symbol(batch, 1)
    :param heads: Number of heads.
    :return: Symbol(batch * heads, 1)
    """
    # x: (batch, 1)
    x = mx.sym.expand_dims(x, axis=1)
    # x: (batch, heads)
    x = mx.sym.broadcast_to(x, shape=(0, heads))
    # x: (batch * heads, 1)
    x = mx.sym.reshape(x, shape=(-3,))
    return x


def dot_attention(queries, keys, values, length: mx.sym.Symbol, dropout: float = 0.0,
                  bias: Optional[mx.sym.Symbol] = None):
    """

    :param queries: (n, lq, d)
    :param keys: (n, lk, d)
    :param values: (n, lk, dv)
    :param length: (n,)
    :param dropout: Dropout probability.
    :param bias: (1, lq, lk)
    :return: (n, lq, dv)
    """
    # (n, lq, lk)
    logits = mx.sym.batch_dot(lhs=queries, rhs=keys, transpose_b=True)

    # mask lk dimension
    # (lk, n, lq)
    logits = mx.sym.transpose(data=logits, axes=(2, 0, 1))
    logits = mx.sym.SequenceMask(data=logits,
                                 use_sequence_length=True,
                                 sequence_length=length,
                                 value=-99999999.)
    # (n, lq, lk)
    logits = mx.sym.transpose(data=logits, axes=(1, 2, 0))

    if bias is not None:
        logits = mx.sym.broadcast_add(logits, bias)

    probs = mx.sym.softmax(logits, axis=-1)
    probs = mx.sym.Dropout(probs, p=dropout) if dropout > 0.0 else probs

    # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
    return mx.sym.batch_dot(lhs=probs, rhs=values)


class MultiHeadAttentionBase:

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        self.prefix = prefix
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.dropout = dropout
        self.depth_per_head = self.depth // self.heads

        self.w_h2o = mx.sym.Variable("%sh2o_weight" % prefix)
        self.b_h2o = mx.sym.Variable("%sh2o_bias" % prefix)

    def _attend(self,
                queries: mx.sym.Symbol,
                keys: mx.sym.Symbol,
                values: mx.sym.Symbol,
                lengths: mx.sym.Symbol,
                queries_max_length: int,
                memory_max_length: int,
                bias: Optional[mx.sym.Symbol] = None):
        # scale by sqrt(depth_per_head)
        queries *= self.depth_per_head ** -0.5

        # (batch*heads, length, depth/heads)
        queries = split_heads(queries, queries_max_length, self.heads)
        keys = split_heads(keys, memory_max_length, self.heads)
        values = split_heads(values, memory_max_length, self.heads)
        lengths = broadcast_lengths(lengths, self.heads)

        # (batch*heads, queries_max_length, depth_per_head)
        contexts = dot_attention(queries, keys, values, lengths, dropout=self.dropout, bias=bias)

        # (batch, queries_max_length, depth)
        contexts = combine_heads(contexts, queries_max_length, self.heads)

        if self.depth_out != self.depth:
            contexts = mx.sym.reshape(contexts, shape=(-3, -1))
            # contexts: (batch * queries_max_length, output_depth)
            contexts = mx.sym.FullyConnected(data=contexts,
                                             weight=self.w_h2o,
                                             bias=self.b_h2o,
                                             num_hidden=self.depth_out)
            # contexts: (batch, queries_max_length, output_depth)
            contexts = mx.sym.reshape(contexts, shape=(-1, queries_max_length, self.depth_out))
        return contexts


class MultiHeadSelfAttention(MultiHeadAttentionBase):

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout)

        self.w_i2h = mx.sym.Variable("%si2h_weight" % prefix)
        self.b_i2h = mx.sym.Variable("%si2h_bias" % prefix)

    def __call__(self,
                 inputs: mx.sym.Symbol,
                 lengths: mx.sym.Symbol,
                 max_length: int,
                 bias: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns a symbol of shape (batch, max_length, output_depth).

        :param inputs: Symbol of shape (batch, max_length, input_depth).
        :param lengths: Symbol of shape (batch, 1).
        :param max_length: Size of time dimension.
        :param bias: Symbol of shape (1, max_length, max_length).
        :return: Symbol of shape (batch, max_length, output_depth).
        """
        # combine batch and time dimension
        # inputs: (batch * max_length, num_hidden)
        inputs = mx.sym.reshape(data=inputs, shape=(-3, -1))

        # combined: (batch * max_length, depth * 3)
        combined = mx.sym.FullyConnected(data=inputs,
                                         weight=self.w_i2h,
                                         bias=self.b_i2h,
                                         num_hidden=self.depth * 3,
                                         name="%sqkv_transform" % self.prefix)
        # split batch and time dimension
        # combined: (batch, max_length, depth * 3)
        combined = mx.sym.reshape(data=combined, shape=(-1, max_length, self.depth * 3))

        # split into query, keys and values
        # (batch, max_length, depth)
        queries, keys, values = mx.sym.split(data=combined, num_outputs=3, axis=2)

        return self._attend(queries,
                            keys,
                            values,
                            lengths,
                            queries_max_length=max_length,
                            memory_max_length=max_length,
                            bias=bias)


class MultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout)

        # TODO: query and memory depth different?

        self.w_q2h = mx.sym.Variable("%sq2h_weight" % prefix)
        self.b_q2h = mx.sym.Variable("%sq2h_bias" % prefix)
        self.w_kv2h = mx.sym.Variable("%skv2h_weight" % prefix)
        self.b_kv2h = mx.sym.Variable("%skv2h_bias" % prefix)

    def __call__(self,
                 queries: mx.sym.Symbol,
                 queries_max_length: int,
                 memory: mx.sym.Symbol,
                 memory_lengths: mx.sym.Symbol,
                 memory_max_length: int) -> mx.sym.Symbol:
        """
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Symbol of shape (batch, queries_max_length, input_depth). TODO?
        :param queries_max_length: Size of queries time dimension.
        :param memory: Symbol of shape (batch, memory_max_length, input_depth).
        :param memory_lengths: Symbol of shape (batch, 1).
        :param memory_max_length: Size of memory time dimension.
        :return: Symbol of shape (batch, queries_max_length, output_depth).
        """
        # Combine batch and time dimension
        # inputs: (batch * memory_max_length, num_hidden)
        memory = mx.sym.reshape(data=memory, shape=(-3, -1))

        # (batch * memory_max_length, depth * 2)
        combined = mx.sym.FullyConnected(data=memory,
                                         weight=self.w_kv2h,
                                         bias=self.b_kv2h,
                                         num_hidden=self.depth * 2,
                                         name="%skv_transform" % self.prefix)

        # split batch and time dimension
        # (batch, memory_max_length, depth * 2)
        combined = mx.sym.reshape(data=combined, shape=(-1, memory_max_length, self.depth * 2))

        # split into query, keys and values
        # (batch, memory_max_length, depth)
        # NOTE: requires depth to be equal across all 2 parts.
        keys, values = mx.sym.split(data=combined, num_outputs=2, axis=2)

        queries = mx.sym.reshape(data=queries, shape=(-3, -1))
        # (batch * memory_max_length, depth * 2)
        queries = mx.sym.FullyConnected(data=queries,
                                        weight=self.w_q2h,
                                        bias=self.b_q2h,
                                        num_hidden=self.depth,
                                        name="%sq_transform" % self.prefix)
        # (batch, memory_max_length, depth)
        queries = mx.sym.reshape(data=queries, shape=(-1, queries_max_length, self.depth))

        return self._attend(queries,
                            keys,
                            values,
                            memory_lengths,
                            queries_max_length=queries_max_length,
                            memory_max_length=memory_max_length,
                            bias=None)


class FFNRelu:
    """
    Position-wise feed-forward network with ReLU activation.
    """

    def __init__(self, num_hidden: int, num_model: int, dropout: float, prefix: str):
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.prefix = prefix
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

    def __call__(self, x, length):
        """
        Position-wise feed-forward network with ReLU activation.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :param length: sequence length
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        # TODO: use a convolution to avoid needing to know the sequence length and reshapes?
        # FIXME reuse variables?
        x = mx.sym.reshape(x, shape=(-3, -1))
        h = mx.sym.FullyConnected(data=x, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h)
        h = mx.sym.Activation(h, act_type="relu")
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        y = mx.sym.FullyConnected(data=h, num_hidden=self.num_model, weight=self.w_h2o, bias=self.b_h2o)
        y = mx.sym.reshape(y, shape=(-1, length, self.num_model))
        return y

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

from typing import Tuple

import mxnet as mx


class LayerNormalization:
    """
    Implements Ba et al, Layer Normalization (https://arxiv.org/abs/1607.06450).
    TODO(fhieber): this should eventually go into MXNet.

    :param num_hidden: Number of hidden units.
    :param prefix: Prefix of layer name.
    :param gamma_init: Initial value of scaling variable.
    :param beta_init: Initial value of shifting variable.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str,
                 gamma_init: float = 1.0,
                 beta_init: float = 0.0) -> None:
        self.prefix = prefix
        self.scale = mx.sym.Variable('%sgamma' % prefix, shape=(num_hidden,), init=mx.init.Constant(value=gamma_init))
        self.shift = mx.sym.Variable('%sbeta' % prefix, shape=(num_hidden,), init=mx.init.Constant(value=beta_init))

    @staticmethod
    def moments(inputs: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Computes mean and variance of a Symbol across axis 1.

        :param inputs: Shape(batch_size, hidden).
        :return: mean, var: Shape(batch_size, 1)
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


class MultiHeadAttention:

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        self.prefix = prefix
        self.depth_att = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.dropout = dropout

        self.depth_per_head = self.depth_att // self.heads
        self.w_i2h = mx.sym.Variable("%si2h_weight" % prefix)
        self.b_i2h = mx.sym.Variable("%si2h_bias" % prefix)
        self.w_h2o = mx.sym.Variable("%sh2o_weight" % prefix)
        self.b_h2o = mx.sym.Variable("%sh2o_bias" % prefix)
        self.use_loop = False  # use naive loop

    def _split_heads(self, x: mx.sym.Symbol, length: int) -> mx.sym.Symbol:
        """
        Returns a symbol with head dimension folded into batch and depth divided by the number of heads.

        :param x: Symbol of shape (batch, length, depth).
        :return: Symbol of shape (batch * heads, length, depth_per_heads).
        """
        # (batch, length, heads, depth_per_head)
        x = mx.sym.reshape(data=x, shape=(0, length, self.heads, -1))
        # (batch, heads, length, depth/heads)
        x = mx.sym.transpose(data=x, axes=(0, 2, 1, 3))
        # (batch * heads, length, depth/heads)
        return mx.sym.reshape(data=x, shape=(-3, length, -1))

    def _combine_heads(self, x: mx.sym.Symbol, length: int) -> mx.sym.Symbol:
        """
        Returns a symbol with both batch & length, and head & depth dimensions combined.

        :param x: Symbol of shape (batch * heads, length, depth_per_head).
        :return: Symbol of shape (batch * length, depth).
        """
        # (batch, heads, length, depth_per_head)
        x = mx.sym.reshape(data=x, shape=(-4, -1, self.heads, length, 0))
        # (batch, length, heads, depth_per_head)
        x = mx.sym.transpose(x, axes=(0, 2, 1, 3))
        # (batch * length, depth)
        return mx.sym.reshape(x, shape=(-3, -3))

    def _broadcast_lengths(self, x: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Broadcasts the length information of each sequence to multiple heads.

        :param x: Symbol(batch, 1)
        :return: Symbol(batch * heads, 1)
        """
        # x: (batch, 1)
        x = mx.sym.expand_dims(x, axis=1)
        # x: (batch, heads)
        x = mx.sym.broadcast_to(x, shape=(0, self.heads))
        # x: (batch * heads, 1)
        x = mx.sym.reshape(x, shape=(-3,))
        return x

    def on(self, inputs: mx.sym.Symbol, inputs_length: mx.sym.Symbol, length: int) -> mx.sym.Symbol:
        """
        Returns a symbol of shape (batch, length, output_depth).

        :param inputs: Symbol of shape (batch, length, input_depth).
        :param inputs_length: Symbol of shape (batch, 1).
        :param length: Size of time dimension.
        :return: Symbol of shape (batch, length, output_depth).
        """
        # lets do self attention first (code is constrained to q length == k length)
        # inputs: (batch, length, num_hidden)

        # Combine batch and time dimension
        # inputs: (batch * length, num_hidden)
        inputs = mx.sym.reshape(data=inputs, shape=(-3, -1))

        # project with 1 large matrix
        # combined: (batch * length, depth * 3)
        combined = mx.sym.FullyConnected(data=inputs,
                                         weight=self.w_i2h,
                                         bias=self.b_i2h,
                                         num_hidden=self.depth_att * 3,
                                         name="%sqkv_transform" % self.prefix)

        # split batch and time dimension
        # combined: (batch, length, depth * 3)
        combined = mx.sym.reshape(data=combined, shape=(-1, length, self.depth_att * 3))

        # split into query, keys and values
        # q/k/v: (batch, length, depth)
        # NOTE: requires depth to be equal across all 3 parts.
        q, k, v = mx.sym.split(data=combined, num_outputs=3, axis=2)

        # q/k/v: (batch * heads, length, depth/heads)
        q = self._split_heads(q, length)
        k = self._split_heads(k, length,)
        v = self._split_heads(v, length)
        # scale by sqrt(depth_per_head)
        q *= self.depth_per_head ** -0.5

        # (batch*heads, length, depth_per_head) X (batch*heads, depth_per_head, length)
        #   -> (batch*heads, length, length)
        logits = mx.sym.batch_dot(lhs=q, rhs=k, transpose_b=True)

        # mask variable sequence-length (unfortunately requires time-major data)
        # logits: (length, batch * heads, length)
        logits = mx.sym.swapaxes(data=logits, dim1=0, dim2=1)
        logits = mx.sym.SequenceMask(data=logits,
                                     use_sequence_length=True,
                                     sequence_length=self._broadcast_lengths(inputs_length),
                                     value=-99999999.)
        # logits: (batch * heads, length, length)
        logits = mx.sym.swapaxes(data=logits, dim1=0, dim2=1)

        # weights: (batch * heads, length, length)
        weights = mx.sym.softmax(logits)

        if self.dropout > 0.0:
            weights = mx.sym.Dropout(weights, p=self.dropout)

        if self.use_loop:
            contexts = []
            # weights: length * (batch * heads, 1, length)
            weights = mx.sym.split(weights, axis=1, num_outputs=length, squeeze_axis=False)
            for t in range(length):
                # (_, 1, length) * (_, length, depth_per_head) -> (X, 1, depth_per_head)
                context_t = mx.sym.batch_dot(lhs=weights[t], rhs=v)
                # context_t: (batch * heads, 1, depth/heads
                contexts.append(context_t)
            # contexts: (batch_size * heads, length, depth_per_head)
            contexts = mx.sym.concat(*contexts, dim=1)
        else:
            # contexts: (B*H, L, L) X (B*H, L, D) –> (B*H, L, D).
            # contexts: (batch * heads, length, depth_per_head)
            contexts = mx.sym.batch_dot(lhs=weights, rhs=v)

        # contexts: (batch * length, depth)
        contexts = self._combine_heads(contexts, length)

        # contexts: (batch * length, output_depth)
        contexts = mx.sym.FullyConnected(data=contexts,
                                         weight=self.w_h2o,
                                         bias=self.b_h2o,
                                         num_hidden=self.depth_out)
        # contexts: (batch, length, output_depth)
        return mx.sym.reshape(contexts, shape=(-1, length, self.depth_out))


class FFNRelu:
    """
    Position-wise feed-forward network with ReLU activation.
    """

    def __init__(self, prefix: str, num_hidden: int = 2048, num_model: int = 512, dropout: float = 0.0):
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

    def apply(self, x, length):
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

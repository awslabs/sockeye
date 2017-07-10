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
        self.scale = mx.sym.Variable('%s_gamma' % prefix, shape=(num_hidden,), init=mx.init.Constant(value=gamma_init))
        self.shift = mx.sym.Variable('%s_beta' % prefix, shape=(num_hidden,), init=mx.init.Constant(value=beta_init))

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
        inputs_norm = mx.sym.broadcast_minus(inputs, mean, name='%s_inp_minus_mean' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, mx.sym.rsqrt(var + eps), name='%s_inp_norm' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, self.scale, name='%s_inp_norm_scaled' % self.prefix)
        inputs_norm = mx.sym.broadcast_add(inputs_norm, self.shift, name='%s_inp_norm_scaled_shifted' % self.prefix)
        return inputs_norm


def _split_heads(x, length, heads):
    # (batch, length, depth) -> (batch, length, heads, depth/heads)
    x = mx.sym.reshape(data=x, shape=(0, length, heads, -1))
    # (batch, heads, length, depth/heads)
    x =  mx.sym.transpose(data=x, shape=(0, 2, 1, 3))
    # (batch * heads, length, depth/heads)
    return mx.sym.reshape(data=x, shape=(-3, length, -1))


def multihead_attention(inputs,
                        length,
                        depth=512,
                        heads=8,
                        output_depth=512,
                        dropout = 0.0):
    """
    Multi-head self attention. WIP.

    :param inputs:
    :param length:
    :param depth:
    :param heads:
    :param output_depth:
    :param dropout:
    :return: contexts: (batch, length, output_depth)
    """
    # lets do self attention first (code is constrained to q length == k length)
    # inputs: (batch, length, num_hidden)

    # Combine batch and time dimension
    # inputs: (batch * length, num_hidden)
    inputs = mx.sym.reshape(data=inputs, shape=(-3, -1))

    # project with 1 large matrix
    # combined: (batch * length, depth * 3)
    combined = mx.sym.FullyConnected(data=inputs, weight=w_in, no_bias=True, num_hidden=depth * 3)

    # split batch and time dimension
    # combined: (batch, length, depth * 3)
    combined = mx.sym.reshape(data=combined, shape=(-1, length, depth * 3))

    # split into query, keys and values
    # q/k/v: (batch, length, depth)
    # NOTE: requires depth to be equal across all 3 parts.
    q, k, v = mx.sym.split(data=combined, num_outputs=3, axis=2)

    # q/k/v: (batch * heads, length, depth/heads)
    q = _split_heads(q, length, heads)
    k = _split_heads(k, length, heads)
    v = _split_heads(v, length, heads)
    depth_per_head = depth // heads
    q *= depth_per_head**-0.5

    # (batch * heads, length, depth/heads) X (batch * heads, depth/heads, length) = (batch * heads, length, length)
    # (B, L, D) X (B, D, L) = (B, L, L)
    logits = mx.sym.batch_dot(lhs=q, rhs=k, transpose_b=True)
    # TODO masking. SequenceMask requires time-major....

    # weights: (batch * heads, length, length)
    weights = mx.sym.softmax(logits)

    if dropout > 0.0:
        weights = mx.sym.Dropout(weights, p=dropout)

    if True:
        # lets try naive approach first (stacked for loop)
        contexts = []
        # weights: length * (batch * heads, 1, length)
        weights = mx.sym.split(weights, axis=1, num_outputs=length, squeeze_axis=False)
        for t in range(length):
            # w_t: (batch * heads, 1, length)
            w_t = weights[t]
            # w_t * v = c_t
            # (_, 1, length) * (_, length, depth/heads) -> (X, 1, depth/heads)
            context_t = mx.sym.batch_dot(lhs=w_t, rhs=v)
            # context_t: (batch * heads, 1, depth/heads
            contexts.append(context_t)
        # contexts: (batch_size * heads, length, depth/heads)
        contexts = mx.sym.concat(*contexts, dim=1)
    else:
        # TODO check if this is correct
        # weights: (batch * heads, length, 1)
        weights = mx.sym.expand_dims(data=weights, axis=2)
        # weights: (batch * heads, length, length)
        weights = mx.sym.broadcast_to(data=weights, shape=(0, length, length))
        # contexts: (B*H, L, L) X (B*H, L, D) –> (B*H, L, D).
        # contexts: (batch * heads, length, depth/heads)
        contexts = mx.sym.batch_dot(lhs=weights, rhs=v)

    # separate out heads
    # contexts: (batch, heads, length, depth/heads)
    contexts = mx.sym.reshape(data=contexts, shape=(-4, -1, heads, length, 0))
    # contexts: (batch, length, heads, depth/heads)
    contexts = mx.sym.transpose(contexts, shape=(0, 2, 1, 3))
    # contexts: (batch * length, depth)
    contexts = mx.sym.reshape(contexts, shape=(-3, -3))

    # contexts: (batch * length, output_depth)
    contexts = mx.sym.FullyConnected(contexts, num_hidden=output_depth)
    # contexts: (batch, length, output_depth)
    contexts = mx.sym.reshape(contexts, shape=(-1, length, output_depth))

    return contexts


class FFNRelu:
    """
    Position-wise feed-forward network with ReLU activation.
    """

    def __init__(self, num_hidden: int = 2014, num_model: int = 512, dropout: float = 0.0):
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.w_i2h = mx.sym.Variable('i2h_weight')
        self.b_i2h = mx.sym.Variable('i2h_bias')
        self.w_h2o = mx.sym.Variable('h2o_weight')
        self.b_h2o = mx.sym.Variable('h2o_bias')

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
        h = mx.sym.FullyConnected(x, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h)
        h = mx.sym.Activation(h, act_type="relu")
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        y = mx.sym.FullyConnected(h, num_hidden=self.num_model, weight=self.w_h2o, bias=self.b_h2o)
        y = mx.sym.reshape(y, shape=(-1, length, self.num_model))
        return y




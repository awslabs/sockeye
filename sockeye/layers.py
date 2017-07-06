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




def split_heads(x, length, num_heads):
    # (batch, length, depth) -> (batch, length, num_heads, depth/num_heads)
    ret = mx.sym.reshape(data=x, shape=(0, length, num_heads, -1))
    # (batch, num_heads, length, depth/num_heads)
    ret =  mx.sym.transpose(data=ret, shape=(0, 2, 1, 3))
    # (batch * num_heads, length, depth/num_heads)
    return mx.sym.reshape(data=ret, shape=(-3, length, -1))


w_in = mx.sym.Variable('project_in')

def multihead_attention(inputs,
                        length,
                        num_hidden,
                        depth=512,
                        num_heads=8,
                        ):
    # lets do self attention first (code is constrained to q length == k length)
    # inputs: (batch, seq_len, num_hidden)

    # Combine batch and time dimension
    # inputs: (batch * seq_len, num_hidden)
    inputs = mx.sym.reshape(data=inputs, shape=(-3, -1))

    # project with 1 large matrix
    # combined: (batch * seq_len, depth * 3)
    combined = mx.sym.FullyConnected(data=inputs, weight=w_in, no_bias=True, num_hidden=depth * 3)

    # split batch and time dimension
    # combined: (batch, seq_len, depth * 3)
    combined = mx.sym.reshape(data=combined, shape=(-1, length, depth * 3))

    # split into query, keys and values
    # q/k/v: (batch, seq_len, depth) NOTE: requires depth to be equal across all 3 parts (tf doesnt require this)
    q, k, v = mx.sym.split(data=combined, num_outputs=3, axis=2)

    #
    # q/k/v: (batch * num_heads, seq_len, depth/num_heads)
    q = split_heads(q, length, num_heads)
    k = split_heads(k, length, num_heads)
    v = split_heads(v, length, num_heads)
    depth_per_head = depth // num_heads
    q *= depth_per_head**-0.5

    # (batch * num_heads, seq_len, depth/num_heads) X (batch * num_heads, depth/num_heads, seq_len) = (batch * num_heads, seq_len)
    logits = mx.sym.batch_dot(lhs=q, rhs=k, transpose_b=True)
    # TODO masking. SequenceMask requires time-major....

    # weights: (batch * num_heads, seq_len)
    weights = mx.sym.softmax(logits)
    # TODO dropout?

    # weights: (batch * num_heads, seq_len, 1)
    weights = mx.sym.expand_dims(data=weights, axis=2)
    # weights: (batch * num_heads, seq_len, seq_len)
    weights = mx.sym.broadcast_to(data=weights, shape=(0, length, length))  # TODO: correct?!

    # contexts: (B*H, L, L) X (B*H, L, D) –> (B*H, L, D).
    # contexts: (batch * num_heads, seq_len, depth/num_heads)
    contexts = mx.sym.batch_dot(lhs=weights, rhs=v)

    # contexts: (batch, num_heads, seq_len, depth/num_heads)
    contexts = mx.sym.reshape(data=contexts, shape=(-4, -1, num_heads, length, 0))

    # batch dot should give: batch, num_heads, seq_len, depth / num_heads)



    #context = mx.sym.reshape(data=context, shape=(0, 0))








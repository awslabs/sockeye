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

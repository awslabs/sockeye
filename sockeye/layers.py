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

import logging
import math
from typing import Dict, Optional, Tuple, Union

import mxnet as mx
import numpy as np

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


def activation(data: mx.sym.Symbol, act_type: str) -> mx.sym.Symbol:
    """
    Apply custom or standard activation.

    Custom activation types include:
     - Swish-1, also called Sigmoid-Weighted Linear Unit (SiLU): Ramachandran et
       al. (https://arxiv.org/pdf/1710.05941.pdf), Elfwing et al.
       (https://arxiv.org/pdf/1702.03118.pdf)
     - Gaussian Error Linear Unit (GELU): Hendrycks and Gimpel
       (https://arxiv.org/pdf/1606.08415.pdf)

    :param data: input Symbol of any shape.
    :param act_type: Type of activation.
    :return: output Symbol with same shape as input.
    """
    # TODO: Contribute these to MXNet?  For now it appears that registered activation types must be implemented in C++.
    if act_type == C.SWISH1:
        return data * mx.sym.Activation(data, act_type="sigmoid")
    elif act_type == C.GELU:
        # Approximation of x * gaussian_cdf(x) used by Hendrycks and Gimpel
        return 0.5 * data * (1 + mx.sym.Activation((math.sqrt(2 / math.pi) * (data + (0.044715 * (data**3)))),
                                                   act_type="tanh"))
    else:
        return mx.sym.Activation(data, act_type=act_type)


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
        Computes mean and variance of the last dimension of a Symbol.

        :param inputs: Shape: (d0, ..., dn, hidden).
        :return: mean, var: Shape: (d0, ..., dn, 1).
        """
        mean = mx.sym.mean(data=inputs, axis=-1, keepdims=True)
        # TODO(fhieber): MXNet should have this.
        var = mx.sym.mean(mx.sym.square(mx.sym.broadcast_minus(inputs, mean)), axis=-1, keepdims=True)
        return mean, var

    def normalize(self, inputs: mx.sym.Symbol, eps: float = 0.000001) -> mx.sym.Symbol:
        """
        Normalizes hidden units of inputs as follows:

        inputs = scale * (inputs - mean) / sqrt(var + eps) + shift

        Normalization is performed over the last dimension of the input data.

        :param inputs: Inputs to normalize. Shape: (d0, ..., dn, num_hidden).
        :param eps: Variance epsilon.
        :return: inputs_norm: Normalized inputs. Shape: (d0, ..., dn, num_hidden).
        """
        mean, var = self.moments(inputs)
        inputs_norm = mx.sym.broadcast_minus(inputs, mean, name='%sinp_minus_mean' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, mx.sym.rsqrt(var + eps), name='%sinp_norm' % self.prefix)
        inputs_norm = mx.sym.broadcast_mul(inputs_norm, self.scale, name='%sinp_norm_scaled' % self.prefix)
        inputs_norm = mx.sym.broadcast_add(inputs_norm, self.shift, name='%sinp_norm_scaled_shifted' % self.prefix)
        return inputs_norm


class LHUC:
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    :param weight: Optional parameter vector.
    :param prefix: Optional prefix for created parameters (if not given as weight).
    """
    def __init__(self,
                 num_hidden: int,
                 weight: Optional[mx.sym.Symbol] = None,
                 prefix: str = "") -> None:
        self.num_hidden = num_hidden
        self.prefix = prefix
        if weight is None:
            self.params = mx.sym.Variable(self.prefix + C.LHUC_NAME,
                                          shape=(self.num_hidden,),
                                          init=mx.init.Uniform(0.1),
                                          dtype="float32")
        else:
            self.params = weight

    def apply(self,
              inputs: mx.sym.Symbol,
              name: Optional[str] = None) -> mx.sym.Symbol:

        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight_vector = 2 * mx.sym.Activation(data=self.params, act_type="sigmoid")
        out = mx.sym.broadcast_mul(weight_vector, inputs, name=name)

        return out


class WeightNormalization:
    """
    Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
    For a given tensor the normalization is done per hidden dimension.

    :param weight: Weight tensor of shape: (num_hidden, d1, d2, ...).
    :param num_hidden: Size of the first dimension.
    :param ndim: The total number of dimensions of the weight tensor.
    :param prefix: The prefix used for naming.
    """

    def __init__(self, weight, num_hidden, ndim=2, prefix: str = '') -> None:
        self.prefix = prefix
        self.weight = weight
        self.num_hidden = num_hidden
        self.scale = mx.sym.Variable("%swn_scale" % prefix,
                                     shape=tuple([num_hidden] + [1] * (ndim - 1)),
                                     init=mx.init.Constant(value=1.0))

    def __call__(self, weight: Optional[mx.nd.NDArray] = None, scale: Optional[mx.nd.NDArray] = None) -> mx.sym.Symbol:
        """
        Normalize each hidden dimension and scale afterwards

        :return: A weight normalized weight tensor.
        """
        if weight is None and scale is None:
            return mx.sym.broadcast_mul(lhs=mx.sym.L2Normalization(self.weight, mode='instance'),
                                        rhs=self.scale, name="%swn_scale" % self.prefix)
        else:
            assert isinstance(weight, mx.nd.NDArray)
            assert isinstance(scale, mx.nd.NDArray)
            return mx.nd.broadcast_mul(lhs=mx.nd.L2Normalization(weight, mode='instance'), rhs=scale)


class OutputLayer:
    """
    Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.

    :param hidden_size: Decoder hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight_normalization: Whether to apply weight normalization.
    :param prefix: Prefix used for naming.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 weight: Optional[mx.sym.Symbol],
                 weight_normalization: bool,
                 prefix: str = C.DEFAULT_OUTPUT_LAYER_PREFIX) -> None:
        self.vocab_size = vocab_size
        self.prefix = prefix

        if weight is None:
            self.w = mx.sym.Variable("%sweight" % self.prefix, shape=(vocab_size, hidden_size))
        else:
            self.w = weight

        self.weight_normalization = weight_normalization
        if weight_normalization:
            logger.info("Normalizing output layer weights.")
            self.weight_norm = WeightNormalization(self.w,
                                                   num_hidden=vocab_size,
                                                   ndim=2,
                                                   prefix=self.prefix)
            self.w = self.weight_norm()

        self.b = mx.sym.Variable("%sbias" % self.prefix)

    def __call__(self,
                 hidden: Union[mx.sym.Symbol, mx.nd.NDArray],
                 weight: Optional[mx.nd.NDArray] = None,
                 bias: Optional[mx.nd.NDArray] = None):
        """
        Linear transformation to vocab size. Returns logits.

        :param hidden: Decoder representation for n elements. Shape: (n, self.num_hidden).
        :return: Logits. Shape(n, self.vocab_size).
        """
        if isinstance(hidden, mx.sym.Symbol):
            # TODO dropout?
            return mx.sym.FullyConnected(data=hidden,
                                         num_hidden=self.vocab_size,
                                         weight=self.w,
                                         bias=self.b,
                                         flatten=False,
                                         name=C.LOGITS_NAME)

        # Equivalent NDArray implementation (requires passed weights/biases)
        assert isinstance(hidden, mx.nd.NDArray)
        utils.check_condition(weight is not None and bias is not None,
                              "OutputLayer NDArray implementation requires passing weight and bias NDArrays.")

        return mx.nd.FullyConnected(data=hidden,
                                    num_hidden=bias.shape[0],
                                    weight=weight,
                                    bias=bias,
                                    flatten=False)


def split_heads(x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with head dimension folded into batch and depth divided by the number of heads.

    :param x: Symbol of shape (batch, length, depth).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch * heads, length, depth_per_heads).
    """
    # (batch, length, heads, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(0, -1, heads, depth_per_head))
    # (batch, heads, length, depth/heads)
    x = mx.sym.transpose(data=x, axes=(0, 2, 1, 3))
    # (batch * heads, length, depth/heads)
    return mx.sym.reshape(data=x, shape=(-3, -1, depth_per_head))


def combine_heads(x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with both batch & length, and head & depth dimensions combined.

    :param x: Symbol of shape (batch * heads, length, depth_per_head).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch, length, depth).
    """
    # (batch, heads, length, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(-4, -1, heads, 0, depth_per_head))
    # (batch, length, heads, depth_per_head)
    x = mx.sym.transpose(x, axes=(0, 2, 1, 3))
    # (batch, length, depth)
    return mx.sym.reshape(x, shape=(-1, 0, depth_per_head * heads))


def broadcast_to_heads(x: mx.sym.Symbol, num_heads: int, ndim: int, fold_heads: bool = True) -> mx.sym.Symbol:
    """
    Broadcasts batch-major input of shape (batch, d1 ... dn-1) to (batch*heads, d1 ... dn-1).

    :param x: Batch-major input. Shape: (batch, d1 ... dn-1).
    :param num_heads: Number of heads.
    :param ndim: Number of dimensions in x.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :return: Tensor with each sample repeated heads-many times.
             Shape: (batch * heads, d1 ... dn-1) if fold_heads == True, (batch, heads, d1 ... dn-1) else.
    """
    dims = [0] * (ndim - 1)
    # x: (batch, 1)
    x = mx.sym.expand_dims(x, axis=1)
    # x: (batch, heads, dims...)
    x = mx.sym.broadcast_to(x, shape=[0, num_heads] + dims)
    if fold_heads:
        # (batch * heads, dims...)
        return mx.sym.reshape(x, shape=[-3] + dims)
    else:
        # x: (batch, heads, dims...)
        return x


def dot_attention(queries: mx.sym.Symbol,
                  keys: mx.sym.Symbol,
                  values: mx.sym.Symbol,
                  lengths: Optional[mx.sym.Symbol] = None,
                  dropout: float = 0.0,
                  bias: Optional[mx.sym.Symbol] = None,
                  prefix: Optional[str] = ''):
    """
    Computes dot attention for a set of queries, keys, and values.

    :param queries: Attention queries. Shape: (n, lq, d).
    :param keys: Attention keys. Shape: (n, lk, d).
    :param values: Attention values. Shape: (n, lk, dv).
    :param lengths: Optional sequence lengths of the keys. Shape: (n,).
    :param dropout: Dropout probability.
    :param bias: Optional 3d bias tensor.
    :param prefix: Optional prefix
    :return: 'Context' vectors for each query. Shape: (n, lq, dv).
    """
    utils.check_condition(lengths is not None or bias is not None,
                          "Must provide either length or bias argument for masking")

    # (n, lq, lk)
    logits = mx.sym.batch_dot(lhs=queries, rhs=keys, transpose_b=True, name='%sdot' % prefix)

    if lengths is not None:
        # mask lk dimension
        # (lk, n, lq)
        logits = mx.sym.transpose(data=logits, axes=(2, 0, 1))
        logits = mx.sym.SequenceMask(data=logits,
                                     use_sequence_length=True,
                                     sequence_length=lengths,
                                     value=C.LARGE_NEGATIVE_VALUE)
        # (n, lq, lk)
        logits = mx.sym.transpose(data=logits, axes=(1, 2, 0))

    if bias is not None:
        logits = mx.sym.broadcast_add(logits, bias, name='%sbias_add' % prefix)

    probs = mx.sym.softmax(logits, axis=-1)
    probs = mx.sym.Dropout(probs, p=dropout) if dropout > 0.0 else probs

    # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
    return mx.sym.batch_dot(lhs=probs, rhs=values, name='%scontexts' % prefix)


class MultiHeadAttentionBase:
    """
    Base class for Multi-head attention.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """
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

    def _attend(self,
                queries: mx.sym.Symbol,
                keys: mx.sym.Symbol,
                values: mx.sym.Symbol,
                lengths: Optional[mx.sym.Symbol] = None,
                bias: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (batch_size, query_max_length, depth).
        :param keys: Keys. Shape: (batch_size, memory_max_length, depth).
        :param values: Values. Shape: (batch_size, memory_max_length, depth).
        :param lengths: Optional lengths of keys. Shape: (batch_size,).
        :param bias: Optional 3d bias.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """
        # scale by sqrt(depth_per_head)
        queries = queries * (self.depth_per_head ** -0.5)

        # (batch*heads, length, depth/heads)
        queries = split_heads(queries, self.depth_per_head, self.heads)
        keys = split_heads(keys, self.depth_per_head, self.heads)
        values = split_heads(values, self.depth_per_head, self.heads)
        lengths = broadcast_to_heads(lengths, self.heads, ndim=1, fold_heads=True) if lengths is not None else lengths

        # (batch*heads, query_max_length, depth_per_head)
        contexts = dot_attention(queries, keys, values,
                                 lengths=lengths, dropout=self.dropout, bias=bias, prefix=self.prefix)

        # (batch, query_max_length, depth)
        contexts = combine_heads(contexts, self.depth_per_head, self.heads)

        # contexts: (batch, query_max_length, output_depth)
        contexts = mx.sym.FullyConnected(data=contexts,
                                         weight=self.w_h2o,
                                         no_bias=True,
                                         num_hidden=self.depth_out,
                                         flatten=False)

        return contexts


class MultiHeadSelfAttention(MultiHeadAttentionBase):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """
    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout)
        self.w_i2h = mx.sym.Variable("%si2h_weight" % prefix)

    def __call__(self,
                 inputs: mx.sym.Symbol,
                 input_lengths: Optional[mx.sym.Symbol] = None,
                 bias: Optional[mx.sym.Symbol] = None,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a symbol of shape (batch, max_length, output_depth).

        :param inputs: Input Data. Shape: (batch, max_length, input_depth).
        :param input_lengths: Optional lengths of inputs to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param cache: Optional dictionary of previously computed keys and values.
        :return: Symbol of shape (batch, max_length, output_depth).
        """
        # combined: (batch, max_length, depth * 3)
        combined = mx.sym.FullyConnected(data=inputs,
                                         weight=self.w_i2h,
                                         no_bias=True,
                                         num_hidden=self.depth * 3,
                                         flatten=False,
                                         name="%sqkv_transform" % self.prefix)
        # split into query, keys and values
        # (batch, max_length, depth)
        # pylint: disable=unbalanced-tuple-unpacking
        queries, keys, values = mx.sym.split(data=combined, num_outputs=3, axis=2)

        if cache is not None:
            # append new keys & values to cache, update the cache
            keys = cache['k'] = keys if cache['k'] is None else mx.sym.concat(cache['k'], keys, dim=1)
            values = cache['v'] = values if cache['v'] is None else mx.sym.concat(cache['v'], values, dim=1)

        return self._attend(queries,
                            keys,
                            values,
                            lengths=input_lengths,
                            bias=bias)


class MultiHeadAttention(MultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout)
        self.w_q2h = mx.sym.Variable("%sq2h_weight" % prefix)
        self.w_k2h = mx.sym.Variable("%sk2h_weight" % prefix)
        self.w_v2h = mx.sym.Variable("%sv2h_weight" % prefix)

    def __call__(self,
                 queries: mx.sym.Symbol,
                 memory: mx.sym.Symbol,
                 memory_lengths: Optional[mx.sym.Symbol] = None,
                 bias: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Query tensor. Shape: (batch, query_max_length, input_depth).
        :param memory: Memory data to attend to. Shape: (batch, memory_max_length, input_depth).
        :param memory_lengths: Optional lengths of memory to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :return: Symbol of shape (batch, query_seq_len, output_depth).
        """
        # (batch, query_max_length, depth)
        queries = mx.sym.FullyConnected(data=queries,
                                        weight=self.w_q2h,
                                        no_bias=True,
                                        num_hidden=self.depth,
                                        flatten=False,
                                        name="%sq_transform" % self.prefix)

        # (batch, memory_max_length, depth)
        keys = mx.sym.FullyConnected(data=memory,
                                     weight=self.w_k2h,
                                     no_bias=True,
                                     num_hidden=self.depth,
                                     flatten=False,
                                     name="%sk_transform" % self.prefix)

        # (batch, memory_max_length, depth)
        values = mx.sym.FullyConnected(data=memory,
                                       weight=self.w_v2h,
                                       no_bias=True,
                                       num_hidden=self.depth,
                                       flatten=False,
                                       name="%sv_transform" % self.prefix)

        return self._attend(queries,
                            keys,
                            values,
                            bias=bias,
                            lengths=memory_lengths)


class ProjectedDotAttention:
    """
    Dot attention layer for queries independent from keys/values.

    :param prefix: Attention prefix.
    :param num_hidden: Attention depth / number of hidden units.
    """

    def __init__(self,
                 prefix: str,
                 num_hidden) -> None:
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.w_q2h = mx.sym.Variable("%sq2h_weight" % prefix)
        self.b_q2h = mx.sym.Variable("%sq2h_bias" % prefix)
        self.w_kv2h = mx.sym.Variable("%skv2h_weight" % prefix)
        self.b_kv2h = mx.sym.Variable("%skv2h_bias" % prefix)

    def __call__(self,
                 queries: mx.sym.Symbol,
                 memory: mx.sym.Symbol,
                 memory_lengths: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Apply project, apply dot attention and return new context vectors.

        :param queries: Symbol of shape (batch, queries_max_length, input_num_hidden).
        :param memory: Symbol of shape (batch, memory_max_length, input_num_hidden).
        :param memory_lengths: Symbol of shape (batch, 1).
        :return: Symbol of shape (batch, queries_max_length, num_hidden).
        """
        # (batch, memory_max_length, num_hidden * 2)
        combined = mx.sym.FullyConnected(data=memory,
                                         weight=self.w_kv2h,
                                         bias=self.b_kv2h,
                                         num_hidden=self.num_hidden * 2,
                                         flatten=False,
                                         name="%skv_transform" % self.prefix)

        # split into keys and values
        # pylint: disable=unbalanced-tuple-unpacking
        keys, values = mx.sym.split(data=combined, num_outputs=2, axis=2)

        # (batch, queries_max_length, num_hidden)
        queries = mx.sym.FullyConnected(data=queries,
                                        weight=self.w_q2h,
                                        bias=self.b_q2h,
                                        num_hidden=self.num_hidden,
                                        flatten=False,
                                        name="%sq_transform" % self.prefix)
        # scale by sqrt(num_hidden)
        queries = queries * (self.num_hidden ** -0.5)

        # (batch, queries_max_length, num_hidden)
        contexts = dot_attention(queries, keys, values, memory_lengths)

        return contexts


class PlainDotAttention:
    """
    Dot attention layer for queries independent from keys/values.
    """

    def __call__(self,
                 queries: mx.sym.Symbol,
                 memory: mx.sym.Symbol,
                 memory_lengths: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Symbol of shape (batch, queries_max_length, input_depth).
        :param memory: Symbol of shape (batch, memory_max_length, input_depth).
        :param memory_lengths: Symbol of shape (batch, 1).
        :return: Symbol of shape (batch, queries_max_length, output_depth).
        """

        # (batch*heads, queries_max_length, depth_per_head)
        contexts = dot_attention(queries, memory, memory, memory_lengths)

        return contexts


class PositionalEncodings(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, max_seq_len, num_embed)
    with positional encodings as in Vaswani et al, 2017.

    :param length: Maximum sequence length.
    :param depth: Embedding size.
    """

    def __init__(self, length: int, depth: int) -> None:
        super().__init__()
        self.encodings = self.get_encodings(length, depth)

    @staticmethod
    def get_encodings(length, depth) -> np.ndarray:
        utils.check_condition(depth % 2 == 0, "Positional embeddings require an even embedding size it "
                                              "is however %d." % depth)
        # (1, depth)
        channels = np.arange(depth // 2).reshape((1, -1))

        # (length, 1)
        positions = np.arange(0, length).reshape((-1, 1))
        scaled_positions = positions / np.power(10000, (2 * channels) / depth)
        # sinusoids:
        sin = np.sin(scaled_positions)
        # cosines:
        cos = np.cos(scaled_positions)
        # interleave: (1, length, num_embed)
        encodings = np.hstack([sin, cos]).reshape(1, length, depth)
        return encodings

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.encodings)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("positional_encodings")
class PositionalEncodingsProp(mx.operator.CustomOpProp):

    def __init__(self, length: str, depth: str) -> None:
        super().__init__()
        self.length = int(length)
        self.depth = int(depth)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.depth)], []

    def infer_type(self, in_type):
        return [], [np.float32], []

    def create_operator(self, ctx, shapes, dtypes):
        return PositionalEncodings(length=self.length, depth=self.depth)

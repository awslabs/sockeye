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
from typing import Dict, Optional, Union

import mxnet as mx

from . import config
from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


class GeLU(mx.gluon.HybridBlock):

    def __init__(self, prefix=''):
        super().__init__(prefix=prefix)
        with self.name_scope():
            self.act = mx.gluon.nn.Activation(activation="tanh")

    def hybrid_forward(self, F, x):
        # Approximation of x * gaussian_cdf(x) used by Hendrycks and Gimpel
        return 0.5 * x * (1 + self.act((math.sqrt(2 / math.pi) * (x + (0.044715 * (x ** 3))))))


def get_activation(act_type: str) -> mx.gluon.Block:
    """
    Returns Gluon Block for given activation type.

    Custom activation types include:
     - Swish-1, also called Sigmoid-Weighted Linear Unit (SiLU): Ramachandran et
       al. (https://arxiv.org/pdf/1710.05941.pdf), Elfwing et al.
       (https://arxiv.org/pdf/1702.03118.pdf)
     - Gaussian Error Linear Unit (GELU): Hendrycks and Gimpel
       (https://arxiv.org/pdf/1606.08415.pdf)

    :param act_type: Type of activation.
    :return: output Symbol with same shape as input.
    """
    if act_type == C.SWISH1:
        return mx.gluon.nn.Swish()
    elif act_type == C.GELU:
        return GeLU()
    else:
        return mx.gluon.nn.Activation(activation=act_type)


# TODO: remove with next major version update to use mx.gluon.nn.LayerNorm (which uses different parameter naming).
class LayerNormalization(mx.gluon.nn.HybridBlock):
    """
    Implements Ba et al, Layer Normalization (https://arxiv.org/abs/1607.06450).

    Normalizes hidden units of data as follows:

    data = scale * (data - mean) / sqrt(var + eps) + shift

    Normalization is performed over the last dimension of the input data.

    :param prefix: Optional prefix of layer name.
    :param scale: Optional variable for scaling of shape (num_hidden,). Will be created if None.
    :param shift: Optional variable for shifting of shape (num_hidden,). Will be created if None.
    :param scale_init: Initial value of scale variable if scale is None. Default 1.0.
    :param shift_init: Initial value of shift variable if shift is None. Default 0.0.
    """
    def __init__(self,
                 prefix: str = 'layernorm',
                 scale: Optional[mx.sym.Symbol] = None,
                 shift: Optional[mx.sym.Symbol] = None,
                 scale_init: float = 1.0,
                 shift_init: float = 0.0,
                 eps: float = 1e-06) -> None:
        super().__init__(prefix=prefix)
        self.eps = eps
        self.scale = scale
        if self.scale is None:
            with self.name_scope():
                self.scale = self.params.get('_gamma',
                                             init=mx.init.Constant(value=scale_init),
                                             allow_deferred_init=True)
        self.shift = shift
        if self.shift is None:
            with self.name_scope():
                self.shift = self.params.get('_beta',
                                             init=mx.init.Constant(value=shift_init),
                                             allow_deferred_init=True)

    def hybrid_forward(self, F, data, **params):
        if isinstance(self.scale, mx.sym.Symbol):
            scale = self.scale
        else:
            scale = params['scale']
        if isinstance(self.shift, mx.sym.Symbol):
            shift = self.shift
        else:
            shift = params['shift']
        return F.LayerNorm(data=data, gamma=scale, beta=shift, axis=-1, eps=self.eps, output_mean_var=False)


class LHUC(mx.gluon.HybridBlock):
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
        super().__init__(prefix=prefix)
        self.num_hidden = num_hidden
        self.weight = weight
        if self.weight is None:
            with self.name_scope():
                self.lhuc = self.params.get(C.LHUC_NAME, shape=(num_hidden,), init=mx.init.Uniform(0.1))

    def hybrid_forward(self, F, inputs: mx.sym.Symbol, **params) -> mx.sym.Symbol:
        if isinstance(self.weight, mx.sym.Symbol):
            weight = self.weight
        else:
            weight = params[C.LHUC_NAME]

        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight_vector = 2 * F.Activation(data=weight, act_type="sigmoid")
        out = F.broadcast_mul(weight_vector, inputs)

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
                 prefix: str = C.DEFAULT_OUTPUT_LAYER_PREFIX,
                 name: str = C.LOGITS_NAME) -> None:
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.name = name

        if weight is None:
            self.w = mx.sym.Variable("%sweight" % self.prefix, shape=(vocab_size, hidden_size), dtype='float32')
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
                                         name=self.name)

        # Equivalent NDArray implementation (requires passed weights/biases)
        assert isinstance(hidden, mx.nd.NDArray)
        utils.check_condition(weight is not None and bias is not None,
                              "OutputLayer NDArray implementation requires passing weight and bias NDArrays.")

        return mx.nd.FullyConnected(data=hidden,
                                    num_hidden=bias.shape[0],
                                    weight=weight,
                                    bias=bias,
                                    flatten=False)


class LengthRatioConfig(config.Config):
    """
    Configuration of the length ratio predictor.

    :param layers: Number of layers.
    :param weight: Weight of this loss.
    """

    def __init__(self, num_layers: int, weight: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        # TODO: keeping weight here is redundant because it is also stored
        # in the loss config, but it's used to test if we need length prediction
        self.weight = weight


class LengthRatio:
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    :param prefix: Prefix used for naming.
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 prefix: str = C.LENRATIOS_OUTPUT_LAYER_PREFIX) -> None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        self.prefix = prefix
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.layers = [mx.gluon.nn.Dense(units=hidden_size, activation='tanh', flatten=False, prefix=prefix + 'dense%d_' % l) \
                        for l in range(num_layers - 1)]
        # SoftReLU activation to ensure positiveness of the predicted length ratio
        self.layers.append(mx.gluon.nn.Dense(units=1, activation='softrelu', flatten=False, prefix=prefix + 'dense%d_' % (num_layers - 1)))

    def __call__(self,
                 source_encoded: mx.sym.Symbol,
                 source_encoded_length: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        # data: (n, hidden_size)
        data = LengthRatio.average_sources(source_encoded, source_encoded_length)
        # MLP
        for layer in self.layers:
            data = layer(data)
        # data: (n, 1)
        return data

    @staticmethod
    def average_sources(source_encoded: mx.sym.Symbol, source_encoded_length: mx.sym.Symbol) -> mx.nd.NDArray:
        """
        Calculate the average of encoded sources taking into account their lengths.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Average vectors. Shape(n, hidden_size).
        """
        # source_masked: (n, source_encoded_length, hidden_size)
        source_masked = mx.sym.SequenceMask(data=source_encoded,
                                            axis=1,
                                            sequence_length=source_encoded_length,
                                            use_sequence_length=True,
                                            value=0.)
        # calculate the proper means of encoded sources
        averaged = mx.sym.broadcast_div(mx.sym.sum(source_masked, axis=1, keepdims=False),
                                                   mx.sym.reshape(source_encoded_length, shape=(-1, 1)))
        return averaged


def split_heads(F, x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with head dimension folded into batch and depth divided by the number of heads.

    :param x: Symbol of shape (batch, length, depth).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch * heads, length, depth_per_heads).
    """
    # (batch, length, heads, depth_per_head)
    x = F.reshape(x, shape=(0, -1, heads, depth_per_head))
    # (batch, heads, length, depth/heads)
    x = F.transpose(x, axes=(0, 2, 1, 3))
    # (batch * heads, length, depth/heads)
    return F.reshape(x, shape=(-3, -1, depth_per_head))


def combine_heads(F, x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with both batch & length, and head & depth dimensions combined.

    :param x: Symbol of shape (batch * heads, length, depth_per_head).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch, length, depth).
    """
    # (batch, heads, length, depth_per_head)
    x = F.reshape(x, shape=(-4, -1, heads, 0, depth_per_head))
    # (batch, length, heads, depth_per_head)
    x = F.transpose(x, axes=(0, 2, 1, 3))
    # (batch, length, depth)
    return F.reshape(x, shape=(-1, 0, depth_per_head * heads))


def broadcast_to_heads(F, x: mx.sym.Symbol, num_heads: int, ndim: int, fold_heads: bool = True) -> mx.sym.Symbol:
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
    x = F.expand_dims(x, axis=1)
    # x: (batch, heads, dims...)
    x = F.broadcast_to(x, shape=[0, num_heads] + dims)
    if fold_heads:
        # (batch * heads, dims...)
        return F.reshape(x, shape=[-3] + dims)
    else:
        # x: (batch, heads, dims...)
        return x


class DotAttentionCell(mx.gluon.HybridBlock):

    def __init__(self, dropout: float = 0.0, prefix: str = '') -> None:
        super().__init__(prefix=prefix)
        self.dropout = dropout

    def hybrid_forward(self, F, queries, keys, values, lengths=None, bias=None):
        utils.check_condition(lengths is not None or bias is not None,
                              "Must provide either length or bias argument for masking")
        # (n, lq, lk)
        logits = F.batch_dot(lhs=queries, rhs=keys, transpose_b=True)

        if lengths is not None:
            # mask lk dimension
            # (lk, n, lq)
            logits = F.transpose(logits, axes=(2, 0, 1))
            logits = F.SequenceMask(logits,
                                    use_sequence_length=True,
                                    sequence_length=lengths,
                                    value=C.LARGE_NEGATIVE_VALUE)
            # (n, lq, lk)
            logits = F.transpose(data=logits, axes=(1, 2, 0))

        if bias is not None:
            logits = F.broadcast_add(logits, bias)

        probs = F.softmax(logits, axis=-1)
        probs = F.Dropout(probs, p=self.dropout) if self.dropout > 0.0 else probs

        # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
        return F.batch_dot(lhs=probs, rhs=values)


class MultiHeadAttentionBase(mx.gluon.HybridBlock):
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
        super().__init__(prefix=prefix)
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads

        with self.name_scope():
            self.dot_att = DotAttentionCell(dropout=dropout, prefix='dot_att')
            self.ff_out = mx.gluon.nn.Dense(units=depth_out, flatten=False, use_bias=False, prefix='h2o_')

    def _attend(self,
                F,
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
        queries = split_heads(F, queries, self.depth_per_head, self.heads)
        keys = split_heads(F, keys, self.depth_per_head, self.heads)
        values = split_heads(F, values, self.depth_per_head, self.heads)
        lengths = broadcast_to_heads(F, lengths, self.heads, ndim=1, fold_heads=True) if lengths is not None else lengths

        # (batch*heads, query_max_length, depth_per_head)
        contexts = self.dot_att(queries, keys, values, lengths, bias)

        # (batch, query_max_length, depth)
        contexts = combine_heads(F, contexts, self.depth_per_head, self.heads)

        # contexts: (batch, query_max_length, output_depth)
        contexts = self.ff_out(contexts)

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

        with self.name_scope():
            self.ff_in = mx.gluon.nn.Dense(units=depth_att * 3, flatten=False, use_bias=False, prefix='i2h_')

    # TODO: input types will be problematic when using full Gluon, no Dict allowed. Need to think about cache unpacking.
    def hybrid_forward(self, F,
                       inputs: mx.sym.Symbol,
                       input_lengths: Optional[mx.sym.Symbol] = None,
                       bias: Optional[mx.sym.Symbol] = None,
                       cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:  # mypy: ignore
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
        combined = self.ff_in(inputs)
        # split into query, keys and values
        # (batch, max_length, depth)
        # pylint: disable=unbalanced-tuple-unpacking
        queries, keys, values = F.split(combined, num_outputs=3, axis=2)

        if cache is not None:
            # append new keys & values to cache, update the cache
            keys = cache['k'] = keys if cache['k'] is None else F.concat(cache['k'], keys, dim=1)
            values = cache['v'] = values if cache['v'] is None else F.concat(cache['v'], values, dim=1)

        return self._attend(F, queries, keys, values, lengths=input_lengths, bias=bias)


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

        with self.name_scope():
            self.ff_q = mx.gluon.nn.Dense(units=depth_att, flatten=False, use_bias=False, prefix='q2h_')
            self.ff_k = mx.gluon.nn.Dense(units=depth_att, flatten=False, use_bias=False, prefix='k2h_')
            self.ff_v = mx.gluon.nn.Dense(units=depth_att, flatten=False, use_bias=False, prefix='v2h_')

    def hybrid_forward(self, F,
                       queries: mx.sym.Symbol,
                       memory: mx.sym.Symbol,
                       memory_lengths: Optional[mx.sym.Symbol] = None,
                       bias: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:  # mypy: ignore
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
        queries = self.ff_q(queries)
        # (batch, memory_max_length, depth)
        keys = self.ff_k(memory)
        # (batch, memory_max_length, depth)
        values = self.ff_v(memory)

        return self._attend(F, queries, keys, values, bias=bias, lengths=memory_lengths)


class PlainDotAttention(mx.gluon.HybridBlock):
    """
    Dot attention layer for queries independent from keys/values.
    """
    def __init__(self, prefix=''):
        super().__init__(prefix=prefix)
        with self.name_scope():
            self.dot_att = DotAttentionCell()

    def hybrid_forward(self, F, queries, memory, memory_lengths):
        """
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Symbol of shape (batch, queries_max_length, input_depth).
        :param memory: Symbol of shape (batch, memory_max_length, input_depth).
        :param memory_lengths: Symbol of shape (batch, 1).
        :return: Symbol of shape (batch, queries_max_length, output_depth).
       """

        # (batch*heads, queries_max_length, depth_per_head)
        return self.dot_att(queries, memory, memory, memory_lengths, None)


class ProjectedDotAttention(mx.gluon.HybridBlock):
    """
    Dot attention layer for queries independent from keys/values.

    :param prefix: Attention prefix.
    :param num_hidden: Attention depth / number of hidden units.
    """

    def __init__(self,
                 prefix: str,
                 num_hidden: int) -> None:
        super().__init__(prefix=prefix)
        self.num_hidden = num_hidden
        with self.name_scope():
            self.q2h = mx.gluon.nn.Dense(units=num_hidden, flatten=False, use_bias=True)
            self.kv2h = mx.gluon.nn.Dense(units=num_hidden * 2, flatten=False, use_bias=True)
            self.dot_att = DotAttentionCell()

    def hybrid_forward(self, F,
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
        combined = self.kv2h(memory)

        # split into keys and values
        # pylint: disable=unbalanced-tuple-unpacking
        keys, values = F.split(data=combined, num_outputs=2, axis=2)

        # (batch, queries_max_length, num_hidden)
        queries = self.q2h(queries)

        # scale by sqrt(num_hidden)
        queries = queries * (self.num_hidden ** -0.5)

        # (batch, queries_max_length, num_hidden)
        contexts = self.dot_att(queries, keys, values, memory_lengths, None)

        return contexts

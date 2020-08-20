# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from abc import abstractmethod
from typing import Optional, Union, Tuple, List
from functools import lru_cache

import mxnet as mx
import numpy as np

from . import config
from . import constants as C
from . import quantization
from . import utils

logger = logging.getLogger(__name__)


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
    if act_type == C.GELU:
        return mx.gluon.nn.GELU()
    return mx.gluon.nn.Activation(activation=act_type)


class LHUC(mx.gluon.HybridBlock):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    :param prefix: Optional prefix for created parameters (if not given as weight).
    """
    def __init__(self,
                 num_hidden: int,
                 prefix: str = C.LHUC_PREFIX,
                 weight_init: Union[str, mx.init.Initializer] = mx.init.Uniform(0.1)) -> None:
        super().__init__(prefix=prefix)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_hidden,), init=weight_init)

    def hybrid_forward(self, F, data, weight) -> mx.sym.Symbol:
        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight = 2 * F.Activation(weight, act_type="sigmoid")
        return F.broadcast_mul(weight, data)


class WeightNormalization(mx.gluon.HybridBlock):
    """
    Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
    For a given tensor the normalization is done per hidden dimension.

    :param num_hidden: Size of the first dimension.
    :param ndim: The total number of dimensions of the weight tensor.
    :param prefix: The prefix used for naming.
    """

    def __init__(self,
                 num_hidden: int,
                 ndim: int = 2,
                 prefix: str = 'wn_') -> None:
        super().__init__(prefix=prefix)
        with self.name_scope():
            self.scale = self.params.get("scale",
                                         shape=tuple([num_hidden] + [1] * (ndim - 1)),
                                         init=mx.init.Constant(value=1.0))

    def hybrid_forward(self, F, weight, scale):
        return F.broadcast_mul(lhs=F.L2Normalization(weight, mode='instance'), rhs=scale)


class OutputLayer(mx.gluon.HybridBlock):
    """
    Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.

    :param hidden_size: Input hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight: Optional shared weight Parameter.
    :param weight_initializer: Initializer for weight.
    :param bias_initializer: Initializer for bias.
    :param dtype: Data type.
    :param prefix: Prefix used for naming.
    :params params: Optional parameter dict for shared parameters.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 weight: Optional[mx.gluon.Parameter] = None,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: str = 'zeros',
                 dtype: str = C.DTYPE_FP32,
                 prefix: str = C.DEFAULT_OUTPUT_LAYER_PREFIX) -> None:
        super().__init__(prefix=prefix)
        self.vocab_size = vocab_size

        with self.name_scope():
            if dtype == C.DTYPE_INT8:
                self.scaling = self.params.get('scaling', shape=(1,), init=mx.initializer.Constant(-1.0), dtype=C.DTYPE_FP32, allow_deferred_init=False)
                # This is only for inference but MXNet tries to create an
                # initializer anyway, then fails because most random
                # generators don't support int8 output.
                weight_initializer = 'zeros'
            if weight is None:
                self.weight = self.params.get("weight",
                                              shape=(vocab_size, hidden_size),
                                              init=weight_initializer,
                                              dtype=dtype,
                                              allow_deferred_init=False)
            else:
                self.weight = weight  # adds to self._reg_params
                self.params.update({weight.name: weight})  # adds to self.params

            self.bias = self.params.get("bias",
                                        shape=(vocab_size,),
                                        init=bias_initializer,
                                        dtype=dtype if dtype != C.DTYPE_INT8 else C.DTYPE_FP32, # Bias stays fp32 even with int8 weights.
                                        allow_deferred_init=False)

    @lru_cache(maxsize=1)
    def _take_slice(self, vocab_slice_ids: mx.nd.NDArray) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        if self.weight.dtype == C.DTYPE_INT8:
            weight = mx.nd.contrib.intgemm_take_weight(self.weight.data(), vocab_slice_ids)
        else:
            weight = self.weight.data().take(vocab_slice_ids)
        bias = self.bias.data().take(vocab_slice_ids)
        return weight, bias

    def forward(self, data, vocab_slice_ids):
        if vocab_slice_ids is not None:
            # imperative, reduced matrix multiplication for vocabulary selection
            weight, bias = self._take_slice(vocab_slice_ids)
            if self.weight.dtype == C.DTYPE_INT8:
                return mx.nd.contrib.intgemm_fully_connected(data, weight, self.scaling.data(), bias,
                                                             num_hidden=vocab_slice_ids.shape[0],
                                                             flatten=False,
                                                             name=C.LOGITS_NAME)
            else:
                return mx.nd.FullyConnected(data=data,
                                            num_hidden=vocab_slice_ids.shape[0],
                                            weight=weight,
                                            bias=bias,
                                            flatten=False,
                                            name=C.LOGITS_NAME)
        return super().forward(data)

    def hybrid_forward(self, F, data, weight, bias, scaling = None):
        if self.weight.dtype == C.DTYPE_INT8:
            return F.contrib.intgemm_fully_connected(data=data,
                                    num_hidden=self.vocab_size,
                                    weight=weight,
                                    scaling=scaling,
                                    bias=bias,
                                    flatten=False,
                                    name=C.LOGITS_NAME)
        else:
            return F.FullyConnected(data=data,
                                    num_hidden=self.vocab_size,
                                    weight=weight,
                                    bias=bias,
                                    flatten=False,
                                    name=C.LOGITS_NAME)


class LengthRatioConfig(config.Config):
    """
    Configuration of the length ratio predictor.

    :param num_layers: Number of layers.
    :param weight: Weight of this loss.
    """

    def __init__(self, num_layers: int, weight: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.weight = weight


class LengthRatio(mx.gluon.HybridBlock):
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    :param prefix: Prefix used for naming.
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 prefix: str = C.LENRATIOS_OUTPUT_LAYER_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__(prefix=prefix)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        with self.name_scope():
            self.layers = mx.gluon.nn.HybridSequential()
            for l in range(num_layers - 1):
                self.layers.add(quantization.QuantizableDense(units=hidden_size, activation='tanh',
                                                  flatten=False, prefix='dense%d_' % l, dtype=dtype))
            # SoftReLU activation to ensure positiveness of the predicted length ratio
            self.layers.add(quantization.QuantizableDense(units=1, activation='softrelu',
                                              flatten=False, prefix='dense%d_' % (num_layers - 1), dtype=dtype))

    def hybrid_forward(self, F, source_encoded, source_encoded_length):
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        # source_masked: (n, source_encoded_length, hidden_size)
        source_masked = F.SequenceMask(data=source_encoded,
                                       axis=1,
                                       sequence_length=source_encoded_length,
                                       use_sequence_length=True,
                                       value=0.)
        # calculate the proper means of encoded sources
        # data: (n, hidden_size)
        data = F.broadcast_div(F.sum(source_masked, axis=1, keepdims=False),
                               F.reshape(source_encoded_length, shape=(-1, 1)))
        # MLP. Shape: (n, 1)
        data = self.layers(data)
        # Shape: (n,)
        return F.squeeze(data)


def split_heads(F, x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with heads as second dimension and channel depth / number of heads as last dimension.

    :param x: Symbol of shape (batch, length, depth).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch, heads, length, depth_per_heads).
    """
    # (batch, length, heads, depth_per_head)
    x = F.reshape(x, shape=(0, -1, heads, depth_per_head))
    # (batch, heads, length, depth/heads)
    return F.transpose(x, axes=(0, 2, 1, 3))


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
    #x = F.broadcast_to(x, shape=[0, num_heads] + dims)
    x = F.repeat(x, repeats=num_heads, axis=1)
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
        self._dtype = C.DTYPE_FP32

    def cast(self, dtype):
        self._dtype = dtype
        super().cast(dtype)

    def hybrid_forward(self, F, queries, keys, values, lengths=None, bias=None):
        # (n, lq, lk)
        logits = F.batch_dot(lhs=queries, rhs=keys, transpose_b=True)

        # TODO(fhieber): consider softmax with length argument once available.
        # TODO(fhieber: Also see https://github.com/dmlc/gluon-nlp/pull/910
        if lengths is not None:
            # mask lk dimension
            # (lk, n, lq)
            logits = F.transpose(logits, axes=(2, 0, 1))
            logits = F.SequenceMask(logits,
                                    use_sequence_length=True,
                                    sequence_length=lengths,
                                    value=-C.LARGE_VALUES[self._dtype])
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
    :param dtype: Data type for weights
    """
    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(prefix=prefix)
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads

        with self.name_scope():
            self.dot_att = DotAttentionCell(dropout=dropout, prefix='dot_att')
            self.ff_out = quantization.QuantizableDense(in_units=depth_att, units=depth_out, flatten=False, use_bias=False, prefix='h2o_', dtype = dtype)

    def _attend(self,
                F,
                queries: mx.sym.Symbol,
                keys: mx.sym.Symbol,
                values: mx.sym.Symbol,
                lengths: Optional[mx.sym.Symbol] = None,
                bias: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (batch_size, heads, query_max_length, depth_per_head).
        :param keys: Keys. Shape: (batch_size, heads, memory_max_length, depth_per_head).
        :param values: Values. Shape: (batch_size, heads, memory_max_length, depth_per_head).
        :param lengths: Optional lengths of keys. Shape: (batch_size,).
        :param bias: Optional 3d bias.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """
        # fold head dimension into batch dimension
        # (batch*heads, length, depth/heads)
        queries = F.reshape(queries, shape=(-3, -1, self.depth_per_head))
        keys = F.reshape(keys, shape=(-3, -1, self.depth_per_head))
        values = F.reshape(values, shape=(-3, -1, self.depth_per_head))
        lengths = broadcast_to_heads(F, lengths, self.heads, ndim=1,
                                     fold_heads=True) if lengths is not None else lengths

        # (batch*heads, query_max_length, depth_per_head)
        contexts = self.dot_att(queries, keys, values, lengths, bias)

        # (batch, query_max_length, depth)
        contexts = combine_heads(F, contexts, self.depth_per_head, self.heads)

        # contexts: (batch, query_max_length, output_depth)
        contexts = self.ff_out(contexts)

        return contexts


class AutoregressiveLayer(mx.gluon.HybridBlock):
    @property
    @abstractmethod
    def prefix(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        raise NotImplementedError

    @abstractmethod
    def hybrid_forward(self, F, inputs: mx.sym.Symbol, previous_states: mx.sym.Symbol, *args) -> Tuple:
        """
        :param F: ndarray or Symbol
        :param inputs: layer input
        :param previous_states: Symbol or list of Symbols
        :param args: layer-specific arguments and/or arguments to be ignored
        :return: layer output and new states
        """
        raise NotImplementedError


class MultiHeadSelfAttention(MultiHeadAttentionBase, AutoregressiveLayer):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """
    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout, dtype)

        self.depth_att = depth_att
        with self.name_scope():
            self.ff_in = quantization.QuantizableDense(in_units=depth_att, units=depth_att * 3, flatten=False, use_bias=False, prefix='i2h_', dtype=dtype)

    @property
    def prefix(self) -> str:
        return "att_self_"

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 2

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return True

    def get_state_shape(self, batch_size: int) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        # shape: (batch, heads, length, depth_per_head)
        return batch_size, self.heads, 1, self.depth_out // self.heads

    def hybrid_forward(self, F,
                       inputs: mx.sym.Symbol,
                       previous_states: List[mx.sym.Symbol],
                       input_lengths: Optional[mx.sym.Symbol] = None,
                       bias: Optional[mx.sym.Symbol] = None,
                       *args):  # mypy: ignore
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a symbol of shape (batch, max_length, output_depth).

        :param inputs: Input Data. Shape: (batch, max_length, input_depth).
        :param input_lengths: Optional lengths of inputs to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param previous_states: Optional list with two Symbols - previous input's keys and values.
                                Shape: 2 * (batch, max_length+1, depth_att).
        :return: Symbol of shape (batch, max_length, output_depth).
        """
        # combined: (batch, max_length, depth * 3)
        combined = self.ff_in(inputs)
        # split into query, keys and values
        # (batch, max_length, depth)
        # pylint: disable=unbalanced-tuple-unpacking
        queries, keys, values = F.split(combined, num_outputs=3, axis=2)

        # scale by sqrt(depth_per_head)
        queries = queries * (self.depth_per_head ** -0.5)
        # (batch, heads, length, depth/heads)
        queries = split_heads(F, queries, self.depth_per_head, self.heads)
        keys = split_heads(F, keys, self.depth_per_head, self.heads)
        values = split_heads(F, values, self.depth_per_head, self.heads)

        updated_keys = keys

        previous_keys, previous_values = previous_states
        if previous_keys is not None:
            updated_keys = F.concat(previous_keys, keys, dim=2)
            keys = _remove_first_step(F, updated_keys)

        updated_values = values
        if previous_values is not None:
            updated_values = F.concat(previous_values, values, dim=2)
            values = _remove_first_step(F, updated_values)

        return self._attend(F, queries, keys, values, lengths=input_lengths, bias=bias), updated_keys, updated_values


def _remove_first_step(F, data):
    """
    :param F: MXNet namespace.
    :param data: Input data. Shape: (batch, heads, length, num_hidden).
    :return: Output data. Shape: (batch, heads, length[1:], num_hidden
    """
    return F.slice(data, begin=(None, None, 1, None), end=(None, None, None, None))


class MultiHeadAttention(MultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param depth_key_value: Dimension of input key and value vectors.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32,
                 depth_key_value: int = 0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout, dtype)

        with self.name_scope():
            self.ff_q = quantization.QuantizableDense(in_units=depth_out, units=depth_att, flatten=False, use_bias=False, prefix='q2h_', dtype=dtype)
            self.ff_k = quantization.QuantizableDense(in_units=depth_key_value, units=depth_att, flatten=False, use_bias=False, prefix='k2h_', dtype=dtype)
            self.ff_v = quantization.QuantizableDense(in_units=depth_key_value, units=depth_att, flatten=False, use_bias=False, prefix='v2h_', dtype=dtype)

    def project_and_isolate_heads(self, F, memory: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Projects memory into keys and values, and separates attention heads dimension.

        :param memory: Memory tensor. Shape: (batch, memory_max_length, input_depth).
        :return: Symbol of shape (batch, heads, memory_max_length, depth_per_head).
        """
        keys = self.ff_k(memory)
        values = self.ff_v(memory)
        keys = split_heads(F, keys, depth_per_head=self.depth_per_head, heads=self.heads)
        values = split_heads(F, values, depth_per_head=self.depth_per_head, heads=self.heads)
        return keys, values

    def hybrid_forward(self, F,
                       queries: mx.sym.Symbol,
                       memory: mx.sym.Symbol,
                       memory_lengths: Optional[mx.sym.Symbol] = None,
                       bias: Optional[mx.sym.Symbol] = None,
                       projected_memory_keys: Optional[mx.sym.Symbol] = None,
                       projected_memory_values: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:  # mypy: ignore
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Query tensor. Shape: (batch, query_max_length, input_depth).
        :param memory: Memory data to attend to. Shape: (batch, memory_max_length, input_depth).
        :param memory_lengths: Optional lengths of memory to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param projected_memory_keys: Optional previously projected memory keys.
        :param projected_memory_values: Optional previously projected memory values.
        :return: Symbol of shape (batch, query_seq_len, output_depth).
        """
        # (batch, query_max_length, depth)
        queries = self.ff_q(queries)
        # scale by sqrt(depth_per_head)
        queries = queries * (self.depth_per_head ** -0.5)
        # (batch, heads, length, depth/heads)
        queries = split_heads(F, queries, self.depth_per_head, self.heads)

        if projected_memory_keys is not None and projected_memory_values is not None:
            keys, values = projected_memory_keys, projected_memory_values
        else:
            keys, values = self.project_and_isolate_heads(F, memory)

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
                 num_hidden: int,
                 dtype: str) -> None:
        super().__init__(prefix=prefix)
        self.num_hidden = num_hidden
        with self.name_scope():
            self.q2h = quantization.QuantizableDense(units=num_hidden, flatten=False, use_bias=True, dtype=dtype)
            self.kv2h = quantization.QuantizableDense(units=num_hidden * 2, flatten=False, use_bias=True, dtype=dtype)
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


def get_positional_embeddings(length, depth) -> np.ndarray:
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
    # interleave: (length, num_embed)
    encodings = np.hstack([sin, cos])
    return encodings


class PositionalEmbeddings(mx.gluon.HybridBlock):
    """
    Takes an encoded sequence and adds sinusoidal or learned positional embeddings as in Vaswani et al, 2017 to it.

    :param weight_type: type of embeddings, fixed or learned.
    :param num_embed: Embedding size.
    :param max_seq_len: Maximum sequence length.
    :param prefix: Name prefix for symbols of this encoder.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    :param weight_init: Optional initializer for learned embeddings.
    """

    def __init__(self,
                 weight_type: str,
                 num_embed: int,
                 max_seq_len: int,
                 prefix: str,
                 scale_up_input: bool,
                 scale_down_positions: bool,
                 weight_init: Optional[Union[str, mx.init.Initializer]] = None) -> None:
        utils.check_condition(num_embed % 2 == 0, "Positional embeddings require an even embedding size it "
                                                  "is however %d." % num_embed)
        super().__init__(prefix=prefix)
        self.weight_type = weight_type
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions

        with self.name_scope():
            if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
                pos_weight = get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
                if self.scale_down_positions:
                    pos_weight *= self.num_embed ** -0.5
                self.weight = self.params.get_constant('weight', pos_weight)
            elif self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
                self.weight = self.params.get('weight', shape=(self.max_seq_len, self.num_embed), init=weight_init)
            else:
                raise ValueError("weight_type '%s' is not supported!" % self.weight_type)

    def hybrid_forward(self, F, data, steps, weight):  # pylint: disable=arguments-differ
        """
        Applies positional embeddings to input data.

        :param data: Input data. Shape: (batch, length or 1, num_embed)
        :param steps: Optional steps input. If given, shape is (batch_size or 1, seq_len,)
        :param weight: Positional embedding constant.
        :return: Data with positional embeddings added
        """
        # (length, num_embed)
        if steps is None:
            # (batch, length, num_embed)
            pos_embedding = F.slice_like(F.expand_dims(weight, axis=0), data, axes=(1,))
        else:
            # (batch_size or 1, seq_len, num_embed)
            pos_embedding = F.Embedding(steps, weight, self.max_seq_len, self.num_embed)

        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            pos_embedding = F.BlockGrad(pos_embedding)

        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        return F.broadcast_add(data, pos_embedding)


class SSRU(AutoregressiveLayer):
    """
    Simpler Simple Recurrent Unit

    Kim et al, "From Research to Production and Back: Ludicrously Fast Neural Machine Translation" WNGT 2019

    Variant of an LSTM cell aimed at reducing computational dependency across time steps.
    Formally described as:

    (1) f[t] = sigmoid(W1[t] * x[t] + b[t])
    (2) c[t] = f[t] . c[t-1] + (1 - f[t]) . W2[t] * x[t]
    (3) h = ReLU(c[t])

    where:
        . represents elementwise multiplication;
        x[t] is the input at time step t;
        f[t] is the output of the forget gate at time step t;
        c[t] is the cell state at time step t;
        h is the output of the unit.

    :param model_size: number of hidden units
    :param inference_only: flag used to indicate execution at inference time
    :param prefix: prefix prepended to the names of internal Symbol instances
    :param dtype: data type
    """
    def __init__(self,
                 model_size: int,
                 inference_only: bool,
                 prefix: str = C.SSRU_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super(SSRU, self).__init__(prefix=prefix)

        self.model_size = model_size
        self.inference_only = inference_only

        self.cell_state_transform = self._inference_cell_state_transform \
                                    if inference_only else self._training_cell_state_transform

        with self.name_scope():
            self.forget_gate = quantization.QuantizableDense(in_units=model_size,
                                                             units=model_size,
                                                             activation="sigmoid",
                                                             flatten=False,
                                                             prefix="forget_gate_",
                                                             dtype=dtype)

            self.linear = quantization.QuantizableDense(in_units=model_size,
                                                        units=model_size,
                                                        use_bias=False,
                                                        flatten=False,
                                                        prefix="linear_",
                                                        dtype=dtype)

    @property
    def prefix(self) -> str:
        return C.SSRU_PREFIX

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return False

    def get_state_shape(self, batch_size: int) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        if self.inference_only:
            return batch_size, 1, self.model_size
        else:
            return batch_size, self.model_size

    @staticmethod
    def _training_cell_state_transform(F, previous_cell_state, weighted_inputs, forget_rates) -> Tuple:
        """Update SSRU cell at training time"""
        def _time_step_update(step_input_and_forget_rate, previous_step_state) -> Tuple:
            """
            Recurrently update the SSRU cell state for one time step.

            :param step_input_and_forget_rate: List = [step_input, forget_rate]
            :param previous_step_state: cell state at (t-1)
            :return: twice the current time step state. NOTE: The first instance will be stacked in the final
            foreach output and the second will be the input to the next time_step_update iteration.
            """
            step_input, forget_rate = step_input_and_forget_rate  # each of shape (batch_size, model_size)
            current_step_state = forget_rate * previous_step_state + step_input
            return current_step_state, current_step_state

        weighted_inputs = F.transpose(weighted_inputs, axes=(1, 0, 2))  # (max_length, batch, input_depth)
        forget_rates = F.transpose(forget_rates, axes=(1, 0, 2))  # (max_length, batch, input_depth)

        # (max_length, batch, input_depth), (batch, input_depth)
        cell_state, last_step_state = F.contrib.foreach(_time_step_update,
                                                        [weighted_inputs, forget_rates],
                                                        previous_cell_state)

        return F.transpose(cell_state, axes=(1, 0, 2)), last_step_state

    @staticmethod
    def _inference_cell_state_transform(F, previous_cell_state, weighted_inputs, forget_rates) -> Tuple:
        """Update SSRU cell at inference time"""
        new_step_state = forget_rates * previous_cell_state + weighted_inputs  # (batch, 1, input_depth)
        return new_step_state, new_step_state

    def hybrid_forward(self, F, inputs: mx.sym.Symbol, previous_states: mx.sym.Symbol, *args) -> Tuple:
        """
        :param F: ndarray or Symbol
        :param inputs: input data. Shape: (batch, max_length, input_depth).
        :param previous_states: previous cell states. Shape: (batch, max_length, input_depth)
        :return: cell output and new cell states.  Both with shape (batch, max_length, input_depth).
        """
        forget_rates = self.forget_gate(inputs)
        weighted_inputs = (1 - forget_rates) * self.linear(inputs)

        cell_state, last_step_state = self.cell_state_transform(F, previous_states, weighted_inputs, forget_rates)

        return F.relu(cell_state), last_step_state


# Copyright 2017--2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import mxnet as mx
from mxnet import gluon, np, npx

from . import config
from . import constants as C
from . import quantization
from . import utils

logger = logging.getLogger(__name__)


def get_activation(act_type: str) -> gluon.Block:
    """
    Returns Gluon Block for given activation type.

    Custom activation types include:
     - Swish-1, also called Sigmoid-Weighted Linear Unit (SiLU): Ramachandran et
       al. (https://arxiv.org/pdf/1710.05941.pdf), Elfwing et al.
       (https://arxiv.org/pdf/1702.03118.pdf)
     - Gaussian Error Linear Unit (GELU): Hendrycks and Gimpel
       (https://arxiv.org/pdf/1606.08415.pdf)

    :param act_type: Type of activation.
    :return: output ndarray with same shape as input.
    """
    if act_type == C.SWISH1:
        return gluon.nn.Swish()
    if act_type == C.GELU:
        return gluon.nn.GELU()
    return gluon.nn.Activation(activation=act_type)


class LHUC(gluon.HybridBlock):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    """
    def __init__(self,
                 num_hidden: int,
                 weight_init: Union[str, mx.init.Initializer] = mx.init.Uniform(0.1)) -> None:
        super().__init__()
        self.weight = gluon.Parameter('weight', shape=(num_hidden,), init=weight_init)

    def forward(self, data: np.ndarray) -> np.ndarray:
        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight = 2 * npx.activation(self.weight.data(), act_type="sigmoid")
        return weight * data


class WeightNormalization(mx.gluon.HybridBlock):
    """
    Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
    For a given tensor the normalization is done per hidden dimension.

    :param num_hidden: Size of the first dimension.
    :param ndim: The total number of dimensions of the weight tensor.
    """

    def __init__(self, num_hidden: int, ndim: int = 2) -> None:
        super().__init__()
        self.scale = gluon.Constant(np.ones(shape=tuple([num_hidden] + [1] * (ndim - 1))))
        self._axis_arg = tuple(range(1, ndim))

    def forward(self, weight: np.ndarray) -> np.ndarray:
        return weight / np.linalg.norm(weight, keepdims=True, axis=self._axis_arg) * self.scale.data()


class OutputLayer(gluon.HybridBlock):
    """
    Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.

    :param hidden_size: Input hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight: Optional shared weight Parameter.
    :param weight_initializer: Initializer for weight.
    :param bias_initializer: Initializer for bias.
    :param dtype: Data type.
    :params params: Optional parameter dict for shared parameters.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 weight: Optional[mx.gluon.Parameter] = None,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: str = 'zeros',
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        if dtype == C.DTYPE_INT8:
            self.scaling = gluon.Parameter('scaling',
                                           shape=(1,), init=mx.initializer.Constant(-1.0),
                                           dtype=C.DTYPE_FP32, allow_deferred_init=False)
            # This is only for inference but MXNet tries to create an
            # initializer anyway, then fails because most random
            # generators don't support int8 output.
            weight_initializer = 'zeros'
        if weight is None:
            self.weight = gluon.Parameter("weight",
                                          shape=(vocab_size, hidden_size),
                                          init=weight_initializer,
                                          dtype=dtype,
                                          allow_deferred_init=False)
        else:
            self.weight = weight

        self.bias = gluon.Parameter("bias",
                                    shape=(vocab_size,),
                                    init=bias_initializer,
                                    # Bias stays fp32 even with int8 weights.
                                    dtype=dtype if dtype != C.DTYPE_INT8 else C.DTYPE_FP32,
                                    allow_deferred_init=False)

        self._cache_key = None  # type: Optional[int]
        self._weight_slice_cache = None  # type: Optional[np.ndarray]
        self._bias_slice_cache = None  # type: Optional[np.ndarray]

    def _is_new_vocab_slices(self, x: np.ndarray) -> bool:
        # MXNet ndarrays (like Numpy ndarrays) do not support hashing, using string representation.
        x_hash = hash(str(x))
        if x_hash != self._cache_key:
            self._cache_key = x_hash
            return True
        return False

    def _take_slice(self, vocab_slice_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.weight.dtype == C.DTYPE_INT8:
            weight = npx.intgemm_take_weight(self.weight.data(), vocab_slice_ids)
        else:
            weight = self.weight.data().take(vocab_slice_ids, axis=0)
        bias = self.bias.data().take(vocab_slice_ids, axis=0)
        return weight, bias

    def __call__(self, data: np.ndarray, vocab_slice_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if vocab_slice_ids is not None:
            # imperative, reduced matrix multiplication for vocabulary selection
            weight, bias = self._take_slice(vocab_slice_ids)
            if self._is_new_vocab_slices(vocab_slice_ids):
                weight, bias = self._take_slice(vocab_slice_ids)
                self._weight_slice_cache, self._bias_slice_cache = weight, bias
            else:
                weight, bias = self._weight_slice_cache, self._bias_slice_cache
            if self.weight.dtype == C.DTYPE_INT8:
                return npx.intgemm_fully_connected(data, weight, self.scaling.data(), bias,
                                                   num_hidden=vocab_slice_ids.shape[0],
                                                   flatten=False,
                                                   name=C.LOGITS_NAME)
            else:
                return npx.fully_connected(data,
                                           num_hidden=vocab_slice_ids.shape[0],
                                           weight=weight,
                                           bias=bias,
                                           no_bias=False,
                                           flatten=False,
                                           name=C.LOGITS_NAME)
        return super().__call__(data)

    def forward(self, data: np.ndarray) -> np.ndarray:
        if self.weight.dtype == C.DTYPE_INT8:
            return npx.intgemm_fully_connected(data,
                                               num_hidden=self.vocab_size,
                                               weight=self.weight.data(),
                                               scaling=self.scaling.data(),
                                               bias=self.bias.data(),
                                               flatten=False,
                                               name=C.LOGITS_NAME)
        else:
            return npx.fully_connected(data,
                                       num_hidden=self.vocab_size,
                                       weight=self.weight.data(),
                                       bias=self.bias.data(),
                                       no_bias=False,
                                       flatten=False,
                                       name=C.LOGITS_NAME)


@dataclass
class LengthRatioConfig(config.Config):
    num_layers: int  # Number of layers
    weight: float  # Weight of this loss


class LengthRatio(mx.gluon.HybridBlock):
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 dtype: str = C.DTYPE_FP32) -> None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.layers = gluon.nn.HybridSequential()
        for l in range(num_layers - 1):
            self.layers.add(quantization.QuantizableDense(units=hidden_size, activation='tanh',
                                                          flatten=False, dtype=dtype))
        # SoftReLU activation to ensure positiveness of the predicted length ratio
        self.layers.add(quantization.QuantizableDense(units=1, activation='softrelu', flatten=False, dtype=dtype))

    def forward(self, source_encoded: np.ndarray, source_encoded_length: np.ndarray) -> np.ndarray:
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        # source_masked: (n, source_encoded_length, hidden_size)
        source_masked = npx.sequence_mask(source_encoded,
                                          axis=1,
                                          sequence_length=source_encoded_length,
                                          use_sequence_length=True,
                                          value=0.)
        # calculate the proper means of encoded sources
        # data: (n, hidden_size)
        data = np.sum(source_masked, axis=1, keepdims=False) / np.reshape(source_encoded_length, (-1, 1))
        # MLP. Shape: (n, 1)
        data = self.layers(data)
        # Shape: (n,)
        return np.squeeze(data)


class DotAttentionCell(gluon.HybridBlock):

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout
        self._dtype = C.DTYPE_FP32

    def cast(self, dtype):
        self._dtype = dtype
        super().cast(dtype)

    def forward(self, queries: np.ndarray, key_values: np.ndarray, heads: np.ndarray,
                lengths: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):

        # (n*h, lq, lk)
        logits = npx.interleaved_matmul_encdec_qk(queries, key_values, heads=heads)

        if bias is not None:
            logits = logits + bias

        if lengths is not None:
            # required shape for lengths: (n*h, lq); required dtype: int32
            probs = npx.softmax(logits, axis=-1, length=lengths, use_length=True)
        else:
            probs = npx.softmax(logits, axis=-1)

        probs = npx.dropout(probs, p=self.dropout) if self.dropout > 0.0 else probs
        
        # key_values: (lk, n, dv * 2)
        # probs: (n*h, lq, lk)
        # result: (n, lq, dv)
        return npx.interleaved_matmul_encdec_valatt(key_values, probs, heads=heads)


def prepare_source_valid_lengths(valid_length: np.ndarray, query_data: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns an int32 valid length tensor of shape (batch * num_heads, query_length) to be used in
    the softmax operation in DotAttentionCell with the length argument.
    Due to broadcast_like, dtypes of valid_length and query_data must be the same.

    :param valid_length: Valid length information. Shape: (batch,).
    :param query_data: Tensor from which the query_length dimension is derived.
                       Expected shape: (X, query_length, ...).
    :param num_heads: Number of attention heads.
    :return: int32 tensor of shape (batch * num_heads, query_length).
    """
    # (batch * heads,)
    att_valid_length = np.repeat(valid_length, repeats=num_heads, axis=0)
    att_valid_length = npx.broadcast_like(np.expand_dims(att_valid_length, axis=1),
                                          query_data,
                                          lhs_axes=(1,), rhs_axes=(1,))
    return att_valid_length.astype(dtype='int32', copy=False)


class MultiHeadAttentionBase(gluon.HybridBlock):
    """
    Base class for Multi-head attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """
    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads

        self.dot_att = DotAttentionCell(dropout=dropout)
        self.ff_out = quantization.QuantizableDense(in_units=depth_att, units=depth_out,
                                                    flatten=False, use_bias=False, dtype=dtype)

    def _attend(self,
                queries: np.ndarray,
                key_values: np.ndarray,
                lengths: Optional[np.ndarray] = None,
                bias: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (query_max_length, batch_size, depth).
        :param key_values: Keys. Shape: (memory_max_length, batch_size, depth * 2).
        :param lengths: Optional lengths of keys. Shape: (batch_size*heads,).
        :param bias: Optional 3d bias.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """

        # (query_max_length, batch, depth)
        contexts = self.dot_att(queries, key_values, self.heads, lengths, bias)

        # (query_max_length, batch, output_depth)
        contexts = self.ff_out(contexts)

        return contexts


class AutoregressiveLayer(mx.gluon.HybridBlock):
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
    def forward(self, inputs: np.ndarray, previous_states: np.ndarray, *args) -> Tuple:
        """
        :param inputs: layer input
        :param previous_states: Previous states array or list of arrays
        :param args: layer-specific arguments and/or arguments to be ignored
        :return: layer output and new states
        """
        raise NotImplementedError


class MultiHeadSelfAttention(MultiHeadAttentionBase, AutoregressiveLayer):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """
    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype)

        self.depth_att = depth_att
        self.ff_in = quantization.QuantizableDense(in_units=depth_att, units=depth_att * 3,
                                                   flatten=False, use_bias=False, dtype=dtype)

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return True

    def get_state_shape(self, batch_size: int) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        # shape: (length, batch, key_depth + value_depth)
        return 0, batch_size, self.depth_out * 2

    def forward(self,
                inputs: np.ndarray,
                previous_states: Optional[np.ndarray] = None,
                input_lengths: Optional[np.ndarray] = None,
                bias: Optional[np.ndarray] = None,
                *args) -> Tuple[np.ndarray, np.ndarray]:  # mypy: ignore
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a ndarray of shape (batch, max_length, output_depth).

        :param inputs: Input Data. Shape: (max_length, batch, input_depth).
        :param input_lengths: Optional lengths of inputs to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param previous_states: Optional list with two ndarrays - previous input's keys and values.
                                Shape: 2 * (batch, max_length+1, depth_att).
        :return: ndarray of shape (batch, max_length, output_depth).
        """

        proj = self.ff_in(inputs)
        queries, kv_1, kv_2 = np.split(proj, 3, axis=2)
        states = np.concatenate((kv_1, kv_2), axis=2)

        if previous_states is not None:
            states = np.concatenate((previous_states, states), axis=0)

        return self._attend(queries, states, lengths=input_lengths, bias=bias), states


class MultiHeadAttention(MultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param depth_key_value: Dimension of input key and value vectors.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """

    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32,
                 depth_key_value: int = 0) -> None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype)

        self.ff_q = quantization.QuantizableDense(in_units=depth_out, units=depth_att,
                                                  flatten=False, use_bias=False, dtype=dtype)
        self.ff_kv = quantization.QuantizableDense(in_units=depth_key_value, units=2 * depth_att,
                                                   flatten=False, use_bias=False, dtype=dtype)

    def forward(self, queries: np.ndarray,
                memory: np.ndarray,
                memory_lengths: Optional[np.ndarray] = None,
                bias: Optional[np.ndarray] = None,
                projected_memory_kv: Optional[np.ndarray] = None) -> np.ndarray:  # mypy: ignore
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns an ndarray of shape (max_length, batch, output_depth).

        :param queries: Query tensor. Shape: (query_max_length, batch, input_depth).
        :param memory: Memory data to attend to. Shape: (memory_max_length, batch, input_depth).
        :param memory_lengths: Optional lengths of memory to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param projected_memory_kv: Optional previously projected memory keys and values.
        :return: ndarray of shape (query_seq_len, batch, output_depth).
        """

        queries = self.ff_q(queries)
        kv = projected_memory_kv if projected_memory_kv is not None else self.ff_kv(memory)
        return self._attend(queries, kv, bias=bias, lengths=memory_lengths)


# unused, not ported
class PlainDotAttention(gluon.HybridBlock):
    """
    Dot attention layer for queries independent from keys/values.
    """
    def __init__(self):
        super().__init__()
        self.dot_att = DotAttentionCell()

    def forward(self, queries: np.ndarray, memory: np.ndarray, memory_lengths: np.ndarray) -> np.ndarray:
        """
        Returns an ndarray of shape (batch, max_length, output_depth).

        :param queries: ndarray of shape (queries_max_length, batch, input_depth).
        :param memory: ndarray of shape (memory_max_length, batch, input_depth).
        :param memory_lengths: ndarray of shape (batch, 1).
        :return: ndarray of shape (queries_max_length, batch, output_depth).
       """

        # (queries_max_length, batch, output_depth)
        return self.dot_att(queries, memory, 1, memory_lengths, None)


# unused, not ported
class ProjectedDotAttention(gluon.HybridBlock):
    """
    Dot attention layer for queries independent from keys/values.

    :param num_hidden: Attention depth / number of hidden units.
    """

    def __init__(self,
                 num_hidden: int,
                 dtype: str) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.q2h = quantization.QuantizableDense(units=num_hidden, flatten=False, use_bias=True, dtype=dtype)
        self.kv2h = quantization.QuantizableDense(units=num_hidden * 2, flatten=False, use_bias=True, dtype=dtype)
        self.dot_att = DotAttentionCell()

    def forward(self, queries: np.ndarray,
                      memory: np.ndarray,
                      memory_lengths: np.ndarray) -> np.ndarray:
        """
        Apply project, apply dot attention and return new context vectors.

        :param queries: ndarray of shape (queries_max_length, batch, input_num_hidden).
        :param memory: ndarray of shape (memory_max_length, batch, input_num_hidden).
        :param memory_lengths: ndarray of shape (batch, 1).
        :return: ndarray of shape (queries_max_length, batch, num_hidden).
        """
        # (memory_max_length, batch, num_hidden * 2)
        combined = self.kv2h(memory)

        # (queries_max_length, batch, num_hidden)
        queries = self.q2h(queries)

        # (queries_max_length, batch, num_hidden)
        contexts = self.dot_att(queries, combined, 1, memory_lengths, None)

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


class PositionalEmbeddings(gluon.HybridBlock):
    """
    Takes an encoded sequence and adds sinusoidal or learned positional embeddings as in Vaswani et al, 2017 to it.

    :param weight_type: type of embeddings, fixed or learned.
    :param num_embed: Embedding size.
    :param max_seq_len: Maximum sequence length.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    :param weight_init: Optional initializer for learned embeddings.
    """

    def __init__(self,
                 weight_type: str,
                 num_embed: int,
                 max_seq_len: int,
                 scale_up_input: bool,
                 scale_down_positions: bool,
                 weight_init: Optional[Union[str, mx.init.Initializer]] = None) -> None:
        utils.check_condition(num_embed % 2 == 0, "Positional embeddings require an even embedding size it "
                                                  "is however %d." % num_embed)
        super().__init__()
        self.weight_type = weight_type
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions

        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            pos_weight = get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
            if self.scale_down_positions:
                pos_weight *= self.num_embed ** -0.5
            self.weight = mx.gluon.Constant(pos_weight)
        elif self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight = mx.gluon.Parameter('weight', shape=(self.max_seq_len, self.num_embed), init=weight_init)
        else:
            raise ValueError("weight_type '%s' is not supported!" % self.weight_type)

    def forward(self, data, steps):  # pylint: disable=arguments-differ
        """
        Applies positional embeddings to input data.

        :param data: Input data. Shape: (batch, length or 1, num_embed)
        :param steps: Optional steps input. If given, shape is (batch_size or 1, seq_len,)

        :return: Data with positional embeddings added
        """
        # (length, num_embed)
        if steps is None:
            # (batch, length, num_embed)
            pos_embedding = npx.slice_like(np.expand_dims(self.weight.data(), axis=0), data, axes=(1,))
        else:
            # (batch_size or 1, seq_len, num_embed)
            pos_embedding = npx.embedding(steps, self.weight.data(), self.max_seq_len, self.num_embed)

        if self.weight_type == 'fixed':
            pos_embedding = npx.stop_gradient(pos_embedding)

        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        return data + pos_embedding


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
    :param dtype: data type
    """
    def __init__(self,
                 model_size: int,
                 inference_only: bool,
                 dtype: str = C.DTYPE_FP32) -> None:
        super(SSRU, self).__init__()

        self.model_size = model_size
        self.inference_only = inference_only

        self.cell_state_transform = self._inference_cell_state_transform \
                                    if inference_only else self._training_cell_state_transform

        self.forget_gate = quantization.QuantizableDense(in_units=model_size,
                                                         units=model_size,
                                                         activation="sigmoid",
                                                         flatten=False,
                                                         dtype=dtype)

        self.linear = quantization.QuantizableDense(in_units=model_size,
                                                    units=model_size,
                                                    use_bias=False,
                                                    flatten=False,
                                                    dtype=dtype)

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
        return 1, batch_size, self.model_size

    @staticmethod
    def _training_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) -> Tuple[np.ndarray,
                                                                                                    np.ndarray]:
        """Update SSRU cell at training time"""

        def _time_step_update(step_input_and_forget_rate, previous_step_state) -> Tuple[np.ndarray, np.ndarray]:
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

        # (max_length, batch, input_depth), (batch, input_depth)
        cell_state, last_step_state = npx.foreach(_time_step_update,
                                                  [weighted_inputs, forget_rates],
                                                  np.squeeze(previous_cell_state, axis=0))

        return cell_state, np.expand_dims(last_step_state, axis=0)

    @staticmethod
    def _inference_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) -> Tuple[np.ndarray,
                                                                                                     np.ndarray]:
        """Update SSRU cell at inference time"""
        new_step_state = forget_rates * previous_cell_state + weighted_inputs  # (1, batch, input_depth)
        return new_step_state, new_step_state

    def forward(self, inputs: np.ndarray, previous_states: np.ndarray, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param inputs: input data. Shape: (max_length, batch, input_depth).
        :param previous_states: previous cell states. Shape: (max_length, batch, input_depth)
        :return: cell output and new cell states.  Both with shape (max_length, batch, input_depth).
        """
        forget_rates = self.forget_gate(inputs)
        weighted_inputs = (1 - forget_rates) * self.linear(inputs)

        cell_state, last_step_state = self.cell_state_transform(previous_states, weighted_inputs, forget_rates)

        return npx.relu(cell_state), last_step_state

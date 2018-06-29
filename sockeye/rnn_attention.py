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

"""
Implementations of different attention mechanisms in sequence-to-sequence models.
"""
import logging
import inspect
from typing import Callable, NamedTuple, Optional, Tuple, Dict, Type

import numpy as np
import mxnet as mx

from . import config
from . import constants as C
from . import coverage
from . import layers
from . import utils

logger = logging.getLogger(__name__)


class AttentionConfig(config.Config):
    """
    Attention configuration.

    :param type: Attention name.
    :param num_hidden: Number of hidden units for attention networks.
    :param input_previous_word: Feeds the previous target embedding into the attention mechanism.
    :param source_num_hidden: Number of hidden units of the source.
    :param query_num_hidden: Number of hidden units of the query.
    :param layer_normalization: Apply layer normalization to MLP attention.
    :param config_coverage: Optional coverage configuration.
    :param num_heads: Number of attention heads. Only used for Multi-head dot attention.
    :param is_scaled: If 'dot' attentions should be scaled.
    :param dtype: Data type.
    """
    def __init__(self,
                 type: str,
                 num_hidden: int,
                 input_previous_word: bool,
                 source_num_hidden: int,
                 query_num_hidden: int,
                 layer_normalization: bool,
                 config_coverage: Optional[coverage.CoverageConfig] = None,
                 num_heads: Optional[int] = None,
                 is_scaled: Optional[bool] = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.type = type
        self.num_hidden = num_hidden
        self.input_previous_word = input_previous_word
        self.source_num_hidden = source_num_hidden
        self.query_num_hidden = query_num_hidden
        self.layer_normalization = layer_normalization
        self.config_coverage = config_coverage
        self.num_heads = num_heads
        self.is_scaled = is_scaled
        self.dtype = dtype


def _instantiate(cls, params):
    """
    Helper to instantiate Attention classes from parameters. Warns in log if parameter is not supported
    by class constructor.

    :param cls: Attention class.
    :param params: configuration parameters.
    :return: instance of `cls` type.
    """
    sig_params = inspect.signature(cls.__init__).parameters
    valid_params = dict()
    for key, value in params.items():
        if key in sig_params:
            valid_params[key] = value
        else:
            logger.debug('Type %s does not support parameter \'%s\'' % (cls.__name__, key))
    return cls(**valid_params)


def get_attention(config: AttentionConfig, max_seq_len: int, prefix: str = C.ATTENTION_PREFIX) -> 'Attention':
    """
    Returns an Attention instance based on attention_type.

    :param config: Attention configuration.
    :param max_seq_len: Maximum length of source sequences.
    :param prefix: Name prefix.
    :return: Instance of Attention.
    """

    att_cls = Attention.get_attention_cls(config.type)
    params = config.__dict__.copy()
    params.pop('_frozen')
    params['max_seq_len'] = max_seq_len
    params['prefix'] = prefix
    return _instantiate(att_cls, params)


AttentionInput = NamedTuple('AttentionInput', [('seq_idx', int), ('query', mx.sym.Symbol)])
"""
Input to attention callables.

:param seq_idx: Decoder time step / sequence index.
:param query: Query input to attention mechanism, e.g. decoder hidden state (plus previous word).
"""

AttentionState = NamedTuple('AttentionState', [
    ('context', mx.sym.Symbol),
    ('probs', mx.sym.Symbol),
    ('dynamic_source', mx.sym.Symbol),
])
"""
Results returned from attention callables.

:param context: Context vector (Bahdanau et al, 15). Shape: (batch_size, encoder_num_hidden)
:param probs: Attention distribution over source encoder states. Shape: (batch_size, source_seq_len).
:param dynamic_source: Dynamically updated source encoding.
       Shape: (batch_size, source_seq_len, dynamic_source_num_hidden)
"""


class Attention(object):
    """
    Generic attention interface that returns a callable for attending to source states.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param dynamic_source_num_hidden: Number of hidden units of dynamic source encoding update mechanism.
    :param dtype: Data type.
    """

    __registry = {}  # type: Dict[str, Type['Attention']]

    @classmethod
    def register(cls, att_type: str):
        def wrapper(target_cls):
            cls.__registry[att_type] = target_cls
            return target_cls
        return wrapper

    @classmethod
    def get_attention_cls(cls, att_type: str):
        if att_type not in cls.__registry:
            raise ValueError('Unknown attention type %s' % att_type)
        return cls.__registry[att_type]

    def __init__(self,
                 input_previous_word: bool,
                 dynamic_source_num_hidden: int = 1,
                 prefix: str = C.ATTENTION_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        self.dynamic_source_num_hidden = dynamic_source_num_hidden
        self._input_previous_word = input_previous_word
        self.prefix = prefix
        self.dtype = dtype

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """
            raise NotImplementedError()

        return attend

    def get_initial_state(self, source_length: mx.sym.Symbol, source_seq_len: int) -> AttentionState:
        """
        Returns initial attention state. Dynamic source encoding is initialized with zeros.

        :param source_length: Source length. Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        """
        dynamic_source = mx.sym.reshape(mx.sym.zeros_like(source_length), shape=(-1, 1, 1))
        # dynamic_source: (batch_size, source_seq_len, num_hidden_dynamic_source)
        dynamic_source = mx.sym.broadcast_to(dynamic_source, shape=(0, source_seq_len, self.dynamic_source_num_hidden))
        return AttentionState(context=None, probs=None, dynamic_source=dynamic_source)

    def make_input(self,
                   seq_idx: int,
                   word_vec_prev: mx.sym.Symbol,
                   decoder_state: mx.sym.Symbol) -> AttentionInput:
        """
        Returns AttentionInput to be fed into the attend callable returned by the on() method.

        :param seq_idx: Decoder time step.
        :param word_vec_prev: Embedding of previously predicted ord
        :param decoder_state: Current decoder state
        :return: Attention input.
        """
        query = decoder_state
        if self._input_previous_word:
            # (batch_size, num_target_embed + rnn_num_hidden)
            query = mx.sym.concat(word_vec_prev, decoder_state, dim=1,
                                  name='%sconcat_prev_word_%d' % (self.prefix, seq_idx))
        return AttentionInput(seq_idx=seq_idx, query=query)


@Attention.register(C.ATT_BILINEAR)
class BilinearAttention(Attention):
    """
    Bilinear attention based on Luong et al. 2015.

    :math:`score(h_t, h_s) = h_t^T \\mathbf{W} h_s`

    For implementation reasons we modify to:

    :math:`score(h_t, h_s) = h_s^T \\mathbf{W} h_t`

    :param query_num_hidden: Number of hidden units the source will be projected to.
    :param dtype: data type.
    :param prefix: Name prefix.
    """

    def __init__(self, query_num_hidden: int, dtype: str = C.DTYPE_FP32, prefix: str = C.ATTENTION_PREFIX) -> None:
        super().__init__(False, dtype=dtype, prefix=prefix)
        self.num_hidden = query_num_hidden
        self.s2t_weight = mx.sym.Variable("%ss2t_weight" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        # (batch_size, seq_len, self.num_hidden)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.s2t_weight,
                                              num_hidden=self.num_hidden,
                                              no_bias=True,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """
            # (batch_size, decoder_num_hidden, 1)
            query = mx.sym.expand_dims(att_input.query, axis=2)

            # in:  (batch_size, source_seq_len, self.num_hidden) X (batch_size, self.num_hidden, 1)
            # out: (batch_size, source_seq_len, 1).
            attention_scores = mx.sym.batch_dot(lhs=source_hidden, rhs=query, name="%sbatch_dot" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores,
                                                                       self.dtype)

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


@Attention.register(C.ATT_DOT)
class DotAttention(Attention):
    """
    Attention mechanism with dot product between encoder and decoder hidden states [Luong et al. 2015].

    :math:`score(h_t, h_s) =  \\langle h_t, h_s \\rangle`

    :math:`a = softmax(score(*, h_s))`

    If rnn_num_hidden != num_hidden, states are projected with additional parameters to num_hidden.

    :math:`score(h_t, h_s) = \\langle \\mathbf{W}_t h_t, \\mathbf{W}_s h_s \\rangle`

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param source_num_hidden: Number of hidden units in source.
    :param query_num_hidden: Number of hidden units in query.
    :param num_hidden: Number of hidden units.
    :param is_scaled: Optionally scale query before dot product [Vaswani et al, 2017].
    :param prefix: Name prefix.
    :param dtype: data type.
    """

    def __init__(self,
                 input_previous_word: bool,
                 source_num_hidden: int,
                 query_num_hidden: int,
                 num_hidden: int,
                 is_scaled: bool = False,
                 prefix: str = C.ATTENTION_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(input_previous_word, dtype=dtype, prefix=prefix)
        self.project_source = source_num_hidden != num_hidden
        self.project_query = query_num_hidden != num_hidden
        self.num_hidden = num_hidden
        self.is_scaled = is_scaled
        self.scale = num_hidden ** -0.5 if is_scaled else None
        self.s2h_weight = mx.sym.Variable("%ss2h_weight" % self.prefix) if self.project_source else None
        self.t2h_weight = mx.sym.Variable("%st2h_weight" % self.prefix) if self.project_query else None

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        if self.project_source:
            # (batch_size, seq_len, self.num_hidden)
            source_hidden = mx.sym.FullyConnected(data=source,
                                                  weight=self.s2h_weight,
                                                  num_hidden=self.num_hidden,
                                                  no_bias=True,
                                                  flatten=False,
                                                  name="%ssource_hidden_fc" % self.prefix)
        else:
            source_hidden = source

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """
            query = att_input.query
            if self.project_query:
                # query: (batch_size, self.num_hidden)
                query = mx.sym.FullyConnected(data=query,
                                              weight=self.t2h_weight,
                                              num_hidden=self.num_hidden,
                                              no_bias=True, name="%squery_hidden_fc" % self.prefix)

            # scale down dot product by sqrt(num_hidden) [Vaswani et al, 17]
            if self.is_scaled:
                query = query * self.scale

            # (batch_size, decoder_num_hidden, 1)
            expanded_decoder_state = mx.sym.expand_dims(query, axis=2)

            # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.batch_dot(lhs=source_hidden, rhs=expanded_decoder_state,
                                                name="%sbatch_dot" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores,
                                                                       self.dtype)
            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


@Attention.register(C.ATT_MH_DOT)
class MultiHeadDotAttention(Attention):
    """
    Dot product attention with multiple heads as proposed in Vaswani et al, Attention is all you need.
    Can be used with a RecurrentDecoder.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param source_num_hidden: Number of hidden units.
    :param num_heads: Number of attention heads / independently computed attention scores.
    :param prefix: Name prefix.
    :param dtype: data type.
    """

    def __init__(self,
                 input_previous_word: bool,
                 source_num_hidden: int,
                 num_heads: int,
                 prefix: str = C.ATTENTION_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(input_previous_word, dtype=dtype, prefix=prefix)
        utils.check_condition(num_heads is not None, "%s requires setting num-heads." % C.ATT_MH_DOT)
        utils.check_condition(source_num_hidden % num_heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (num_heads, source_num_hidden))
        self.num_hidden = source_num_hidden
        self.heads = num_heads
        self.num_hidden_per_head = self.num_hidden // self.heads
        self.s2h_weight = mx.sym.Variable("%ss2h_weight" % self.prefix)
        self.s2h_bias = mx.sym.Variable("%ss2h_bias" % self.prefix)
        self.t2h_weight = mx.sym.Variable("%st2h_weight" % self.prefix)
        self.t2h_bias = mx.sym.Variable("%st2h_bias" % self.prefix)
        self.h2o_weight = mx.sym.Variable("%sh2o_weight" % self.prefix)
        self.h2o_bias = mx.sym.Variable("%sh2o_bias" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """
        # (batch, length, num_hidden * 2)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.s2h_weight,
                                              bias=self.s2h_bias,
                                              num_hidden=self.num_hidden * 2,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)
        # split keys and values
        # (batch, length, num_hidden)
        # pylint: disable=unbalanced-tuple-unpacking
        keys, values = mx.sym.split(data=source_hidden, num_outputs=2, axis=2)

        # (batch*heads, length, num_hidden/head)
        keys = layers.split_heads(keys, self.num_hidden_per_head, self.heads)
        values = layers.split_heads(values, self.num_hidden_per_head, self.heads)

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """
            # (batch, num_hidden)
            query = mx.sym.FullyConnected(data=att_input.query,
                                          weight=self.t2h_weight, bias=self.t2h_bias,
                                          num_hidden=self.num_hidden, name="%squery_hidden_fc" % self.prefix)
            # (batch, length, heads, num_hidden/head)
            query = mx.sym.reshape(query, shape=(0, 1, self.heads, self.num_hidden_per_head))
            # (batch, heads, num_hidden/head, length)
            query = mx.sym.transpose(query, axes=(0, 2, 3, 1))
            # (batch * heads, num_hidden/head, 1)
            query = mx.sym.reshape(query, shape=(-3, self.num_hidden_per_head, 1))

            # scale dot product
            query = query * (self.num_hidden_per_head ** -0.5)

            # (batch*heads, length, num_hidden/head) X (batch*heads, num_hidden/head, 1)
            #   -> (batch*heads, length, 1)
            attention_scores = mx.sym.batch_dot(lhs=keys, rhs=query, name="%sdot" % self.prefix)

            # (batch*heads, 1)
            lengths = layers.broadcast_to_heads(source_length, self.heads, ndim=1, fold_heads=True)

            # context: (batch*heads, num_hidden/head)
            # attention_probs: (batch*heads, length)
            context, attention_probs = get_context_and_attention_probs(values, lengths, attention_scores, self.dtype)

            # combine heads
            # (batch*heads, 1, num_hidden/head)
            context = mx.sym.expand_dims(context, axis=1)
            # (batch, 1, num_hidden)
            context = layers.combine_heads(context, self.num_hidden_per_head, heads=self.heads)
            # (batch, num_hidden)
            context = mx.sym.reshape(context, shape=(-3, -1))

            # (batch, heads, length)
            attention_probs = mx.sym.reshape(data=attention_probs, shape=(-4, -1, self.heads, source_seq_len))
            # just average over distributions
            attention_probs = mx.sym.mean(attention_probs, axis=1, keepdims=False)

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


@Attention.register(C.ATT_FIXED)
class EncoderLastStateAttention(Attention):
    """
    Always returns the last encoder state independent of the query vector.
    Equivalent to no attention.
    """

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """
        encoder_last_state = mx.sym.SequenceLast(data=source, axis=1, sequence_length=source_length,
                                                 use_sequence_length=True)
        fixed_probs = mx.sym.one_hot(source_length - 1, depth=source_seq_len)

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            return AttentionState(context=encoder_last_state,
                                  probs=fixed_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


@Attention.register(C.ATT_LOC)
class LocationAttention(Attention):
    """
    Attends to locations in the source [Luong et al, 2015]

    :math:`a_t = softmax(\\mathbf{W}_a h_t)` for decoder hidden state at time t.

    :note: :math:`\\mathbf{W}_a` is of shape (max_source_seq_len, decoder_num_hidden).

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param max_seq_len: Maximum length of source sequences.
    :param prefix: Name prefix.
    :param dtype: data type.
    """

    def __init__(self,
                 input_previous_word: bool,
                 max_seq_len: int,
                 prefix: str = C.ATTENTION_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(input_previous_word, dtype=dtype, prefix=prefix)
        self.max_source_seq_len = max_seq_len
        self.location_weight = mx.sym.Variable("%sloc_weight" % self.prefix)
        self.location_bias = mx.sym.Variable("%sloc_bias" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """
            # attention_scores: (batch_size, seq_len)
            attention_scores = mx.sym.FullyConnected(data=att_input.query,
                                                     num_hidden=self.max_source_seq_len,
                                                     weight=self.location_weight,
                                                     bias=self.location_bias)

            # attention_scores: (batch_size, seq_len)
            attention_scores = mx.sym.slice_axis(data=attention_scores,
                                                 axis=1,
                                                 begin=0,
                                                 end=source_seq_len)

            # attention_scores: (batch_size, seq_len, 1)
            attention_scores = mx.sym.expand_dims(data=attention_scores, axis=2)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores,
                                                                       self.dtype)
            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


@Attention.register(C.ATT_MLP)
class MlpAttention(Attention):
    """
    Attention computed through a one-layer MLP with num_hidden units [Luong et al, 2015].

    :math:`score(h_t, h_s) = \\mathbf{W}_a tanh(\\mathbf{W}_c [h_t, h_s] + b)`

    :math:`a = softmax(score(*, h_s))`

    Optionally, if attention_coverage_type is not None, attention uses dynamic source encoding ('coverage' mechanism)
    as in Tu et al. (2016): Modeling Coverage for Neural Machine Translation.

    :math:`score(h_t, h_s) = \\mathbf{W}_a tanh(\\mathbf{W}_c [h_t, h_s, c_s] + b)`

    :math:`c_s` is the decoder time-step dependent source encoding which is updated using the current
    decoder state.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param num_hidden: Number of hidden units.
    :param layer_normalization: If true, normalizes hidden layer outputs before tanh activation.
    :param prefix: Name prefix
    :param dtype: data type.
    """

    def __init__(self,
                 input_previous_word: bool,
                 num_hidden: int,
                 layer_normalization: bool = False,
                 prefix: str = C.ATTENTION_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(input_previous_word=input_previous_word,
                         dynamic_source_num_hidden=1,
                         prefix=prefix,
                         dtype=dtype)
        self.attention_num_hidden = num_hidden
        # input (encoder) to hidden
        self.att_e2h_weight = mx.sym.Variable("%se2h_weight" % self.prefix)
        # input (query) to hidden
        self.att_q2h_weight = mx.sym.Variable("%sq2h_weight" % self.prefix)
        # hidden to score
        self.att_h2s_weight = mx.sym.Variable("%sh2s_weight" % self.prefix)
        # coverage
        self.coverage = None  # type: Optional[coverage.Coverage]
        # dynamic source (coverage) weights and settings
        # input (coverage) to hidden
        self.att_c2h_weight = None
        # layer normalization
        self._ln = None
        if layer_normalization:
            self._ln = layers.LayerNormalization(prefix="%snorm" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for recurrent attention in a sequence decoder.
        The callable is a recurrent function of the form:
        AttentionState = attend(AttentionInput, AttentionState).

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Attention callable.
        """

        if self.coverage is not None:
            coverage_func = self.coverage.on(source, source_length, source_seq_len)

        # (batch_size, seq_len, attention_num_hidden)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.att_e2h_weight,
                                              num_hidden=self.attention_num_hidden,
                                              no_bias=True,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """

            # (batch_size, attention_num_hidden)
            query_hidden = mx.sym.FullyConnected(data=att_input.query,
                                                 weight=self.att_q2h_weight,
                                                 num_hidden=self.attention_num_hidden,
                                                 no_bias=True,
                                                 name="%squery_hidden" % self.prefix)

            # (batch_size, 1, attention_num_hidden)
            query_hidden = mx.sym.expand_dims(data=query_hidden,
                                              axis=1,
                                              name="%squery_hidden_expanded" % self.prefix)

            attention_hidden_lhs = source_hidden
            if self.coverage:
                # (batch_size, seq_len, attention_num_hidden)
                dynamic_hidden = mx.sym.FullyConnected(data=att_state.dynamic_source,
                                                       weight=self.att_c2h_weight,
                                                       num_hidden=self.attention_num_hidden,
                                                       no_bias=True,
                                                       flatten=False,
                                                       name="%sdynamic_source_hidden_fc" % self.prefix)

                # (batch_size, seq_len, attention_num_hidden
                attention_hidden_lhs = dynamic_hidden + source_hidden

            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.broadcast_add(lhs=attention_hidden_lhs, rhs=query_hidden,
                                                    name="%squery_plus_input" % self.prefix)

            if self._ln is not None:
                attention_hidden = self._ln(data=attention_hidden)

            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.Activation(attention_hidden, act_type="tanh",
                                                 name="%shidden" % self.prefix)

            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.FullyConnected(data=attention_hidden,
                                                     weight=self.att_h2s_weight,
                                                     num_hidden=1,
                                                     no_bias=True,
                                                     flatten=False,
                                                     name="%sraw_att_score_fc" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores,
                                                                       self.dtype)

            dynamic_source = att_state.dynamic_source
            if self.coverage is not None:
                # update dynamic source encoding
                # Note: this is a slight change to the Tu et al, 2016 paper: input to the coverage update
                # is the attention input query, not the previous decoder state.
                dynamic_source = coverage_func(prev_hidden=att_input.query,
                                               attention_prob_scores=attention_probs,
                                               prev_coverage=att_state.dynamic_source)

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=dynamic_source)

        return attend


@Attention.register(C.ATT_COV)
class MlpCovAttention(MlpAttention):
    """
    MlpAttention with optional coverage config.

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param num_hidden: Number of hidden units.
    :param layer_normalization: If true, normalizes hidden layer outputs before tanh activation.
    :param config_coverage: coverage config.
    :param prefix: Name prefix.
    :param dtype: data type.
    """

    def __init__(self,
                 input_previous_word: bool,
                 num_hidden: int,
                 layer_normalization: bool = False,
                 config_coverage: coverage.CoverageConfig = None,
                 prefix: str = C.ATTENTION_PREFIX,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(input_previous_word=input_previous_word,
                         num_hidden=num_hidden,
                         layer_normalization=layer_normalization,
                         prefix=prefix,
                         dtype=dtype)
        self.coverage = coverage.get_coverage(config_coverage)
        self.dynamic_source_num_hidden = config_coverage.num_hidden
        self.att_c2h_weight = mx.sym.Variable("%sc2h_weight" % self.prefix)


def get_context_and_attention_probs(values: mx.sym.Symbol,
                                    length: mx.sym.Symbol,
                                    logits: mx.sym.Symbol,
                                    dtype: str) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
    """
    Returns context vector and attention probabilities
    via a weighted sum over values.

    :param values: Shape: (batch_size, seq_len, encoder_num_hidden).
    :param length: Shape: (batch_size,).
    :param logits: Shape: (batch_size, seq_len, 1).
    :param dtype: data type.
    :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
    """
    # masks attention scores according to sequence length.
    # (batch_size, seq_len, 1)
    logits = mx.sym.SequenceMask(data=logits,
                                 axis=1,
                                 use_sequence_length=True,
                                 sequence_length=length,
                                 value=-C.LARGE_VALUES[dtype])

    # (batch_size, seq_len, 1)
    probs = mx.sym.softmax(logits, axis=1, name='attention_softmax')

    # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
    # (batch_size, seq_len, num_hidden) X (batch_size, seq_len, 1) -> (batch_size, num_hidden, 1)
    context = mx.sym.batch_dot(lhs=values, rhs=probs, transpose_a=True)
    # (batch_size, encoder_num_hidden, 1)-> (batch_size, encoder_num_hidden)
    context = mx.sym.reshape(data=context, shape=(0, 0))
    probs = mx.sym.reshape(data=probs, shape=(0, 0))

    return context, probs

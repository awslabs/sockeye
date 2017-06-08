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
from typing import Callable, NamedTuple, Optional, Tuple

import mxnet as mx

import sockeye.coverage

logger = logging.getLogger(__name__)


def get_attention(input_previous_word: bool,
                  attention_type: str,
                  attention_num_hidden: int,
                  rnn_num_hidden: int,
                  max_seq_len: int,
                  attention_coverage_type: str,
                  attention_coverage_num_hidden: int) -> 'Attention':
    """
    Returns an Attention instance based on attention_type.

    :param input_previous_word: Feeds the previous target embedding into the attention mechanism.
    :param attention_type: Attention name.
    :param attention_num_hidden: Number of hidden units for attention networks.
    :param rnn_num_hidden: Number of hidden units of encoder/decoder RNNs.
    :param max_seq_len: Maximum length of source sequences.
    :param attention_coverage_type: The type of update for the dynamic source encoding.
    :param attention_coverage_num_hidden: Number of hidden units for coverage attention.
    :return: Instance of Attention.
    """
    if attention_type == "bilinear":
        if input_previous_word:
            logger.warning("bilinear attention does not support input_previous_word")
        return BilinearAttention(rnn_num_hidden)
    elif attention_type == "dot":
        return DotAttention(input_previous_word, rnn_num_hidden, attention_num_hidden)
    elif attention_type == "fixed":
        return EncoderLastStateAttention(input_previous_word)
    elif attention_type == "location":
        return LocationAttention(input_previous_word, max_seq_len)
    elif attention_type == "mlp":
        return MlpAttention(input_previous_word=input_previous_word,
                            attention_num_hidden=attention_num_hidden)
    elif attention_type == "coverage":
        if attention_coverage_type == 'count' and attention_coverage_num_hidden != 1:
            logging.warning("Ignoring coverage_num_hidden=%d and setting to 1" % attention_coverage_num_hidden)
            attention_coverage_num_hidden = 1
        return MlpAttention(input_previous_word=input_previous_word,
                            attention_num_hidden=attention_num_hidden,
                            attention_coverage_type=attention_coverage_type,
                            attention_coverage_num_hidden=attention_coverage_num_hidden)
    else:
        raise ValueError("Unknown attention type %s" % attention_type)


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
    """

    def __init__(self, input_previous_word: bool, dynamic_source_num_hidden: int = 1) -> None:
        self.dynamic_source_num_hidden = dynamic_source_num_hidden
        self._input_previous_word = input_previous_word

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
        dynamic_source = mx.sym.expand_dims(mx.sym.expand_dims(mx.sym.zeros_like(source_length), axis=1), axis=2)
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
            query = mx.sym.concat(word_vec_prev, decoder_state, dim=1, name='att_concat_prev_word_%d' % seq_idx)
        return AttentionInput(seq_idx=seq_idx, query=query)


class BilinearAttention(Attention):
    """
    Bilinear attention based on Luong et al. 2015.

    :math:`score(h_t, h_s) = h_t^T \\mathbf{W} h_s`
    
    For implementation reasons we modify to:

    :math:`score(h_t, h_s) = h_s^T \\mathbf{W} h_t`

    :param num_hidden: Number of hidden units.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = '') -> None:
        super().__init__(False)
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.s2t_weight = mx.sym.Variable("%satt_s2t_weight", self.prefix)

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

        # (batch_size * seq_len, self.num_hidden)
        source_hidden = mx.sym.FullyConnected(data=mx.sym.reshape(data=source, shape=(-3, -1), name="att_flat_source"),
                                              weight=self.s2t_weight, num_hidden=self.num_hidden,
                                              no_bias=True, name="att_source_hidden_fc")
        # (batch_size, seq_len, self.num_hidden)
        source_hidden = mx.sym.reshape(source_hidden, shape=(-1, source_seq_len, self.num_hidden),
                                       name="att_source_hidden")

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
            attention_scores = mx.sym.batch_dot(lhs=source_hidden, rhs=query, name="att_batch_dot")

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)

            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class DotAttention(Attention):
    """
    Attention mechanism with dot product between encoder and decoder hidden states [Luong et al. 2015].

    :math:`score(h_t, h_s) =  \\langle h_t, h_s \\rangle`

    :math:`a = softmax(score(*, h_s))`

    If rnn_num_hidden != num_hidden, states are projected with additional parameters to num_hidden.

    :math:`score(h_t, h_s) = \\langle \\mathbf{W}_t h_t, \\mathbf{W}_s h_s \\rangle`

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param rnn_num_hidden: Number of hidden units in encoder/decoder RNNs.
    :param num_hidden: Number of hidden units.
    """

    def __init__(self,
                 input_previous_word: bool,
                 rnn_num_hidden: int,
                 num_hidden: int) -> None:
        super().__init__(input_previous_word)
        self.project = rnn_num_hidden != num_hidden
        self.num_hidden = num_hidden
        self.t2h_weight = mx.sym.Variable("att_t2h_weight") if self.project else None
        self.s2h_weight = mx.sym.Variable("att_s2h_weight") if self.project else None

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

        if self.project:
            # (batch_size * seq_len, self.num_hidden)
            source_hidden = mx.sym.FullyConnected(
                data=mx.sym.reshape(data=source, shape=(-3, -1), name="att_flat_source"),
                weight=self.s2h_weight, num_hidden=self.num_hidden,
                no_bias=True, name="att_source_hidden_fc")
            # (batch_size, seq_len, self.num_hidden)
            source_hidden = mx.sym.reshape(source_hidden, shape=(-1, source_seq_len, self.num_hidden),
                                           name="att_source_hidden")

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            """
            Returns updated attention state given attention input and current attention state.

            :param att_input: Attention input as returned by make_input().
            :param att_state: Current attention state
            :return: Updated attention state.
            """
            query = att_input.query
            local_source = source
            if self.project:
                local_source = source_hidden
                # query: (batch_size, self.num_hidden)
                query = mx.sym.FullyConnected(data=query,
                                              weight=self.t2h_weight,
                                              num_hidden=self.num_hidden,
                                              no_bias=True, name="att_query_hidden_fc")

            # (batch_size, decoder_num_hidden, 1)
            expanded_decoder_state = mx.sym.expand_dims(query, axis=2)

            # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.batch_dot(lhs=local_source, rhs=expanded_decoder_state, name="att_batch_dot")

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)
            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


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
        source = mx.sym.swapaxes(source, dim1=0, dim2=1)
        encoder_last_state = mx.sym.SequenceLast(data=source, sequence_length=source_length,
                                                 use_sequence_length=True)
        fixed_probs = mx.sym.one_hot(source_length - 1, depth=source_seq_len)

        def attend(att_input: AttentionInput, att_state: AttentionState) -> AttentionState:
            return AttentionState(context=encoder_last_state,
                                  probs=fixed_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


class LocationAttention(Attention):
    """
    Attends to locations in the source [Luong et al, 2015]

    :math:`a_t = softmax(\\mathbf{W}_a h_t)` for decoder hidden state at time t.

    :note: :math:`\\mathbf{W}_a` is of shape (max_source_seq_len, decoder_num_hidden).

    :param input_previous_word: Feed the previous target embedding into the attention mechanism.
    :param max_source_seq_len: Maximum length of source sequences.
    """

    def __init__(self, input_previous_word: bool, max_source_seq_len: int) -> None:
        super().__init__(input_previous_word)
        self.max_source_seq_len = max_source_seq_len
        self.location_weight = mx.sym.Variable("att_loc_weight")
        self.location_bias = mx.sym.Variable("att_loc_bias")

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

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)
            return AttentionState(context=context,
                                  probs=attention_probs,
                                  dynamic_source=att_state.dynamic_source)

        return attend


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
    :param attention_num_hidden: Number of hidden units.
    :param attention_coverage_type: The type of update for the dynamic source encoding.
           If None, no dynamic source encoding is done.
    :param attention_coverage_num_hidden: Number of hidden units for coverage attention.
    """

    def __init__(self,
                 input_previous_word: bool,
                 attention_num_hidden: int,
                 attention_coverage_type: Optional[str] = None,
                 attention_coverage_num_hidden: int = 1,
                 prefix='') -> None:
        dynamic_source_num_hidden = 1 if attention_coverage_type is None else attention_coverage_num_hidden
        super().__init__(input_previous_word=input_previous_word,
                         dynamic_source_num_hidden=dynamic_source_num_hidden)
        self.prefix = prefix
        self.attention_num_hidden = attention_num_hidden
        # input (encoder) to hidden
        self.att_e2h_weight = mx.sym.Variable("%satt_e2h_weight" % prefix)
        # input (query) to hidden
        self.att_q2h_weight = mx.sym.Variable("%satt_q2h_weight" % prefix)
        # hidden to score
        self.att_h2s_weight = mx.sym.Variable("%satt_h2s_weight" % prefix)
        # dynamic source (coverage) weights and settings
        # input (coverage) to hidden
        self.att_c2h_weight = mx.sym.Variable("%satt_c2h_weight" % prefix) if attention_coverage_type else None
        self.coverage = sockeye.coverage.get_coverage(attention_coverage_type,
                                                      dynamic_source_num_hidden) if attention_coverage_type else None

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

        coverage_func = self.coverage.on(source, source_length, source_seq_len) if self.coverage else None

        # (batch_size * seq_len, attention_num_hidden)
        source_hidden = mx.sym.FullyConnected(data=mx.sym.reshape(data=source,
                                                                  shape=(-3, -1),
                                                                  name="%satt_flat_source" % self.prefix),
                                              weight=self.att_e2h_weight,
                                              num_hidden=self.attention_num_hidden,
                                              no_bias=True,
                                              name="%satt_source_hidden_fc" % self.prefix)

        # (batch_size, seq_len, attention_num_hidden)
        source_hidden = mx.sym.reshape(source_hidden,
                                       shape=(-1, source_seq_len, self.attention_num_hidden),
                                       name="%satt_source_hidden" % self.prefix)

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
                                                 name="%satt_query_hidden" % self.prefix)

            # (batch_size, 1, attention_num_hidden)
            query_hidden = mx.sym.expand_dims(data=query_hidden,
                                              axis=1,
                                              name="%satt_query_hidden_expanded" % self.prefix)

            attention_hidden_lhs = source_hidden
            if self.coverage:
                # (batch_size * seq_len, attention_num_hidden)
                dynamic_hidden = mx.sym.FullyConnected(data=mx.sym.reshape(data=att_state.dynamic_source,
                                                                           shape=(-3, -1),
                                                                           name="%satt_flat_dynamic_source" % self.prefix),
                                                       weight=self.att_c2h_weight,
                                                       num_hidden=self.attention_num_hidden,
                                                       no_bias=True,
                                                       name="%satt_dynamic_source_hidden_fc" % self.prefix)

                # (batch_size, seq_len, attention_num_hidden)
                dynamic_hidden = mx.sym.reshape(dynamic_hidden,
                                                shape=(-1, source_seq_len, self.attention_num_hidden),
                                                name="%satt_dynamic_source_hidden" % self.prefix)

                # (batch_size, seq_len, attention_num_hidden
                attention_hidden_lhs = dynamic_hidden + source_hidden

            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.broadcast_add(lhs=attention_hidden_lhs, rhs=query_hidden,
                                                    name="%satt_query_plus_input" % self.prefix)

            # (batch_size * seq_len, attention_num_hidden)
            attention_hidden = mx.sym.reshape(data=attention_hidden,
                                              shape=(-3, -1),
                                              name="%satt_query_plus_input_before_fc" % self.prefix)

            # (batch_size * seq_len, attention_num_hidden)
            attention_hidden = mx.sym.Activation(attention_hidden, act_type="tanh",
                                                 name="%satt_hidden" % self.prefix)

            # (batch_size * seq_len, 1)
            attention_scores = mx.sym.FullyConnected(data=attention_hidden,
                                                     weight=self.att_h2s_weight,
                                                     num_hidden=1,
                                                     no_bias=True,
                                                     name="%sraw_att_score_fc" % self.prefix)

            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.reshape(attention_scores,
                                              shape=(-1, source_seq_len, 1),
                                              name="%sraw_att_score_fc" % self.prefix)

            context, attention_probs = get_context_and_attention_probs(source, source_length, attention_scores)

            dynamic_source = att_state.dynamic_source
            if self.coverage:
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


def get_context_and_attention_probs(source: mx.sym.Symbol,
                                    source_length: mx.sym.Symbol,
                                    attention_scores: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
    """
    Returns context vector and attention probs via a weighted sum over the masked, softmaxed attention scores.
    
    :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
    :param source_length: Shape: (batch_size,).
    :param attention_scores: Shape: (batch_size, seq_len, 1).
    :return: context: (batch_size, encoder_num_hidden), attention_probs: (batch_size, seq_len).
    """

    # TODO: It would be nice if SequenceMask could take a 2d input...
    # Note: we need to add an axis as SequenceMask expects 3D input
    # TODO: we should probably replace this with a multiplication of a 0-1 mask, to avoid the multiplication
    attention_scores = mx.sym.swapaxes(data=attention_scores, dim1=0, dim2=1)
    attention_scores = mx.sym.SequenceMask(data=attention_scores,
                                           use_sequence_length=True,
                                           sequence_length=source_length,
                                           value=-99999999.)
    attention_scores = mx.sym.swapaxes(data=attention_scores, dim1=0, dim2=1)
    # attention_scores is batch_major from here: (batch_size, seq_len, 1)

    # (batch_size, seq_len)
    attention_scores = mx.sym.reshape(data=attention_scores, shape=(0, 0))

    # (batch_size, seq_len)
    attention_probs = mx.sym.softmax(attention_scores, name='attention_softmax')

    # (batch_size, seq_len, 1)
    attention_probs_expanded = mx.sym.expand_dims(data=attention_probs, axis=2)

    # batch_dot: (batch, M, K) X (batch, K, N) –> (batch, M, N).
    # (batch_size, seq_len, encoder_num_hidden) X (batch_size, seq_len, 1) -> (batch_size, encoder_num_hidden)
    context = mx.sym.batch_dot(lhs=source, rhs=attention_probs_expanded, transpose_a=True)
    context = mx.sym.reshape(data=context, shape=(0, 0))

    return context, attention_probs

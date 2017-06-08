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
Defines the dynamic source encodings ('coverage' mechanisms) for encoder/decoder networks as used in Tu et al. (2016).
"""
import logging
from typing import Callable

import mxnet as mx

logger = logging.getLogger(__name__)


def get_coverage(coverage_type: str,
                 coverage_num_hidden: int) -> 'Coverage':
    """
    Returns a Coverage instance.

    :param coverage_type: Name of coverage type.
    :param coverage_num_hidden: Number of hidden units for coverage vectors.
    :return: Instance of Coverage.
    """

    if coverage_type == "gru":
        return GRUCoverage(coverage_num_hidden)
    elif coverage_type in {"tanh", "sigmoid", "relu", "softrelu"}:
        return ActivationCoverage(coverage_num_hidden, coverage_type)
    elif coverage_type == "count":
        return CountCoverage()
    else:
        raise ValueError("Unknown coverage type %s" % coverage_type)


class Coverage:
    """
    Generic coverage class. Similar to Attention classes, a coverage instance returns a callable, update_coverage(),
    function when self.on() is called.
    """

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for updating coverage vectors in a sequence decoder.

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Coverage callable.
        """

        def update_coverage(prev_hidden: mx.sym.Symbol,
                            attention_prob_scores: mx.sym.Symbol,
                            prev_coverage: mx.sym.Symbol):
            """
            :param prev_hidden: Previous hidden decoder state. Shape: (batch_size, decoder_num_hidden).
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len, 1).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """
            raise NotImplementedError()

        return update_coverage


class CountCoverage(Coverage):
    """
    Coverage class that accumulates the attention weights for each source word.
    """

    def __init__(self, prefix='') -> None:
        self.prefix = prefix

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for updating coverage vectors in a sequence decoder.

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Coverage callable.
        """

        def update_coverage(prev_hidden: mx.sym.Symbol,
                            attention_prob_scores: mx.sym.Symbol,
                            prev_coverage: mx.sym.Symbol):
            """
            :param prev_hidden: Previous hidden decoder state. Shape: (batch_size, decoder_num_hidden).
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len, 1).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """
            return prev_coverage + mx.sym.expand_dims(attention_prob_scores, axis=2)

        return update_coverage


class GRUCoverage(Coverage):
    """
    Implements a GRU whose state is the coverage vector.

    TODO: This implementation is slightly inefficient since the source is fed in at every step.
    It would be better to pre-compute the mapping of the source but this will likely mean opening up the GRU.

    :param coverage_num_hidden: Number of hidden units for coverage vectors.
    """

    def __init__(self, coverage_num_hidden: int, prefix='') -> None:
        self.prefix = prefix
        self.num_hidden = coverage_num_hidden
        self.gru = mx.rnn.GRUCell(self.num_hidden, prefix="%scoverage_gru" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for updating coverage vectors in a sequence decoder.

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Coverage callable.
        """

        def update_coverage(prev_hidden: mx.sym.Symbol,
                            attention_prob_scores: mx.sym.Symbol,
                            prev_coverage: mx.sym.Symbol):
            """
            :param prev_hidden: Previous hidden decoder state. Shape: (batch_size, decoder_num_hidden).
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len, 1).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """

            # (batch_size, source_seq_len, decoder_num_hidden)
            expanded_decoder = mx.sym.broadcast_axis(
                data=mx.sym.expand_dims(data=prev_hidden, axis=1, name="%scov_expand_decoder" % self.prefix),
                axis=1, size=source_seq_len, name="%scov_broadcast_decoder" % self.prefix)

            expanded_att_scores = mx.sym.expand_dims(data=attention_prob_scores,
                                                     axis=2,
                                                     name="%scov_expand_attention_scores" % self.prefix)

            # (batch_size, source_seq_len, encoder_num_hidden + decoder_num_hidden + 1)
            # +1 for the attention_prob_score for the source word
            concat_input = mx.sym.concat(source, expanded_decoder, expanded_att_scores, dim=2,
                                         name="%scov_concat_inputs" % self.prefix)

            # (batch_size * source_seq_len, encoder_num_hidden + decoder_num_hidden + 1)
            flat_input = mx.sym.reshape(concat_input, shape=(-3, -1), name="%scov_flatten_inputs")

            # coverage: (batch_size * seq_len, coverage_num_hidden)
            coverage = mx.sym.reshape(data=prev_coverage, shape=(-3, -1))
            updated_coverage, _ = self.gru(flat_input, states=[coverage])

            # coverage: (batch_size, seq_len, coverage_num_hidden)
            coverage = mx.sym.reshape(updated_coverage, shape=(-1, source_seq_len, self.num_hidden))

            return mask_coverage(coverage, source_length)

        return update_coverage


class ActivationCoverage(Coverage):
    """
    Implements a coverage mechanism whose updates are performed by a Perceptron with
    configurable activation function.

    :param coverage_num_hidden: Number of hidden units for coverage vectors.
    :param activation: Type of activation for Perceptron.
    """

    def __init__(self, coverage_num_hidden: int, activation: str, prefix='') -> None:
        self.prefix = prefix
        self.activation = activation
        self.num_hidden = coverage_num_hidden
        # input (encoder) to hidden
        self.cov_e2h_weight = mx.sym.Variable("%scov_e2h_weight" % self.prefix)
        # decoder to hidden
        self.cov_dec2h_weight = mx.sym.Variable("%scov_i2h_weight" % self.prefix)
        # previous coverage to hidden
        self.cov_prev2h_weight = mx.sym.Variable("%scov_prev2h_weight" % self.prefix)
        # attention scores to hidden
        self.cov_a2h_weight = mx.sym.Variable("%scov_a2h_weight" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for updating coverage vectors in a sequence decoder.

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Coverage callable.
        """

        # (batch_size * seq_len, coverage_hidden_num)
        source_hidden = mx.sym.FullyConnected(data=mx.sym.reshape(data=source,
                                                                  shape=(-3, -1),
                                                                  name="%scov_flat_source" % self.prefix),
                                              weight=self.cov_e2h_weight,
                                              no_bias=True,
                                              num_hidden=self.num_hidden,
                                              name="%scov_source_hidden_fc" % self.prefix)

        # (batch_size, seq_len, coverage_hidden_num)
        source_hidden = mx.sym.reshape(source_hidden,
                                       shape=(-1, source_seq_len, self.num_hidden),
                                       name="%scov_source_hidden" % self.prefix)

        def update_coverage(prev_hidden: mx.sym.Symbol,
                            attention_prob_scores: mx.sym.Symbol,
                            prev_coverage: mx.sym.Symbol):
            """
            :param prev_hidden: Previous hidden decoder state. Shape: (batch_size, decoder_num_hidden).
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len, 1).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """

            # (batch_size * seq_len, coverage_hidden_num)
            coverage_hidden = mx.sym.FullyConnected(data=mx.sym.reshape(data=prev_coverage,
                                                                        shape=(-3, -1),
                                                                        name="%scov_flat_previous" % self.prefix),
                                                    weight=self.cov_prev2h_weight,
                                                    no_bias=True,
                                                    num_hidden=self.num_hidden,
                                                    name="%scov_previous_hidden_fc" % self.prefix)

            # (batch_size, source_seq_len, coverage_hidden_num)
            coverage_hidden = mx.sym.reshape(coverage_hidden,
                                             shape=(-1, source_seq_len, self.num_hidden),
                                             name="%scov_previous_hidden" % self.prefix)

            # (batch_size, source_seq_len, 1)
            attention_prob_score = mx.sym.expand_dims(attention_prob_scores, axis=2)

            # (batch_size * source_seq_len, coverage_num_hidden)
            attention_hidden = mx.sym.FullyConnected(data=mx.sym.reshape(attention_prob_score,
                                                                         shape=(-3, 0),
                                                                         name="%scov_reshape_att_probs" % self.prefix),
                                                     weight=self.cov_a2h_weight,
                                                     no_bias=True,
                                                     num_hidden=self.num_hidden,
                                                     name="%scov_attention_fc" % self.prefix)

            # (batch_size, source_seq_len, coverage_num_hidden)
            attention_hidden = mx.sym.reshape(attention_hidden,
                                              shape=(-1, source_seq_len, self.num_hidden),
                                              name="%scov_reshape_att" % self.prefix)

            # (batch_size, coverage_num_hidden)
            prev_hidden = mx.sym.FullyConnected(data=prev_hidden, weight=self.cov_dec2h_weight, no_bias=True,
                                                num_hidden=self.num_hidden, name="%scov_decoder_hidden")

            # (batch_size, 1, coverage_num_hidden)
            prev_hidden = mx.sym.expand_dims(data=prev_hidden, axis=1,
                                             name="%scov_input_decoder_hidden_expanded" % self.prefix)

            # (batch_size, source_seq_len, coverage_num_hidden)
            intermediate = mx.sym.broadcast_add(lhs=source_hidden, rhs=prev_hidden,
                                                name="%scov_source_plus_hidden" % self.prefix)

            # (batch_size, source_seq_len, coverage_num_hidden)
            updated_coverage = intermediate + attention_hidden + coverage_hidden

            # (batch_size, seq_len, coverage_num_hidden)
            coverage = mx.sym.Activation(data=updated_coverage,
                                         act_type=self.activation,
                                         name="%scov_activation" % self.prefix)

            return mask_coverage(coverage, source_length)

        return update_coverage


def mask_coverage(coverage: mx.sym.Symbol, source_length: mx.sym.Symbol) -> mx.sym.Symbol:
    """
    Masks all coverage scores that are outside the actual sequence.

    :param coverage: Input coverage vector. Shape: (batch_size, seq_len, coverage_num_hidden).
    :param source_length: Source length. Shape: (batch_size,).
    :return: Masked coverage vector. Shape: (batch_size, seq_len, coverage_num_hidden).
    """
    coverage = mx.sym.SwapAxis(data=coverage, dim1=0, dim2=1)
    coverage = mx.sym.SequenceMask(data=coverage, use_sequence_length=True, sequence_length=source_length)
    coverage = mx.sym.SwapAxis(data=coverage, dim1=0, dim2=1)
    return coverage

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

from . import config
from . import constants as C
from . import layers
from . import rnn
from . import utils

logger = logging.getLogger(__name__)


class CoverageConfig(config.Config):
    """
    Coverage configuration.

    :param type: Coverage name.
    :param num_hidden: Number of hidden units for coverage networks.
    :param layer_normalization: Apply layer normalization to coverage networks.
    """
    def __init__(self,
                 type: str,
                 num_hidden: int,
                 layer_normalization: bool) -> None:
        super().__init__()
        self.type = type
        self.num_hidden = num_hidden
        self.layer_normalization = layer_normalization


def get_coverage(config: CoverageConfig) -> 'Coverage':
    """
    Returns a Coverage instance.

    :param config: Coverage configuration.
    :return: Instance of Coverage.
    """
    if config.type == 'count':
        utils.check_condition(config.num_hidden == 1, "Count coverage requires coverage_num_hidden==1")
    if config.type == "gru":
        return GRUCoverage(config.num_hidden, config.layer_normalization)
    elif config.type in {"tanh", "sigmoid", "relu", "softrelu"}:
        return ActivationCoverage(config.num_hidden, config.type, config.layer_normalization)
    elif config.type == "count":
        return CountCoverage()
    else:
        raise ValueError("Unknown coverage type %s" % config.type)


class Coverage:
    """
    Generic coverage class. Similar to Attention classes, a coverage instance returns a callable, update_coverage(),
    function when self.on() is called.
    """
    def __init__(self, prefix=C.COVERAGE_PREFIX):
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
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """
            raise NotImplementedError()

        return update_coverage


class CountCoverage(Coverage):
    """
    Coverage class that accumulates the attention weights for each source word.
    """

    def __init__(self) -> None:
        super().__init__()

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
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len).
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
    :param layer_normalization: If true, applies layer normalization for each gate in the GRU cell.
    """

    def __init__(self, coverage_num_hidden: int, layer_normalization: bool) -> None:
        super().__init__()
        self.num_hidden = coverage_num_hidden
        gru_prefix = "%sgru" % self.prefix
        if layer_normalization:
            self.gru = rnn.LayerNormPerGateGRUCell(self.num_hidden, prefix=gru_prefix)
        else:
            self.gru = mx.rnn.GRUCell(self.num_hidden, prefix=gru_prefix)

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
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """

            # (batch_size, source_seq_len, decoder_num_hidden)
            expanded_decoder = mx.sym.broadcast_axis(
                data=mx.sym.expand_dims(data=prev_hidden, axis=1, name="%sexpand_decoder" % self.prefix),
                axis=1, size=source_seq_len, name="%sbroadcast_decoder" % self.prefix)

            # (batch_size, source_seq_len, 1)
            expanded_att_scores = mx.sym.expand_dims(data=attention_prob_scores,
                                                     axis=2,
                                                     name="%sexpand_attention_scores" % self.prefix)

            # (batch_size, source_seq_len, encoder_num_hidden + decoder_num_hidden + 1)
            # +1 for the attention_prob_score for the source word
            concat_input = mx.sym.concat(source, expanded_decoder, expanded_att_scores, dim=2,
                                         name="%sconcat_inputs" % self.prefix)

            # (batch_size * source_seq_len, encoder_num_hidden + decoder_num_hidden + 1)
            flat_input = mx.sym.reshape(concat_input, shape=(-3, -1), name="%sflatten_inputs")

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
    :param layer_normalization: If true, applies layer normalization before non-linear activation.
    """

    def __init__(self,
                 coverage_num_hidden: int,
                 activation: str,
                 layer_normalization: bool) -> None:
        super().__init__()
        self.activation = activation
        self.num_hidden = coverage_num_hidden
        # input (encoder) to hidden
        self.cov_e2h_weight = mx.sym.Variable("%se2h_weight" % self.prefix)
        # decoder to hidden
        self.cov_dec2h_weight = mx.sym.Variable("%si2h_weight" % self.prefix)
        # previous coverage to hidden
        self.cov_prev2h_weight = mx.sym.Variable("%sprev2h_weight" % self.prefix)
        # attention scores to hidden
        self.cov_a2h_weight = mx.sym.Variable("%sa2h_weight" % self.prefix)
        # optional layer normalization
        self.layer_norm = None
        if layer_normalization and not self.num_hidden != 1:
            self.layer_norm = layers.LayerNormalization(prefix="%snorm" % self.prefix)

    def on(self, source: mx.sym.Symbol, source_length: mx.sym.Symbol, source_seq_len: int) -> Callable:
        """
        Returns callable to be used for updating coverage vectors in a sequence decoder.

        :param source: Shape: (batch_size, seq_len, encoder_num_hidden).
        :param source_length: Shape: (batch_size,).
        :param source_seq_len: Maximum length of source sequences.
        :return: Coverage callable.
        """

        # (batch_size, seq_len, coverage_hidden_num)
        source_hidden = mx.sym.FullyConnected(data=source,
                                              weight=self.cov_e2h_weight,
                                              no_bias=True,
                                              num_hidden=self.num_hidden,
                                              flatten=False,
                                              name="%ssource_hidden_fc" % self.prefix)

        def update_coverage(prev_hidden: mx.sym.Symbol,
                            attention_prob_scores: mx.sym.Symbol,
                            prev_coverage: mx.sym.Symbol):
            """
            :param prev_hidden: Previous hidden decoder state. Shape: (batch_size, decoder_num_hidden).
            :param attention_prob_scores: Current attention scores. Shape: (batch_size, source_seq_len).
            :param prev_coverage: Shape: (batch_size, source_seq_len, coverage_num_hidden).
            :return: Updated coverage matrix . Shape: (batch_size, source_seq_len, coverage_num_hidden).
            """

            # (batch_size, seq_len, coverage_hidden_num)
            coverage_hidden = mx.sym.FullyConnected(data=prev_coverage,
                                                    weight=self.cov_prev2h_weight,
                                                    no_bias=True,
                                                    num_hidden=self.num_hidden,
                                                    flatten=False,
                                                    name="%sprevious_hidden_fc" % self.prefix)

            # (batch_size, source_seq_len, 1)
            attention_prob_scores = mx.sym.expand_dims(attention_prob_scores, axis=2)

            # (batch_size, source_seq_len, coverage_num_hidden)
            attention_hidden = mx.sym.FullyConnected(data=attention_prob_scores,
                                                     weight=self.cov_a2h_weight,
                                                     no_bias=True,
                                                     num_hidden=self.num_hidden,
                                                     flatten=False,
                                                     name="%sattention_fc" % self.prefix)

            # (batch_size, coverage_num_hidden)
            prev_hidden = mx.sym.FullyConnected(data=prev_hidden, weight=self.cov_dec2h_weight, no_bias=True,
                                                num_hidden=self.num_hidden, name="%sdecoder_hidden")

            # (batch_size, 1, coverage_num_hidden)
            prev_hidden = mx.sym.expand_dims(data=prev_hidden, axis=1,
                                             name="%sinput_decoder_hidden_expanded" % self.prefix)

            # (batch_size, source_seq_len, coverage_num_hidden)
            intermediate = mx.sym.broadcast_add(lhs=source_hidden, rhs=prev_hidden,
                                                name="%ssource_plus_hidden" % self.prefix)

            # (batch_size, source_seq_len, coverage_num_hidden)
            updated_coverage = intermediate + attention_hidden + coverage_hidden

            if self.layer_norm is not None:
                updated_coverage = self.layer_norm(data=updated_coverage)

            # (batch_size, seq_len, coverage_num_hidden)
            coverage = mx.sym.Activation(data=updated_coverage,
                                         act_type=self.activation,
                                         name="%sactivation" % self.prefix)

            return mask_coverage(coverage, source_length)

        return update_coverage


def mask_coverage(coverage: mx.sym.Symbol, source_length: mx.sym.Symbol) -> mx.sym.Symbol:
    """
    Masks all coverage scores that are outside the actual sequence.

    :param coverage: Input coverage vector. Shape: (batch_size, seq_len, coverage_num_hidden).
    :param source_length: Source length. Shape: (batch_size,).
    :return: Masked coverage vector. Shape: (batch_size, seq_len, coverage_num_hidden).
    """
    return mx.sym.SequenceMask(data=coverage, axis=1, use_sequence_length=True, sequence_length=source_length)

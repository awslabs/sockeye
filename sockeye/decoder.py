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
Decoders for sequence-to-sequence models.
"""
import logging
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Tuple
from typing import Optional

import mxnet as mx

from sockeye.config import Config
from sockeye.utils import check_condition
from . import rnn_attention
from . import constants as C
from . import encoder
from . import layers
from . import lexicon as lexicons
from . import rnn
from . import convolution
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


def get_decoder(config: Config) -> 'Decoder':
    if isinstance(config, RecurrentDecoderConfig):
        return RecurrentDecoder(config=config, prefix=C.RNN_DECODER_PREFIX)
    elif isinstance(config, ConvolutionalDecoderConfig):
        return ConvolutionalDecoder(config=config, prefix=C.CNN_DECODER_PREFIX)
    elif isinstance(config, transformer.TransformerConfig):
        return TransformerDecoder(config=config, prefix=C.TRANSFORMER_DECODER_PREFIX)
    else:
        raise ValueError("Unsupported decoder configuration")


class Decoder(ABC):
    """
    Generic decoder interface.
    A decoder needs to implement code to decode a target sequence known in advance (decode_sequence),
    and code to decode a single word given its decoder state (decode_step).
    The latter is typically used for inference graphs in beam search.
    For the inference module to be able to keep track of decoder's states
    a decoder provides methods to return initial states (init_states), state variables and their shapes.
    """

    @abstractmethod
    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_embed: mx.sym.Symbol,
                        target_embed_lengths: mx.sym.Symbol,
                        target_embed_max_length: int) -> mx.sym.Symbol:
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Dimension of the embedded target sequence.
        :return: Decoder data. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        pass

    @abstractmethod
    def decode_step(self,
                    target_embed: mx.sym.Symbol,
                    target_embed_lengths: mx.sym.Symbol,
                    target_embed_max_length: int,
                    target_embed_prev: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the embedded target sequence and previous decoder states.
        Returns decoder representation for the next prediction, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Size of embedded target sequence dimension.
        :param target_embed_prev: Previous target word embedding. Shape: (batch_size, target_num_embed).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logit inputs, attention probabilities, next decoder states.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset decoder method. Used for inference.
        """
        pass

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this decoder.
        """
        pass

    @abstractmethod
    def init_states(self,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> List[mx.sym.Symbol]:
        """
        Returns a list of symbolic states that represent the initial states of this decoder.
        Used for inference.

        :param source_encoded: Encoded source. Shape: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :return: List of symbolic initial states.
        """
        pass

    @abstractmethod
    def state_variables(self) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :return: List of symbolic variables.
        """
        pass

    @abstractmethod
    def state_shapes(self,
                     batch_size: int,
                     source_encoded_max_length: int,
                     source_encoded_depth: int) -> List[mx.io.DataDesc]:
        """
        Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
        Used for inference.

        :param batch_size: Batch size during inference.
        :param source_encoded_max_length: Size of encoder time dimension.
        :param source_encoded_depth: Depth of encoded source.
        :return: List of shape descriptions.
        """
        pass

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the decoder if such a restriction exists.
        """
        return None


class TransformerDecoder(Decoder):
    """
    Transformer decoder as in Vaswani et al, 2017: Attention is all you need.
    In training, computation scores for each position of the known target sequence are compouted in parallel,
    yielding most of the speedup.
    At inference time, the decoder block is evaluated again and again over a maximum length input sequence that is
    initially filled with zeros and grows during beam search with predicted tokens. Appropriate masking at every
    time-step ensures correct self-attention scores and is updated with every step.

    :param config: Transformer configuration.
    :param prefix: Name prefix for symbols of this decoder.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 prefix: str = C.TRANSFORMER_DECODER_PREFIX) -> None:
        self.config = config
        self.prefix = prefix
        self.layers = [transformer.TransformerDecoderBlock(
            config, prefix="%s%d_" % (prefix, i)) for i in range(config.num_layers)]
        self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                 num_hidden=config.model_size,
                                                                 dropout=config.dropout_prepost,
                                                                 prefix="%sfinal_process_" % prefix)

        self.pos_embedding = encoder.get_positional_embedding(config.positional_embedding_type,
                                                              config.model_size,
                                                              max_seq_len=config.max_seq_len_target,
                                                              fixed_scale_input=True,
                                                              fixed_scale_positions=False,
                                                              prefix=C.TARGET_POSITIONAL_EMBEDDING_PREFIX)

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_embed: mx.sym.Symbol,
                        target_embed_lengths: mx.sym.Symbol,
                        target_embed_max_length: int) -> mx.sym.Symbol:
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Dimension of the embedded target sequence.
        :return: Decoder data. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        # (batch_size, source_max_length, num_source_embed)
        source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

        # (batch_size, target_max_length, model_size)
        target = self._decode(source_encoded, source_encoded_lengths, source_encoded_max_length,
                              target_embed, target_embed_lengths, target_embed_max_length)

        return target

    def _decode(self,
                source_encoded, source_encoded_lengths, source_encoded_max_length,
                target_embed, target_embed_lengths, target_embed_max_length):
        """
        Runs stacked decoder transformer blocks.

        :param source_encoded: Batch-major encoded source: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Size of embedded target sequence dimension.
        :return: Result of stacked transformer blocks.
        """

        # (1, target_max_length, target_max_length)
        target_bias = transformer.get_autoregressive_bias(target_embed_max_length, name="%sbias" % self.prefix)

        # target: (batch_size, target_max_length, model_size)
        target, target_lengths, target_max_length = self.pos_embedding.encode(target_embed,
                                                                              target_embed_lengths,
                                                                              target_embed_max_length)

        if self.config.dropout_prepost > 0.0:
            target = mx.sym.Dropout(data=target, p=self.config.dropout_prepost)

        for layer in self.layers:
            target = layer(target, target_lengths, target_max_length, target_bias,
                           source_encoded, source_encoded_lengths, source_encoded_max_length)
        target = self.final_process(data=target, prev=None)

        return target

    def decode_step(self,
                    target_embed: mx.sym.Symbol,
                    target_embed_lengths: mx.sym.Symbol,
                    target_embed_max_length: int,
                    target_embed_prev: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the embedded target sequence and previous decoder states.
        Returns decoder representation for the next prediction, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Size of embedded target sequence dimension.
        :param target_embed_prev: Previous target word embedding. Shape: (batch_size, target_num_embed).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logit inputs, attention probabilities, next decoder states.
        """
        source_encoded, source_encoded_lengths = states

        # indices: (batch_size,)
        indices = target_embed_lengths - 1  # type: mx.sym.Symbol

        # (batch_size, target_max_length, 1)
        mask = mx.sym.expand_dims(mx.sym.one_hot(indices=indices,
                                                 depth=target_embed_max_length,
                                                 on_value=1, off_value=0), axis=2)

        # (batch_size, target_max_length, model_size)
        target = self._decode(source_encoded, source_encoded_lengths, source_encoded_max_length,
                              target_embed, target_embed_lengths, target_embed_max_length)

        # set all target positions to zero except for current time-step
        # target: (batch_size, target_max_length, model_size)
        target = mx.sym.broadcast_mul(target, mask)
        # reduce to single prediction
        # target: (batch_size, model_size)
        target = mx.sym.sum(target, axis=1, keepdims=False)

        # TODO(fhieber): no attention probs for now
        attention_probs = mx.sym.sum(mx.sym.zeros_like(source_encoded), axis=2, keepdims=False)

        new_states = [source_encoded, source_encoded_lengths]
        return target, attention_probs, new_states

    def reset(self):
        pass

    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this decoder.
        """
        return self.config.model_size

    def init_states(self,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> List[mx.sym.Symbol]:
        """
        Returns a list of symbolic states that represent the initial states of this decoder.
        Used for inference.

        :param source_encoded: Encoded source. Shape: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :return: List of symbolic initial states.
        """
        return [source_encoded, source_encoded_lengths]

    def state_variables(self) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :return: List of symbolic variables.
        """
        return [mx.sym.Variable(C.SOURCE_ENCODED_NAME),
                mx.sym.Variable(C.SOURCE_LENGTH_NAME)]

    def state_shapes(self,
                     batch_size: int,
                     source_encoded_max_length: int,
                     source_encoded_depth: int) -> List[mx.io.DataDesc]:
        """
        Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
        Used for inference.

        :param batch_size: Batch size during inference.
        :param source_encoded_max_length: Size of encoder time dimension.
        :param source_encoded_depth: Depth of encoded source.
        :return: List of shape descriptions.
        """
        return [mx.io.DataDesc(C.SOURCE_ENCODED_NAME,
                               (batch_size, source_encoded_max_length, source_encoded_depth),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(C.SOURCE_LENGTH_NAME, (batch_size,), layout="N")]

    def get_max_seq_len(self) -> Optional[int]:
        #  The positional embeddings potentially pose a limit on the maximum length at inference time.
        return self.pos_embedding.get_max_seq_len()


RecurrentDecoderState = NamedTuple('RecurrentDecoderState', [
    ('hidden', mx.sym.Symbol),
    ('layer_states', List[mx.sym.Symbol]),
])
"""
RecurrentDecoder state.

:param hidden: Hidden state after attention mechanism. Shape: (batch_size, num_hidden).
:param layer_states: Hidden states for RNN layers of RecurrentDecoder. Shape: List[(batch_size, rnn_num_hidden)]
"""


class RecurrentDecoderConfig(Config):
    """
    Recurrent decoder configuration.

    :param max_seq_len_source: Maximum source sequence length
    :param rnn_config: RNN configuration.
    :param attention_config: Attention configuration.
    :param hidden_dropout: Dropout probability on next decoder hidden state.
    :param state_init: Type of RNN decoder state initialization: zero, last, average.
    :param context_gating: Whether to use context gating.
    :param layer_normalization: Apply layer normalization.
    :param attention_in_upper_layers: Pass the attention value to all layers in the decoder.
    """

    def __init__(self,
                 max_seq_len_source: int,
                 rnn_config: rnn.RNNConfig,
                 attention_config: rnn_attention.AttentionConfig,
                 hidden_dropout: float = .0,  # TODO: move this dropout functionality to OutputLayer
                 state_init: str = C.RNN_DEC_INIT_LAST,
                 context_gating: bool = False,
                 layer_normalization: bool = False,
                 attention_in_upper_layers: bool = False) -> None:
        super().__init__()
        self.max_seq_len_source = max_seq_len_source
        self.rnn_config = rnn_config
        self.attention_config = attention_config
        self.hidden_dropout = hidden_dropout
        self.state_init = state_init
        self.context_gating = context_gating
        self.layer_normalization = layer_normalization
        self.attention_in_upper_layers = attention_in_upper_layers


class RecurrentDecoder(Decoder):
    """
    RNN Decoder with attention.
    The architecture is based on Luong et al, 2015: Effective Approaches to Attention-based Neural Machine Translation.

    :param config: Configuration for recurrent decoder.
    :param prefix: Decoder symbol prefix.
    """

    def __init__(self,
                 config: RecurrentDecoderConfig,
                 prefix: str = C.RNN_DECODER_PREFIX) -> None:
        # TODO: implement variant without input feeding
        self.config = config
        self.rnn_config = config.rnn_config
        self.attention = rnn_attention.get_attention(config.attention_config, config.max_seq_len_source)
        self.prefix = prefix

        self.num_hidden = self.rnn_config.num_hidden

        if self.config.context_gating:
            utils.check_condition(not self.config.attention_in_upper_layers,
                                  "Context gating is not supported with attention in upper layers.")
            self.gate_w = mx.sym.Variable("%sgate_weight" % prefix)
            self.gate_b = mx.sym.Variable("%sgate_bias" % prefix)
            self.mapped_rnn_output_w = mx.sym.Variable("%smapped_rnn_output_weight" % prefix)
            self.mapped_rnn_output_b = mx.sym.Variable("%smapped_rnn_output_bias" % prefix)
            self.mapped_context_w = mx.sym.Variable("%smapped_context_weight" % prefix)
            self.mapped_context_b = mx.sym.Variable("%smapped_context_bias" % prefix)
        if self.rnn_config.residual:
            utils.check_condition(self.config.rnn_config.first_residual_layer >= 2,
                                  "Residual connections on the first decoder layer are not supported as input and "
                                  "output dimensions do not match.")

        # Stacked RNN
        if self.rnn_config.num_layers == 1 or not self.rnn_config.attention_in_upper_layers:
            self.rnn_pre_attention = rnn.get_stacked_rnn(self.rnn_config, self.prefix, parallel_inputs=False)
            self.rnn_post_attention = None
        else:
            self.rnn_pre_attention = rnn.get_stacked_rnn(self.rnn_config, self.prefix, parallel_inputs=False,
                                                         layers=[0])
            self.rnn_post_attention = rnn.get_stacked_rnn(self.rnn_config, self.prefix, parallel_inputs=True,
                                                          layers=range(1, self.rnn_config.num_layers))
        self.rnn_pre_attention_n_states = len(self.rnn_pre_attention.state_shape)

        if self.config.state_init != C.RNN_DEC_INIT_ZERO:
            self._create_state_init_parameters()

        # Hidden state parameters
        self.hidden_w = mx.sym.Variable("%shidden_weight" % prefix)
        self.hidden_b = mx.sym.Variable("%shidden_bias" % prefix)
        self.hidden_norm = layers.LayerNormalization(self.num_hidden,
                                                     prefix="%shidden_norm" % prefix) \
            if self.config.layer_normalization else None

    def _create_state_init_parameters(self):
        """
        Creates parameters for encoder last state transformation into decoder layer initial states.
        """
        self.init_ws, self.init_bs, self.init_norms = [], [], []
        state_shapes = self.rnn_pre_attention.state_shape
        if self.rnn_post_attention:
            state_shapes += self.rnn_post_attention.state_shape
        for state_idx, (_, init_num_hidden) in enumerate(state_shapes):
            self.init_ws.append(mx.sym.Variable("%senc2decinit_%d_weight" % (self.prefix, state_idx)))
            self.init_bs.append(mx.sym.Variable("%senc2decinit_%d_bias" % (self.prefix, state_idx)))
            if self.config.layer_normalization:
                self.init_norms.append(layers.LayerNormalization(num_hidden=init_num_hidden,
                                                                 prefix="%senc2decinit_%d_norm" % (
                                                                     self.prefix, state_idx)))

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_embed: mx.sym.Symbol,
                        target_embed_lengths: mx.sym.Symbol,
                        target_embed_max_length: int) -> mx.sym.Symbol:
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Dimension of the embedded target sequence.
        :return: Decoder data. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        # target_embed: target_seq_len * (batch_size, num_target_embed)
        target_embed = mx.sym.split(data=target_embed, num_outputs=target_embed_max_length, axis=1, squeeze_axis=True)

        # get recurrent attention function conditioned on source
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')
        attention_func = self.attention.on(source_encoded_batch_major, source_encoded_lengths,
                                           source_encoded_max_length)
        attention_state = self.attention.get_initial_state(source_encoded_lengths, source_encoded_max_length)

        # initialize decoder states
        # hidden: (batch_size, rnn_num_hidden)
        # layer_states: List[(batch_size, state_num_hidden]
        state = self.get_initial_state(source_encoded, source_encoded_lengths)

        # hidden_all: target_seq_len * (batch_size, 1, rnn_num_hidden)
        hidden_all = []
        # TODO: possible alternative: feed back the context vector instead of the hidden (see lamtram)
        self.reset()
        for seq_idx in range(target_embed_max_length):
            # hidden: (batch_size, rnn_num_hidden)
            state, attention_state = self._step(target_embed[seq_idx],
                                                state,
                                                attention_func,
                                                attention_state,
                                                seq_idx)
            # hidden_expanded: (batch_size, 1, rnn_num_hidden)
            hidden_all.append(mx.sym.expand_dims(data=state.hidden, axis=1))

        # concatenate along time axis
        # hidden_concat: (batch_size, target_seq_len, rnn_num_hidden)
        hidden_concat = mx.sym.concat(*hidden_all, dim=1, name="%shidden_concat" % self.prefix)
        return hidden_concat

    def decode_step(self,
                    target_embed: mx.sym.Symbol,
                    target_embed_lengths: mx.sym.Symbol,
                    target_embed_max_length: int,
                    target_embed_prev: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the embedded target sequence and previous decoder states.
        Returns decoder representation for the next prediction, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Size of embedded target sequence dimension.
        :param target_embed_prev: Previous target word embedding. Shape: (batch_size, target_num_embed).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logit inputs, attention probabilities, next decoder states.
        """
        source_encoded, prev_dynamic_source, source_encoded_length, prev_hidden, *layer_states = states

        attention_func = self.attention.on(source_encoded, source_encoded_length, source_encoded_max_length)

        prev_state = RecurrentDecoderState(prev_hidden, list(layer_states))
        prev_attention_state = rnn_attention.AttentionState(context=None, probs=None,
                                                            dynamic_source=prev_dynamic_source)

        # state.hidden: (batch_size, rnn_num_hidden)
        # attention_state.dynamic_source: (batch_size, source_seq_len, coverage_num_hidden)
        # attention_state.probs: (batch_size, source_seq_len)
        state, attention_state = self._step(target_embed_prev,
                                            prev_state,
                                            attention_func,
                                            prev_attention_state)

        new_states = [source_encoded,
                      attention_state.dynamic_source,
                      source_encoded_length,
                      state.hidden] + state.layer_states

        return state.hidden, attention_state.probs, new_states

    def reset(self):
        """
        Calls reset on the RNN cell.
        """
        self.rnn_pre_attention.reset()
        cells_to_reset = self.rnn_pre_attention._cells
        if self.rnn_post_attention:
            self.rnn_post_attention.reset()
            cells_to_reset += self.rnn_post_attention._cells
        for cell in cells_to_reset:
            # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
            if isinstance(cell, mx.rnn.ModifierCell):
                cell.base_cell.reset()
            cell.reset()

    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this decoder.
        """
        return self.num_hidden

    def init_states(self,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> List[mx.sym.Symbol]:
        """
        Returns a list of symbolic states that represent the initial states of this decoder.
        Used for inference.

        :param source_encoded: Encoded source. Shape: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :return: List of symbolic initial states.
        """
        source_encoded_time_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)
        hidden, layer_states = self.get_initial_state(source_encoded_time_major, source_encoded_lengths)
        context, attention_probs, dynamic_source = self.attention.get_initial_state(source_encoded_lengths,
                                                                                    source_encoded_max_length)
        states = [source_encoded, dynamic_source, source_encoded_lengths, hidden] + layer_states
        return states

    def state_variables(self) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :return: List of symbolic variables.
        """
        return [mx.sym.Variable(C.SOURCE_ENCODED_NAME),
                mx.sym.Variable(C.SOURCE_DYNAMIC_PREVIOUS_NAME),
                mx.sym.Variable(C.SOURCE_LENGTH_NAME),
                mx.sym.Variable(C.HIDDEN_PREVIOUS_NAME)] + \
               [mx.sym.Variable("%senc2decinit_%d" % (self.prefix, i)) for i in
                range(len(sum([rnn.state_info for rnn in self.get_rnn_cells()], [])))]

    def state_shapes(self,
                     batch_size: int,
                     source_encoded_max_length: int,
                     source_encoded_depth: int) -> List[mx.io.DataDesc]:
        """
        Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
        Used for inference.

        :param batch_size: Batch size during inference.
        :param source_encoded_max_length: Size of encoder time dimension.
        :param source_encoded_depth: Depth of encoded source.
        :return: List of shape descriptions.
        """
        return [mx.io.DataDesc(C.SOURCE_ENCODED_NAME,
                               (batch_size, source_encoded_max_length, source_encoded_depth),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(C.SOURCE_DYNAMIC_PREVIOUS_NAME,
                               (batch_size, source_encoded_max_length, self.attention.dynamic_source_num_hidden),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(C.SOURCE_LENGTH_NAME,
                               (batch_size,),
                               layout="N"),
                mx.io.DataDesc(C.HIDDEN_PREVIOUS_NAME,
                               (batch_size, self.num_hidden),
                               layout="NC")] + \
               [mx.io.DataDesc("%senc2decinit_%d" % (self.prefix, i),
                               (batch_size, num_hidden),
                               layout=C.BATCH_MAJOR) for i, (_, num_hidden) in enumerate(
                   sum([rnn.state_shape for rnn in self.get_rnn_cells()], [])
               )]

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.
        """
        cells = [self.rnn_pre_attention]
        if self.rnn_post_attention:
            cells.append(self.rnn_post_attention)
        return cells

    def get_initial_state(self,
                          source_encoded: mx.sym.Symbol,
                          source_encoded_length: mx.sym.Symbol) -> RecurrentDecoderState:
        """
        Computes initial states of the decoder, hidden state, and one for each RNN layer.
        Optionally, init states for RNN layers are computed using 1 non-linear FC
        with the last state of the encoder as input.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_encoded_length: Lengths of source sequences. Shape: (batch_size,).
        :return: Decoder state.
        """
        # we derive the shape of hidden and layer_states from some input to enable
        # shape inference for the batch dimension during inference.
        # (batch_size, 1)
        zeros = mx.sym.expand_dims(mx.sym.zeros_like(source_encoded_length), axis=1)
        # last encoder state: (batch, num_hidden)
        source_encoded_last = mx.sym.SequenceLast(data=source_encoded,
                                                  sequence_length=source_encoded_length,
                                                  use_sequence_length=True) \
            if self.config.state_init == C.RNN_DEC_INIT_LAST else None
        source_masked = mx.sym.SequenceMask(data=source_encoded,
                                            sequence_length=source_encoded_length,
                                            use_sequence_length=True,
                                            value=0.) if self.config.state_init == C.RNN_DEC_INIT_AVG else None

        # decoder hidden state
        hidden = mx.sym.tile(data=zeros, reps=(1, self.num_hidden))

        # initial states for each layer
        layer_states = []
        for state_idx, (_, init_num_hidden) in enumerate(sum([rnn.state_shape for rnn in self.get_rnn_cells()], [])):
            if self.config.state_init == C.RNN_DEC_INIT_ZERO:
                init = mx.sym.tile(data=zeros, reps=(1, init_num_hidden))
            else:
                if self.config.state_init == C.RNN_DEC_INIT_LAST:
                    init = source_encoded_last
                elif self.config.state_init == C.RNN_DEC_INIT_AVG:
                    # (batch_size, encoder_num_hidden)
                    init = mx.sym.broadcast_div(mx.sym.sum(source_masked, axis=0, keepdims=False),
                                                mx.sym.expand_dims(source_encoded_length, axis=1))
                else:
                    raise ValueError("Unknown decoder state init type '%s'" % self.config.state_init)

                init = mx.sym.FullyConnected(data=init,
                                             num_hidden=init_num_hidden,
                                             weight=self.init_ws[state_idx],
                                             bias=self.init_bs[state_idx],
                                             name="%senc2decinit_%d" % (self.prefix, state_idx))
                if self.config.layer_normalization:
                    init = self.init_norms[state_idx].normalize(init)
                init = mx.sym.Activation(data=init, act_type="tanh",
                                         name="%senc2dec_inittanh_%d" % (self.prefix, state_idx))
            layer_states.append(init)

        return RecurrentDecoderState(hidden, layer_states)

    def _step(self, word_vec_prev: mx.sym.Symbol,
              state: RecurrentDecoderState,
              attention_func: Callable,
              attention_state: rnn_attention.AttentionState,
              seq_idx: int = 0) -> Tuple[RecurrentDecoderState, rnn_attention.AttentionState]:

        """
        Performs single-time step in the RNN, given previous word vector, previous hidden state, attention function,
        and RNN layer states.

        :param word_vec_prev: Embedding of previous target word. Shape: (batch_size, num_target_embed).
        :param state: Decoder state consisting of hidden and layer states.
        :param attention_func: Attention function to produce context vector.
        :param attention_state: Previous attention state.
        :param seq_idx: Decoder time step.
        :return: (new decoder state, updated attention state).
        """
        # (1) RNN step
        # concat previous word embedding and previous hidden state
        rnn_input = mx.sym.concat(word_vec_prev, state.hidden, dim=1,
                                  name="%sconcat_target_context_t%d" % (self.prefix, seq_idx))
        # rnn_pre_attention_output: (batch_size, rnn_num_hidden)
        # next_layer_states: num_layers * [batch_size, rnn_num_hidden]
        rnn_pre_attention_output, rnn_pre_attention_layer_states = \
            self.rnn_pre_attention(rnn_input, state.layer_states[:self.rnn_pre_attention_n_states])

        # (2) Attention step
        attention_input = self.attention.make_input(seq_idx, word_vec_prev, rnn_pre_attention_output)
        attention_state = attention_func(attention_input, attention_state)

        # (3) Attention handling (and possibly context gating)
        if self.rnn_post_attention:
            upper_rnn_output, upper_rnn_layer_states = \
                self.rnn_post_attention(rnn_pre_attention_output, attention_state.context,
                                        state.layer_states[self.rnn_pre_attention_n_states:])
            hidden_concat = mx.sym.concat(upper_rnn_output, attention_state.context,
                                          dim=1, name='%shidden_concat_t%d' % (self.prefix, seq_idx))
            if self.config.hidden_dropout > 0:
                hidden_concat = mx.sym.Dropout(data=hidden_concat, p=self.config.hidden_dropout,
                                               name='%shidden_concat_dropout_t%d' % (self.prefix, seq_idx))
            hidden = self._hidden_mlp(hidden_concat, seq_idx)
            # TODO: add context gating?
        else:
            upper_rnn_layer_states = []
            hidden_concat = mx.sym.concat(rnn_pre_attention_output, attention_state.context,
                                          dim=1, name='%shidden_concat_t%d' % (self.prefix, seq_idx))
            if self.config.hidden_dropout > 0:
                hidden_concat = mx.sym.Dropout(data=hidden_concat, p=self.config.hidden_dropout,
                                               name='%shidden_concat_dropout_t%d' % (self.prefix, seq_idx))

            if self.config.context_gating:
                hidden = self._context_gate(hidden_concat, rnn_pre_attention_output, attention_state, seq_idx)
            else:
                hidden = self._hidden_mlp(hidden_concat, seq_idx)

        return RecurrentDecoderState(hidden, rnn_pre_attention_layer_states + upper_rnn_layer_states), attention_state

    def _hidden_mlp(self, hidden_concat: mx.sym.Symbol, seq_idx: int) -> mx.sym.Symbol:
        hidden = mx.sym.FullyConnected(data=hidden_concat,
                                       num_hidden=self.num_hidden,  # to state size of RNN
                                       weight=self.hidden_w,
                                       bias=self.hidden_b,
                                       name='%shidden_fc_t%d' % (self.prefix, seq_idx))
        if self.config.layer_normalization:
            hidden = self.hidden_norm.normalize(hidden)

        # hidden: (batch_size, rnn_num_hidden)
        hidden = mx.sym.Activation(data=hidden, act_type="tanh",
                                   name="%snext_hidden_t%d" % (self.prefix, seq_idx))
        return hidden

    def _context_gate(self,
                      hidden_concat: mx.sym.Symbol,
                      rnn_output: mx.sym.Symbol,
                      attention_state: rnn_attention.AttentionState,
                      seq_idx: int) -> mx.sym.Symbol:
        gate = mx.sym.FullyConnected(data=hidden_concat,
                                     num_hidden=self.num_hidden,
                                     weight=self.gate_w,
                                     bias=self.gate_b,
                                     name='%shidden_gate_t%d' % (self.prefix, seq_idx))
        gate = mx.sym.Activation(data=gate, act_type="sigmoid",
                                 name='%shidden_gate_act_t%d' % (self.prefix, seq_idx))

        mapped_rnn_output = mx.sym.FullyConnected(data=rnn_output,
                                                  num_hidden=self.num_hidden,
                                                  weight=self.mapped_rnn_output_w,
                                                  bias=self.mapped_rnn_output_b,
                                                  name="%smapped_rnn_output_fc_t%d" % (self.prefix, seq_idx))
        mapped_context = mx.sym.FullyConnected(data=attention_state.context,
                                               num_hidden=self.num_hidden,
                                               weight=self.mapped_context_w,
                                               bias=self.mapped_context_b,
                                               name="%smapped_context_fc_t%d" % (self.prefix, seq_idx))

        hidden = gate * mapped_rnn_output + (1 - gate) * mapped_context

        if self.config.layer_normalization:
            hidden = self.hidden_norm.normalize(hidden)

        # hidden: (batch_size, rnn_num_hidden)
        hidden = mx.sym.Activation(data=hidden, act_type="tanh",
                                   name="%snext_hidden_t%d" % (self.prefix, seq_idx))
        return hidden


class ConvolutionalDecoderConfig(Config):
    """
    Convolutional decoder configuration.

    :param cnn_config: Configuration for the convolution block.
    :param max_seq_len_target: Maximum target sequence length.
    :param num_embed: Target word embedding size.
    :param encoder_num_hidden: Number of hidden units of the encoder.
    :param num_layers: The number of convolutional layers.
    :param positional_embedding_type: The type of positional embedding.
    :param hidden_dropout: Dropout probability on next decoder hidden state.
    """

    def __init__(self,
                 cnn_config: convolution.ConvolutionConfig,
                 max_seq_len_target: int,
                 num_embed: int,
                 encoder_num_hidden: int,
                 num_layers: int,
                 positional_embedding_type: str,
                 hidden_dropout: float = .0) -> None:
        super().__init__()
        self.cnn_config = cnn_config
        self.max_seq_len_target = max_seq_len_target
        self.num_embed = num_embed
        self.encoder_num_hidden = encoder_num_hidden
        self.num_layers = num_layers
        self.positional_embedding_type = positional_embedding_type
        self.hidden_dropout = hidden_dropout


class ConvolutionalDecoder(Decoder):
    """
    Convolutional decoder similar to Gehring et al. 2017.

    The decoder consists of an embedding layer, positional embeddings, and layers
    of convolutional blocks with residual connections.

    Notable differences to Gehring et al. 2017:
     * Here the context vectors are created from the last encoder state (instead of using the last encoder state as the
       key and the sum of the encoder state and the source embedding as the value)
     * The encoder gradients are not scaled down by 1/(2 * num_attention_layers).
     * Residual connections are not scaled down by math.sqrt(0.5).
     * Attention is computed in the hidden dimension instead of the embedding dimension (removes need for training
       several projection matrices)

    :param config: Configuration for convolutional decoder.
    :param prefix: Name prefix for symbols of this decoder.
    """

    def __init__(self,
                 config: ConvolutionalDecoderConfig,
                 prefix: str = C.DECODER_PREFIX) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix

        # TODO: potentially project the encoder hidden size to the decoder hidden size.
        utils.check_condition(config.encoder_num_hidden == config.cnn_config.num_hidden,
                              "We need to have the same number of hidden units in the decoder "
                              "as we have in the encoder")

        self.pos_embedding = encoder.get_positional_embedding(config.positional_embedding_type,
                                                              num_embed=config.num_embed,
                                                              max_seq_len=config.max_seq_len_target,
                                                              fixed_scale_input=False,
                                                              fixed_scale_positions=False,
                                                              prefix=C.TARGET_POSITIONAL_EMBEDDING_PREFIX)

        self.layers = [convolution.ConvolutionBlock(
            config.cnn_config,
            pad_type='left',
            prefix="%s%d_" % (prefix, i)) for i in range(config.num_layers)]

        self.i2h_weight = mx.sym.Variable('%si2h_weight' % prefix)

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_embed: mx.sym.Symbol,
                        target_embed_lengths: mx.sym.Symbol,
                        target_embed_max_length: int) -> mx.sym.Symbol:
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Dimension of the embedded target sequence.
        :return: Decoder data. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        # (batch_size, source_encoded_max_length, encoder_depth).
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')

        # (batch_size, target_seq_len, num_hidden)
        target_hidden = self._decode(source_encoded=source_encoded_batch_major,
                                     source_encoded_lengths=source_encoded_lengths,
                                     target_embed=target_embed,
                                     target_embed_lengths=target_embed_lengths,
                                     target_embed_max_length=target_embed_max_length)

        return target_hidden

    def _decode(self,
                source_encoded: mx.sym.Symbol,
                source_encoded_lengths: mx.sym.Symbol,
                target_embed: mx.sym.Symbol,
                target_embed_lengths: mx.sym.Symbol,
                target_embed_max_length: int) -> mx.sym.Symbol:
        """
        Decode the target and produce a sequence of hidden states.

        :param source_encoded:  Shape: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Shape: (batch_size,).
        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Size of embedded target sequence dimension.
        :return: The target hidden states. Shape: (batch_size, target_seq_len, num_hidden).
        """
        target_embed, target_embed_lengths, target_embed_max_length = self.pos_embedding.encode(target_embed,
                                                                                                target_embed_lengths,
                                                                                                target_embed_max_length)
        # target_hidden: (batch_size, target_seq_len, num_hidden)
        target_hidden = mx.sym.FullyConnected(data=target_embed,
                                              num_hidden=self.config.cnn_config.num_hidden,
                                              no_bias=True,
                                              flatten=False,
                                              weight=self.i2h_weight)
        target_hidden_prev = target_hidden

        drop_prob = self.config.hidden_dropout

        for layer in self.layers:
            # (batch_size, target_seq_len, num_hidden)
            target_hidden = layer(mx.sym.Dropout(target_hidden, p=drop_prob) if drop_prob > 0 else target_hidden,
                                  target_embed_lengths, target_embed_max_length)

            # (batch_size, target_seq_len, num_embed)
            context = layers.dot_attention(queries=target_hidden,
                                           keys=source_encoded, values=source_encoded,
                                           length=source_encoded_lengths)

            # residual connection:
            target_hidden = target_hidden_prev + target_hidden + context
            target_hidden_prev = target_hidden

        return target_hidden

    def decode_step(self,
                    target_embed: mx.sym.Symbol,
                    target_embed_lengths: mx.sym.Symbol,
                    target_embed_max_length: int,
                    target_embed_prev: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the embedded target sequence and previous decoder states.
        Returns decoder representation for the next prediction, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param target_embed: Embedded target sequence. Shape: (batch_size, target_embed_max_length, target_num_embed).
        :param target_embed_lengths: Lengths of embedded target sequences. Shape: (batch_size,).
        :param target_embed_max_length: Size of embedded target sequence dimension.
        :param target_embed_prev: Previous target word embedding. Shape: (batch_size, target_num_embed).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logit inputs, attention probabilities, next decoder states.
        """
        indices = target_embed_lengths - 1  # type: mx.sym.Symbol

        # Source_encoded: (batch_size, source_encoded_max_length, encoder_depth)
        source_encoded, source_encoded_lengths, *layer_states = states

        # The last layer doesn't keep any state as we only need the last hidden vector for the next word prediction
        # but none of the previous hidden vectors
        last_layer_state = None
        embed_layer_state = layer_states[0]
        cnn_layer_states = list(layer_states[1:]) + [last_layer_state]

        kernel_width = self.config.cnn_config.kernel_width

        new_layer_states = []

        # (batch_size, num_embed)
        target_embed_prev = self.pos_embedding.encode_positions(indices, target_embed_prev)

        # (batch_size, num_hidden)
        target_hidden_step = mx.sym.FullyConnected(data=target_embed_prev,
                                                   num_hidden=self.config.cnn_config.num_hidden,
                                                   no_bias=True,
                                                   weight=self.i2h_weight)
        # re-arrange outcoming layer to the dimensions of the output
        # (batch_size, 1, num_hidden)
        target_hidden_step = mx.sym.expand_dims(target_hidden_step, axis=1)
        # (batch_size, kernel_width, num_hidden)
        target_hidden = mx.sym.concat(embed_layer_state, target_hidden_step, dim=1)

        new_layer_states.append(mx.sym.slice_axis(data=target_hidden, axis=1, begin=1, end=kernel_width))

        target_hidden_step_prev = target_hidden_step

        drop_prob = self.config.hidden_dropout

        for layer, layer_state in zip(self.layers, cnn_layer_states):
            # (batch_size, kernel_width, num_hidden) -> (batch_size, 1, num_hidden)
            target_hidden_step = layer.step(mx.sym.Dropout(target_hidden, p=drop_prob)
                                            if drop_prob > 0 else target_hidden)

            # (batch_size, 1, num_embed)
            context_step = layers.dot_attention(queries=target_hidden_step,
                                                keys=source_encoded, values=source_encoded,
                                                length=source_encoded_lengths)
            # residual connection:
            target_hidden_step = target_hidden_step_prev + target_hidden_step + context_step
            target_hidden_step_prev = target_hidden_step

            if layer_state is not None:
                # combine with layer state
                # (batch_size, kernel_width, num_hidden)
                target_hidden = mx.sym.concat(layer_state, target_hidden_step, dim=1)

                new_layer_states.append(mx.sym.slice_axis(data=target_hidden, axis=1, begin=1, end=kernel_width))

            else:
                # last state, here we only care about the latest hidden state:
                # (batch_size, 1, num_hidden) -> (batch_size, num_hidden)
                target_hidden = mx.sym.reshape(target_hidden_step, shape=(-3, -1))

        # (batch_size, source_encoded_max_length)
        attention_probs = mx.sym.reshape(mx.sym.slice_axis(mx.sym.zeros_like(source_encoded),
                                                           axis=2, begin=0, end=1),
                                         shape=(0, -1))

        return target_hidden, attention_probs, [source_encoded, source_encoded_lengths] + new_layer_states

    def reset(self):
        pass

    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this decoder.
        """
        return self.config.cnn_config.num_hidden

    def init_states(self,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> List[mx.sym.Symbol]:
        """
        Returns a list of symbolic states that represent the initial states of this decoder.
        Used for inference.

        :param source_encoded: Encoded source. Shape: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :return: List of symbolic initial states.
        """
        # Initially all layers get pad symbols as input (zeros)
        # (batch_size, kernel_width, num_hidden)
        num_hidden = self.config.cnn_config.num_hidden
        kernel_width = self.config.cnn_config.kernel_width
        # Note: We can not use mx.sym.zeros, as otherwise shape inference fails.
        # Therefore we need to get a zero array of the right size through other means.
        # (batch_size, 1, 1)
        zeros = mx.sym.expand_dims(mx.sym.expand_dims(mx.sym.zeros_like(source_encoded_lengths), axis=1), axis=2)
        # (batch_size, kernel_width-1, num_hidden)
        next_layer_inputs = [mx.sym.tile(data=zeros, reps=(1, kernel_width - 1, num_hidden),
                                         name="%s%d_init" % (self.prefix, layer_idx))
                             for layer_idx in range(0, self.config.num_layers)]
        return [source_encoded, source_encoded_lengths] + next_layer_inputs

    def state_variables(self) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :return: List of symbolic variables.
        """
        # we keep a fixed slice of the layer inputs as a state for all upper layers:
        next_layer_inputs = [mx.sym.Variable("cnn_layer%d_in" % layer_idx)
                             for layer_idx in range(0, self.config.num_layers)]
        return [mx.sym.Variable(C.SOURCE_ENCODED_NAME),
                mx.sym.Variable(C.SOURCE_LENGTH_NAME)] + next_layer_inputs

    def state_shapes(self,
                     batch_size: int,
                     source_encoded_max_length: int,
                     source_encoded_depth: int) -> List[mx.io.DataDesc]:
        """
        Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
        Used for inference.

        :param batch_size: Batch size during inference.
        :param source_encoded_max_length: Size of encoder time dimension.
        :param source_encoded_depth: Depth of encoded source.
        :return: List of shape descriptions.
        """
        num_hidden = self.config.cnn_config.num_hidden
        kernel_width = self.config.cnn_config.kernel_width
        next_layer_inputs = [mx.io.DataDesc("cnn_layer%d_in" % layer_idx,
                                            shape=(batch_size, kernel_width - 1, num_hidden),
                                            layout="NTW")
                             for layer_idx in range(0, self.config.num_layers)]
        return [mx.io.DataDesc(C.SOURCE_ENCODED_NAME,
                               (batch_size, source_encoded_max_length, source_encoded_depth),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(C.SOURCE_LENGTH_NAME, (batch_size,), layout="N")] + next_layer_inputs

    def get_max_seq_len(self) -> Optional[int]:
        #  The positional embeddings potentially pose a limit on the maximum length at inference time.
        return self.pos_embedding.get_max_seq_len()

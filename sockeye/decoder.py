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
from . import attention as attentions
from . import constants as C
from . import encoder
from . import layers
from . import lexicon as lexicons
from . import rnn
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


def get_decoder(config: Config,
                lexicon: Optional[lexicons.Lexicon] = None,
                embed_weight: Optional[mx.sym.Symbol] = None) -> 'Decoder':
    if isinstance(config, RecurrentDecoderConfig):
        return RecurrentDecoder(config=config, lexicon=lexicon, embed_weight=embed_weight, prefix=C.DECODER_PREFIX)
    elif isinstance(config, transformer.TransformerConfig):
        return TransformerDecoder(config=config, embed_weight=embed_weight, prefix=C.DECODER_PREFIX)
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
                        target: mx.sym.Symbol,
                        target_lengths: mx.sym.Symbol,
                        target_max_length: int,
                        source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Decodes given a known target sequence and returns logits
        with batch size and target length dimensions collapsed.
        Used for training.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target: Target sequence. Shape: (batch_size, target_max_length).
        :param target_lengths: Lengths of target sequences. Shape: (batch_size,).
        :param target_max_length: Size of target sequence dimension.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len)
        :return: Logits of next-word predictions for target sequence.
                 Shape: (batch_size * target_max_length, target_vocab_size)
        """
        pass

    @abstractmethod
    def decode_step(self,
                    prev_word_id: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) \
            -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the previous word id and previous decoder states.
        Returns logits, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param prev_word_id: Previous word id. Shape: (batch_size,).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logits, attention probabilities, next decoder states.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset decoder method. Used for inference.
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
        return []

    @abstractmethod
    def state_variables(self) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :return: List of symbolic variables.
        """
        return []

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
        return []

    @abstractmethod
    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.

        :raises: NotImplementedError
        """
        return []


class TransformerDecoder(Decoder):
    """
    Transformer decoder as in Vaswani et al, 2017: Attention is all you need.
    In training, computation scores for each position of the known target sequence are compouted in parallel,
    yielding most of the speedup.
    At inference time, the decoder block is evaluated again and again over a maximum length input sequence that is
    initially filled with zeros and grows during beam search with predicted tokens. Appropriate masking at every
    time-step ensures correct self-attention scores and is updated with every step.

    :param config: Transformer configuration.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new target embedding.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 embed_weight: Optional[mx.sym.Symbol] = None,
                 prefix: str = C.TRANSFORMER_DECODER_PREFIX) -> None:
        self.config = config
        self.prefix = prefix
        self.layers = [transformer.TransformerDecoderBlock(
            config, prefix="%s%d_" % (prefix, i)) for i in range(config.num_layers)]

        # Embedding & output parameters
        if embed_weight is None:
            embed_weight = mx.sym.Variable(C.TARGET_EMBEDDING_PREFIX + "weight")

        self.embedding = encoder.Embedding(num_embed=config.model_size,
                                           vocab_size=config.vocab_size,
                                           prefix=C.TARGET_EMBEDDING_PREFIX,
                                           dropout=config.dropout_residual,
                                           embed_weight=embed_weight,
                                           add_positional_encoding=config.positional_encodings)
        if self.config.weight_tying:
            logger.info("Tying the target embeddings and prediction matrix.")
            self.cls_w = embed_weight
        else:
            self.cls_w = mx.sym.Variable("%scls_weight" % prefix)
        self.cls_b = mx.sym.Variable("%scls_bias" % prefix)

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target: mx.sym.Symbol,
                        target_lengths: mx.sym.Symbol,
                        target_max_length: int,
                        source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Decodes given a known target sequence and returns logits
        with batch size and target length dimensions collapsed.
        Used for training.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target: Target sequence. Shape: (batch_size, target_max_length).
        :param target_lengths: Lengths of target sequences. Shape: (batch_size,).
        :param target_max_length: Size of target sequence dimension.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len)
        :return: Logits of next-word predictions for target sequence.
                 Shape: (batch_size * target_max_length, target_vocab_size)
        """
        # (1, target_max_length, target_max_length)
        target_bias = transformer.get_autoregressive_bias(target_max_length, name="%sbias" % self.prefix)

        # (batch_size, source_max_length, num_source_embed)
        source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

        # target: (batch_size, target_max_length, model_size)
        target, target_lengths, target_max_length = self.embedding.encode(target, target_lengths, target_max_length)

        for layer in self.layers:
            target = layer(target, target_lengths, target_max_length, target_bias,
                           source_encoded, source_encoded_lengths, source_encoded_max_length)

        # target: (batch_size * target_max_length, model_size)
        target = mx.sym.reshape(data=target, shape=(-3, -1))

        # logits: (batch_size * target_max_length, vocab_size)
        logits = mx.sym.FullyConnected(data=target, num_hidden=self.config.vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)
        return logits

    def decode_step(self,
                    prev_word_id: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) \
            -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the previous word id and previous decoder states.
        Returns logits, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param prev_word_id: Previous word id. Shape: (batch_size,).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logits, attention probabilities, next decoder states.
        """
        target_max_length = self._get_target_max_length(source_encoded_max_length)

        # sequences: (batch_size, target_max_length)
        source_encoded, source_encoded_lengths, sequences, lengths = states

        # (batch_size, target_max_length)
        mask = mx.sym.one_hot(indices=lengths, depth=target_max_length, on_value=1, off_value=0)

        # all zeros but length position is set to prev_word_id
        # (batch_size, target_max_length)
        prev_word_id = mx.sym.broadcast_mul(mx.sym.expand_dims(prev_word_id, axis=1), mask)

        # append/insert prev_word_id to sequences
        # (batch_size, target_max_length)
        sequences = sequences + prev_word_id
        lengths += 1

        # (1, target_max_length, target_max_length)
        target_bias = transformer.get_autoregressive_bias(target_max_length, name="%sbias" % self.prefix)

        # (batch_size, target_max_length, model_size)
        target, target_lengths, target_max_length = self.embedding.encode(sequences, lengths, target_max_length)

        for layer in self.layers:
            target = layer(target, target_lengths, target_max_length, target_bias,
                           source_encoded, source_encoded_lengths, source_encoded_max_length)

        # set all target positions to zero except for current time-step
        # target: (batch_size, target_max_length, model_size)
        target = mx.sym.broadcast_mul(target, mx.sym.expand_dims(mask, axis=2))
        # reduce to single prediction
        # target: (batch_size, model_size)
        target = mx.sym.sum(target, axis=1, keepdims=False)
        # logits: (batch_size, vocab_size)
        logits = mx.sym.FullyConnected(data=target, num_hidden=self.config.vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

        # TODO(fhieber): no attention probs for now
        attention_probs = mx.sym.sum(mx.sym.zeros_like(source_encoded), axis=2, keepdims=False)

        # next states
        new_states = [source_encoded, source_encoded_lengths, sequences, lengths]
        return logits, attention_probs, new_states

    def reset(self):
        pass

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
        target_max_length = self._get_target_max_length(source_encoded_max_length)
        # 0s: (batch_size, target_max_length)
        sequences = mx.sym.broadcast_axis(mx.sym.expand_dims(mx.sym.zeros_like(source_encoded_lengths), axis=1),
                                          axis=1,
                                          size=target_max_length)
        # 0s: (batch_size, source_encoded_max_length, encoder_depth)
        lengths = mx.sym.zeros_like(source_encoded_lengths)
        return [source_encoded, source_encoded_lengths, sequences, lengths]

    def state_variables(self) -> List[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :return: List of symbolic variables.
        """
        return [mx.sym.Variable(C.SOURCE_ENCODED_NAME),
                mx.sym.Variable(C.SOURCE_LENGTH_NAME),
                mx.sym.Variable('sequences'),
                mx.sym.Variable('lengths')]

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
        target_max_length = self._get_target_max_length(source_encoded_max_length)
        return [mx.io.DataDesc(C.SOURCE_ENCODED_NAME,
                               (batch_size, source_encoded_max_length, source_encoded_depth),
                               layout=C.BATCH_MAJOR),
                mx.io.DataDesc(C.SOURCE_LENGTH_NAME, (batch_size,), layout="N"),
                mx.io.DataDesc('sequences', (batch_size, target_max_length), layout="NC"),
                mx.io.DataDesc('lengths', (batch_size,), layout="N")]

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.
        """
        return super().get_rnn_cells()

    @staticmethod
    def _get_target_max_length(source_encoded_max_length: int):
        # TODO(fhieber): we need to hardcode this for the inference algorithm to work.
        # This is currently in line with the beam_search algorithm but not ideal.
        return source_encoded_max_length * C.TARGET_MAX_LENGTH_FACTOR


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

    :param vocab_size: Target vocabulary size.
    :param max_seq_len_source: Maximum source sequence length
    :param num_embed: Target word embedding size.
    :param rnn_config: RNN configuration.
    :param attention_config: Attention configuration.
    :param embed_dropout: Dropout probability for target embeddings.
    :param hidden_dropout: Dropout probability on next decoder hidden state.
    :param weight_tying: Whether to share embedding and prediction parameter matrices.
    :param zero_state_init: If true, initialize RNN states with zeros.
    :param context_gating: Whether to use context gating.
    :param layer_normalization: Apply layer normalization.
    """

    def __init__(self,
                 vocab_size: int,
                 max_seq_len_source: int,
                 num_embed: int,
                 rnn_config: rnn.RNNConfig,
                 attention_config: attentions.AttentionConfig,
                 embed_dropout: float = .0,
                 hidden_dropout: float = .0,
                 weight_tying: bool = False,
                 zero_state_init: bool = False,
                 context_gating: bool = False,
                 layer_normalization: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len_source = max_seq_len_source
        self.num_embed = num_embed
        self.rnn_config = rnn_config
        self.attention_config = attention_config
        self.embed_dropout = embed_dropout
        self.hidden_dropout = hidden_dropout
        self.weight_tying = weight_tying
        self.zero_state_init = zero_state_init
        self.context_gating = context_gating
        self.layer_normalization = layer_normalization


class RecurrentDecoder(Decoder):
    """
    RNN Decoder with attention.
    The architecture is based on Luong et al, 2015: Effective Approaches to Attention-based Neural Machine Translation.

    :param config: Configuration for recurrent decoder.
    :param lexicon: Optional Lexicon.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new target embedding.
    :param prefix: Decoder symbol prefix.
    """

    def __init__(self,
                 config: RecurrentDecoderConfig,
                 lexicon: Optional[lexicons.Lexicon] = None,
                 embed_weight: Optional[mx.sym.Symbol] = None,
                 prefix: str = C.DECODER_PREFIX) -> None:
        # TODO: implement variant without input feeding
        self.config = config
        self.rnn_config = config.rnn_config
        self.attention = attentions.get_attention(config.attention_config, config.max_seq_len_source)
        self.lexicon = lexicon
        self.prefix = prefix

        self.num_hidden = self.rnn_config.num_hidden

        if self.config.context_gating:
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

        self.rnn = rnn.get_stacked_rnn(self.rnn_config, self.prefix)

        if not self.config.zero_state_init:
            self._create_state_init_parameters()

        # Hidden state parameters
        self.hidden_w = mx.sym.Variable("%shidden_weight" % prefix)
        self.hidden_b = mx.sym.Variable("%shidden_bias" % prefix)
        self.hidden_norm = layers.LayerNormalization(self.num_hidden,
                                                     prefix="%shidden_norm" % prefix) \
            if self.config.layer_normalization else None

        # Embedding & output parameters
        if embed_weight is None:
            embed_weight = mx.sym.Variable(C.TARGET_EMBEDDING_PREFIX + "weight")
        self.embedding = encoder.Embedding(self.config.num_embed,
                                           self.config.vocab_size,
                                           prefix=C.TARGET_EMBEDDING_PREFIX,
                                           dropout=config.embed_dropout,
                                           embed_weight=embed_weight)
        if self.config.weight_tying:
            check_condition(self.num_hidden == self.config.num_embed,
                            "Weight tying requires target embedding size and rnn_num_hidden to be equal")
            logger.info("Tying the target embeddings and prediction matrix.")
            self.cls_w = embed_weight
        else:
            self.cls_w = mx.sym.Variable("%scls_weight" % prefix)
        self.cls_b = mx.sym.Variable("%scls_bias" % prefix)

    def _create_state_init_parameters(self):
        """
        Creates parameters for encoder last state transformation into decoder layer initial states.
        """
        self.init_ws, self.init_bs, self.init_norms = [], [], []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
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
                        target: mx.sym.Symbol,
                        target_lengths: mx.sym.Symbol,
                        target_max_length: int,
                        source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Decodes given a known target sequence and returns logits
        with batch size and target length dimensions collapsed.
        Used for training.

        :param source_encoded: Encoded source: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target: Target sequence. Shape: (batch_size, target_max_length).
        :param target_lengths: Lengths of target sequences. Shape: (batch_size,).
        :param target_max_length: Size of target sequence dimension.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len)
        :return: Logits of next-word predictions for target sequence.
                 Shape: (batch_size * target_max_length, target_vocab_size)
        """
        # embed and slice target words
        # target_embed: (batch_size, target_seq_len, num_target_embed)
        target_embed, target_lengths, target_max_length = self.embedding.encode(target, target_lengths,
                                                                                target_max_length)
        # target_embed: target_seq_len * (batch_size, num_target_embed)
        target_embed = mx.sym.split(data=target_embed, num_outputs=target_max_length, axis=1, squeeze_axis=True)

        # get recurrent attention function conditioned on source
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')
        attention_func = self.attention.on(source_encoded_batch_major, source_encoded_lengths, source_encoded_max_length)
        attention_state = self.attention.get_initial_state(source_encoded_lengths, source_encoded_max_length)

        # initialize decoder states
        # hidden: (batch_size, rnn_num_hidden)
        # layer_states: List[(batch_size, state_num_hidden]
        state = self.get_initial_state(source_encoded, source_encoded_lengths)

        # hidden_all: target_seq_len * (batch_size, 1, rnn_num_hidden)
        hidden_all = []

        # TODO: possible alternative: feed back the context vector instead of the hidden (see lamtram)

        lexical_biases = []

        self.reset()

        for seq_idx in range(target_max_length):
            # hidden: (batch_size, rnn_num_hidden)
            state, attention_state = self._step(target_embed[seq_idx],
                                                state,
                                                attention_func,
                                                attention_state,
                                                seq_idx)

            # hidden_expanded: (batch_size, 1, rnn_num_hidden)
            hidden_all.append(mx.sym.expand_dims(data=state.hidden, axis=1))

            if source_lexicon is not None:
                assert self.lexicon is not None, "source_lexicon should not be None if no lexicon available"
                lexical_biases.append(self.lexicon.calculate_lex_bias(source_lexicon, attention_state.probs))

        # concatenate along time axis
        # hidden_concat: (batch_size, target_seq_len, rnn_num_hidden)
        hidden_concat = mx.sym.concat(*hidden_all, dim=1, name="%shidden_concat" % self.prefix)
        # hidden_concat: (batch_size * target_seq_len, rnn_num_hidden)
        hidden_concat = mx.sym.reshape(data=hidden_concat, shape=(-1, self.num_hidden))

        # logits: (batch_size * target_seq_len, target_vocab_size)
        logits = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.config.vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

        if source_lexicon is not None:
            # lexical_biases_concat: (batch_size, target_seq_len, target_vocab_size)
            lexical_biases_concat = mx.sym.concat(*lexical_biases, dim=1, name='lex_bias_concat')
            # lexical_biases_concat: (batch_size * target_seq_len, target_vocab_size)
            lexical_biases_concat = mx.sym.reshape(data=lexical_biases_concat, shape=(-1, self.config.vocab_size))
            logits = mx.sym.broadcast_add(lhs=logits, rhs=lexical_biases_concat,
                                          name='%s_plus_lex_bias' % C.LOGITS_NAME)

        return logits

    def decode_step(self,
                    prev_word_id: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    *states: mx.sym.Symbol) \
            -> Tuple[mx.sym.Symbol, mx.sym.Symbol, List[mx.sym.Symbol]]:
        """
        Decodes a single time step given the previous word id and previous decoder states.
        Returns logits, attention probabilities, and next decoder states.
        Implementations can maintain an arbitrary number of states.

        :param prev_word_id: Previous word id. Shape: (batch_size,).
        :param source_encoded_max_length: Length of encoded source time dimension.
        :param states: Arbitrary list of decoder states.
        :return: logits, attention probabilities, next decoder states.
        """
        source_encoded, prev_dynamic_source, source_encoded_length, prev_hidden, *layer_states = states

        word_vec_prev, _, _ = self.embedding.encode(prev_word_id, None, 1)

        attention_func = self.attention.on(source_encoded, source_encoded_length, source_encoded_max_length)

        prev_state = RecurrentDecoderState(prev_hidden, list(layer_states))
        prev_attention_state = attentions.AttentionState(context=None, probs=None, dynamic_source=prev_dynamic_source)

        # state.hidden: (batch_size, rnn_num_hidden)
        # attention_state.dynamic_source: (batch_size, source_seq_len, coverage_num_hidden)
        # attention_state.probs: (batch_size, source_seq_len)
        state, attention_state = self._step(word_vec_prev,
                                            prev_state,
                                            attention_func,
                                            prev_attention_state)

        # logits: (batch_size, target_vocab_size)
        logits = mx.sym.FullyConnected(data=state.hidden, num_hidden=self.config.vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

        new_states = [source_encoded,
                      attention_state.dynamic_source,
                      source_encoded_length,
                      state.hidden] + state.layer_states

        return logits, attention_state.probs, new_states

    def reset(self):
        """
        Calls reset on the RNN cell.
        """
        self.rnn.reset()
        # TODO remove this once mxnet.rnn.SequentialRNNCell.reset() invokes recursive calls on layer cells
        for cell in self.rnn._cells:
            # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
            if isinstance(cell, mx.rnn.ModifierCell):
                cell.base_cell.reset()
            cell.reset()

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
                range(len(self.rnn.state_info))]

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
                               layout=C.BATCH_MAJOR) for i, (_, num_hidden) in enumerate(self.rnn.state_shape)]

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.
        """
        return [self.rnn]

    def get_initial_state(self,
                          source_encoded: mx.sym.Symbol,
                          source_encoded_length: mx.sym.Symbol) -> RecurrentDecoderState:
        """
        Computes initial states of the decoder, hidden state, and one for each RNN layer.
        Optionally, init states for RNN layers are computed using 1 non-linear FC
        with the last state of the encoder as input.

        :param source_encoded: Concatenated encoder states. Shape: (batch_size, source_seq_len, encoder_num_hidden).
        :param source_encoded_length: Lengths of source sequences. Shape: (batch_size,).
        :return: Decoder state.
        """
        # we derive the shape of hidden and layer_states from some input to enable
        # shape inference for the batch dimension during inference.
        # (batch_size, 1)
        zeros = mx.sym.expand_dims(mx.sym.zeros_like(source_encoded_length), axis=1)
        # last encoder state
        source_encoded_last = mx.sym.SequenceLast(data=source_encoded,
                                                  sequence_length=source_encoded_length,
                                                  use_sequence_length=True) if not self.config.zero_state_init else None

        # decoder hidden state
        hidden = mx.sym.tile(data=zeros, reps=(1, self.num_hidden))

        # initial states for each layer
        layer_states = []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            if self.config.zero_state_init:
                init = mx.sym.tile(data=zeros, reps=(1, init_num_hidden))
            else:
                init = mx.sym.FullyConnected(data=source_encoded_last,
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

    def _step(self,
              word_vec_prev: mx.sym.Symbol,
              state: RecurrentDecoderState,
              attention_func: Callable,
              attention_state: attentions.AttentionState,
              seq_idx: int = 0) -> Tuple[RecurrentDecoderState, attentions.AttentionState]:

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
        # rnn_output: (batch_size, rnn_num_hidden)
        # next_layer_states: num_layers * [batch_size, rnn_num_hidden]
        rnn_output, layer_states = self.rnn(rnn_input, state.layer_states)

        # (2) Attention step
        attention_input = self.attention.make_input(seq_idx, word_vec_prev, rnn_output)
        attention_state = attention_func(attention_input, attention_state)

        # (3) Combine previous word vector, rnn output, and context
        hidden_concat = mx.sym.concat(rnn_output, attention_state.context,
                                      dim=1, name='%shidden_concat_t%d' % (self.prefix, seq_idx))
        if self.config.hidden_dropout > 0:
            hidden_concat = mx.sym.Dropout(data=hidden_concat, p=self.config.hidden_dropout,
                                           name='%shidden_concat_dropout_t%d' % (self.prefix, seq_idx))

        if self.config.context_gating:
            hidden = self._context_gate(hidden_concat, rnn_output, attention_state, seq_idx)
        else:
            hidden = self._hidden_mlp(hidden_concat, seq_idx)

        return RecurrentDecoderState(hidden, layer_states), attention_state

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
                      attention_state: attentions.AttentionState,
                      seq_idx: int) -> mx.sym.Symbol:
        gate = mx.sym.FullyConnected(data=hidden_concat,
                                     num_hidden=self.num_hidden,
                                     weight=self.gate_w,
                                     bias=self.gate_b,
                                     name = '%shidden_gate_t%d' % (self.prefix, seq_idx))
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

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
from typing import Callable, List, NamedTuple, Tuple
from typing import Optional
import logging

import mxnet as mx

from sockeye.config import Config
from sockeye.layers import LayerNormalization
from sockeye.utils import check_condition
from . import attention as attentions
from . import constants as C
from . import encoder
from . import lexicon as lexicons
from . import rnn


logger = logging.getLogger(__name__)


class RecurrentDecoderConfig(Config):
    """
    Recurrent decoder configuration.

    :param vocab_size: Target vocabulary size.
    :param num_embed: Target word embedding size.
    :param rnn_config: RNN configuration.
    :param dropout: Dropout probability for decoder RNN.
    :param weight_tying: Whether to share embedding and prediction parameter matrices.
    :param context_gating: Whether to use context gating.
    :param layer_normalization: Apply layer normalization.
    """
    def __init__(self,
                 vocab_size: int,
                 num_embed: int,
                 rnn_config: rnn.RNNConfig,
                 dropout: float = .0,
                 weight_tying: bool = False,
                 context_gating: bool = False,
                 layer_normalization: bool = False,
                 scheduled_sampling_type: Optional[str] = None,
                 scheduled_sampling_decay_params: Optional[List[float]] = None,
                 use_mrt: bool = False,
                 mrt_num_samples: int = 0,
                 mrt_sup_grad_scale: float = 1.0,
                 mrt_entropy_reg: float = 0.0,
                 mrt_max_target_len_ratio: float = 1.0,
                 mrt_metric: str = C.BLEU) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.rnn_config = rnn_config
        self.dropout = dropout
        self.weight_tying = weight_tying
        self.context_gating = context_gating
        self.layer_normalization = layer_normalization
        self.scheduled_sampling_type = scheduled_sampling_type
        self.scheduled_sampling_decay_params = scheduled_sampling_decay_params
        self.use_mrt = use_mrt
        self.mrt_num_samples = mrt_num_samples
        self.mrt_sup_grad_scale = mrt_sup_grad_scale
        self.mrt_entropy_reg = mrt_entropy_reg
        self.mrt_max_target_len_ratio = mrt_max_target_len_ratio
        self.mrt_metric = mrt_metric


def get_recurrent_decoder(config: RecurrentDecoderConfig,
                          attention: attentions.Attention,
                          lexicon: Optional[lexicons.Lexicon] = None,
                          embed_weight: Optional[mx.sym.Symbol] = None) -> 'Decoder':
    """
    Returns a recurrent decoder.

    :param config: Configuration for RecurrentDecoder.
    :param attention: Attention model.
    :param lexicon: Optional Lexicon.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new one.
    :return: Decoder instance.
    """
    return RecurrentDecoder(config,
                            attention=attention,
                            lexicon=lexicon,
                            prefix=C.DECODER_PREFIX,
                            embed_weight=embed_weight)


class Decoder:
    """
    Generic decoder interface.
    """

    def get_num_hidden(self) -> int:
        """
        Returns the representation size of this decoder.

        :raises: NotImplementedError
        """
        raise NotImplementedError()

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.

        :raises: NotImplementedError
        """
        raise NotImplementedError()


DecoderState = NamedTuple('DecoderState', [
    ('hidden', mx.sym.Symbol),
    ('layer_states', List[mx.sym.Symbol]),
])
"""
Decoder state.

:param hidden: Hidden state after attention mechanism. Shape: (batch_size, num_hidden).
:param layer_states: Hidden states for RNN layers of RecurrentDecoder. Shape: List[(batch_size, rnn_num_hidden)]

"""


class RecurrentDecoder(Decoder):
    """
    Class to generate the decoder part of the computation graph in sequence-to-sequence models.
    The architecture is based on Luong et al, 2015: Effective Approaches to Attention-based Neural Machine Translation.

    :param config: Configuration for recurrent decoder.
    :param attention: Attention model.
    :param lexicon: Optional Lexicon.
    :param prefix: Decoder symbol prefix.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new target embedding.
    """

    def __init__(self,
                 config: RecurrentDecoderConfig,
                 attention: attentions.Attention,
                 lexicon: Optional[lexicons.Lexicon] = None,
                 prefix=C.DECODER_PREFIX,
                 embed_weight: Optional[mx.sym.Symbol]=None) -> None:
        # TODO: implement variant without input feeding
        self.rnn_config = config.rnn_config
        self.target_vocab_size = config.vocab_size
        self.num_target_embed = config.num_embed
        self.attention = attention
        self.weight_tying = config.weight_tying
        self.context_gating = config.context_gating
        self.layer_norm = config.layer_normalization
        self.lexicon = lexicon
        self.prefix = prefix

        self.num_hidden = self.rnn_config.num_hidden

        if self.context_gating:
            self.gate_w = mx.sym.Variable("%sgate_weight" % prefix)
            self.gate_b = mx.sym.Variable("%sgate_bias" % prefix)
            self.mapped_rnn_output_w = mx.sym.Variable("%smapped_rnn_output_weight" % prefix)
            self.mapped_rnn_output_b = mx.sym.Variable("%smapped_rnn_output_bias" % prefix)
            self.mapped_context_w = mx.sym.Variable("%smapped_context_weight" % prefix)
            self.mapped_context_b = mx.sym.Variable("%smapped_context_bias" % prefix)

        # Stacked RNN
        self.rnn = rnn.get_stacked_rnn(self.rnn_config, self.prefix)
        # RNN init state parameters
        self._create_layer_parameters()

        # Hidden state parameters
        self.hidden_w = mx.sym.Variable("%shidden_weight" % prefix)
        self.hidden_b = mx.sym.Variable("%shidden_bias" % prefix)
        self.hidden_norm = LayerNormalization(self.num_hidden,
                                              prefix="%shidden_norm" % prefix) if self.layer_norm else None
        # Embedding & output parameters
        if embed_weight is None:
            embed_weight = mx.sym.Variable(C.TARGET_EMBEDDING_PREFIX + "weight")
        self.embedding = encoder.Embedding(self.num_target_embed, self.target_vocab_size,
                                           prefix=C.TARGET_EMBEDDING_PREFIX, dropout=0.,
                                           embed_weight=embed_weight)  # TODO dropout?
        if self.weight_tying:
            check_condition(self.num_hidden == self.num_target_embed,
                            "Weight tying requires target embedding size and rnn_num_hidden to be equal")
            logger.debug("Tying the target embeddings and prediction matrix.")
            self.cls_w = embed_weight
        else:
            self.cls_w = mx.sym.Variable("%scls_weight" % prefix)
        self.cls_b = mx.sym.Variable("%scls_bias" % prefix)

        self.scheduled_sampling_type = config.scheduled_sampling_type
        self.scheduled_sampling_decay_params = config.scheduled_sampling_decay_params
        self._get_sampling_threshold = self._link_sampling_scheduler()

    def _link_sampling_scheduler(self):
         i = mx.sym.Variable('updates')

         def _check_params(params):
             if self.scheduled_sampling_type == 'inv-sigmoid-decay':
                 assert len(params) == 2, \
                     ('The inverse sigmoid decaying option for scheduled sampling requires 2 parameter,'
                     ' but given {}').format(len(params))

                 k, lower_bound = params
                 assert k >= 1, 'Offset should be greater than or equal to 1'
                 assert lower_bound >= 0 and lower_bound < 1

             elif self.scheduled_sampling_type == 'exponential-decay':
                 assert len(params) == 1, \
                     ('The exponential decaying option for scheduled sampling requires 1 parameter,'
                     ' but given {}').format(len(params))
                 k = params[0]
                 assert k < 1, 'Offset for the exponential decay should be less than 1.'

             elif self.scheduled_sampling_type == 'linear-decay':
                 assert len(params) == 3, \
                     ('The linear decaying option for scheduled sampling requires 3 parameter,'
                     ' but given {}').format(len(params))

                 k, epsilon, slope = params

                 assert k <= 1, 'Offset for the linear decay should be less than 1.'
                 assert epsilon >= 0, 'Epsilonfor the linear decay should be greather than or equal to 0.'

         _check_params(self.scheduled_sampling_decay_params)

         if self.scheduled_sampling_type == 'inv-sigmoid-decay':
             k, lower_bound = self.scheduled_sampling_decay_params
             sampling_scheduler = lambda: (k /(k + mx.sym.exp(i / k) -1)) * (1-lower_bound) + lower_bound
         elif self.scheduled_sampling_type == 'exponential-decay':
             k = self.scheduled_sampling_decay_params[0]
             sampling_scheduler = lambda: pow(k, i)
         elif self.scheduled_sampling_type == 'linear-decay':
             k, epsilon, slope = self.scheduled_sampling_decay_params
             sampling_scheduler = lambda: mx.sym.clip(k - slope * i, epsilon, k)
         else:
             sampling_scheduler = lambda: 1

         return sampling_scheduler

    def get_num_hidden(self) -> int:
        """
        Returns the representation size of this decoder.

        :return: Number of hidden units.
        """
        return self.num_hidden

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.
        """
        return [self.rnn]

    def _create_layer_parameters(self):
        """
        Creates parameters for encoder last state transformation into decoder layer initial states.
        """
        self.init_ws, self.init_bs = [], []
        self.init_norms = []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            self.init_ws.append(mx.sym.Variable("%senc2decinit_%d_weight" % (self.prefix, state_idx)))
            self.init_bs.append(mx.sym.Variable("%senc2decinit_%d_bias" % (self.prefix, state_idx)))
            if self.layer_norm:
                self.init_norms.append(LayerNormalization(num_hidden=init_num_hidden,
                                                          prefix="%senc2decinit_%d_norm" % (self.prefix, state_idx)))

    def create_layer_input_variables(self, batch_size: int) \
            -> Tuple[List[mx.sym.Symbol], List[mx.io.DataDesc], List[str]]:
        """
        Creates RNN layer state variables. Used for inference.
        Returns nested list of layer_states variables, flat list of layer shapes (for module binding),
        and a flat list of layer names (for BucketingModule's data names)

        :param batch_size: Batch size.
        """
        layer_states, layer_shapes, layer_names = [], [], []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            name = "%senc2decinit_%d" % (self.prefix, state_idx)
            layer_states.append(mx.sym.Variable(name))
            layer_shapes.append(mx.io.DataDesc(name=name, shape=(batch_size, init_num_hidden), layout=C.BATCH_MAJOR))
            layer_names.append(name)
        return layer_states, layer_shapes, layer_names

    def compute_init_states(self,
                            source_encoded: mx.sym.Symbol,
                            source_length: mx.sym.Symbol) -> DecoderState:
        """
        Computes initial states of the decoder, hidden state, and one for each RNN layer.
        Init states for RNN layers are computed using 1 non-linear FC with the last state of the encoder as input.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_length: Lengths of source sequences. Shape: (batch_size,).
        :return: Decoder state.
        """
        # initial decoder hidden state
        hidden = mx.sym.tile(data=mx.sym.expand_dims(data=source_length * 0, axis=1), reps=(1, self.num_hidden))
        # initial states for each layer
        layer_states = []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            init = mx.sym.FullyConnected(data=mx.sym.SequenceLast(data=source_encoded,
                                                                  sequence_length=source_length,
                                                                  use_sequence_length=True),
                                         num_hidden=init_num_hidden,
                                         weight=self.init_ws[state_idx],
                                         bias=self.init_bs[state_idx],
                                         name="%senc2decinit_%d" % (self.prefix, state_idx))
            if self.layer_norm:
                init = self.init_norms[state_idx].normalize(init)
            init = mx.sym.Activation(data=init, act_type="tanh",
                                     name="%senc2dec_inittanh_%d" % (self.prefix, state_idx))
            layer_states.append(init)
        return DecoderState(hidden, layer_states)

    def _step(self,
              word_vec_prev: mx.sym.Symbol,
              state: DecoderState,
              attention_func: Callable,
              attention_state: attentions.AttentionState,
              seq_idx: int = 0) -> Tuple[DecoderState, attentions.AttentionState]:

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

        # (3) Combine context with hidden state
        if self.context_gating:
            # context: (batch_size, encoder_num_hidden)
            # gate: (batch_size, rnn_num_hidden)
            gate = mx.sym.FullyConnected(data=mx.sym.concat(word_vec_prev, rnn_output, attention_state.context, dim=1),
                                         num_hidden=self.num_hidden, weight=self.gate_w, bias=self.gate_b)
            gate = mx.sym.Activation(data=gate, act_type="sigmoid",
                                     name="%sgate_activation_t%d" % (self.prefix, seq_idx))

            # mapped_rnn_output: (batch_size, rnn_num_hidden)
            mapped_rnn_output = mx.sym.FullyConnected(data=rnn_output,
                                                      num_hidden=self.num_hidden,
                                                      weight=self.mapped_rnn_output_w,
                                                      bias=self.mapped_rnn_output_b,
                                                      name="%smapped_rnn_output_fc_t%d" % (self.prefix, seq_idx))
            # mapped_context: (batch_size, rnn_num_hidden)
            mapped_context = mx.sym.FullyConnected(data=attention_state.context,
                                                   num_hidden=self.num_hidden,
                                                   weight=self.mapped_context_w,
                                                   bias=self.mapped_context_b,
                                                   name="%smapped_context_fc_t%d" % (self.prefix, seq_idx))

            # hidden: (batch_size, rnn_num_hidden)
            hidden = mx.sym.Activation(data=gate * mapped_rnn_output + (1 - gate) * mapped_context,
                                       act_type="tanh",
                                       name="%snext_hidden_t%d" % (self.prefix, seq_idx))

        else:
            # hidden: (batch_size, rnn_num_hidden)
            hidden = mx.sym.FullyConnected(data=mx.sym.concat(rnn_output, attention_state.context, dim=1),
                                           # use same number of hidden states as RNN
                                           num_hidden=self.num_hidden,
                                           weight=self.hidden_w,
                                           bias=self.hidden_b)

            if self.layer_norm:
                hidden = self.hidden_norm.normalize(hidden)

            # hidden: (batch_size, rnn_num_hidden)
            hidden = mx.sym.Activation(data=hidden, act_type="tanh",
                                       name="%snext_hidden_t%d" % (self.prefix, seq_idx))

        return DecoderState(hidden, layer_states), attention_state

    def decode(self,
               source_encoded: mx.sym.Symbol,
               source_seq_len: int,
               source_length: mx.sym.Symbol,
               target: mx.sym.Symbol,
               target_seq_len: int,
               source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns decoder logits with batch size and target sequence length collapsed into a single dimension.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_seq_len: Maximum source sequence length.
        :param source_length: Lengths of source sequences. Shape: (batch_size,).
        :param target: Target sequence. Shape: (batch_size, target_seq_len).
        :param target_seq_len: Maximum target sequence length.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len)
        :return: Logits of next-word predictions for target sequence.
                 Shape: (batch_size * target_seq_len, target_vocab_size)
        """
        # process encoder states
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')

        # embed and slice target words
        # target_embed: (batch_size, target_seq_len, num_target_embed)
        target_embed, _, _ = self.embedding.encode(target, None, target_seq_len)
        # target_embed: target_seq_len * (batch_size, num_target_embed)
        target_embed = mx.sym.split(data=target_embed, num_outputs=target_seq_len, axis=1, squeeze_axis=True)

        # get recurrent attention function conditioned on source
        attention_func = self.attention.on(source_encoded_batch_major, source_length, source_seq_len)
        attention_state = self.attention.get_initial_state(source_length, source_seq_len)

        # initialize decoder states
        # hidden: (batch_size, rnn_num_hidden)
        # layer_states: List[(batch_size, state_num_hidden]
        state = self.compute_init_states(source_encoded, source_length)

        # hidden_all: target_seq_len * (batch_size, 1, rnn_num_hidden)
        hidden_all = []

        # TODO: possible alternative: feed back the context vector instead of the hidden (see lamtram)

        lexical_biases = []

        self.rnn.reset()
        # TODO remove this once mxnet.rnn.SequentialRNNCell.reset() invokes recursive calls on layer cells
        for cell in self.rnn._cells:
            cell.reset()

        for seq_idx in range(target_seq_len):
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
        logits = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.target_vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

        if source_lexicon is not None:
            # lexical_biases_concat: (batch_size, target_seq_len, target_vocab_size)
            lexical_biases_concat = mx.sym.concat(*lexical_biases, dim=1, name='lex_bias_concat')
            # lexical_biases_concat: (batch_size * target_seq_len, target_vocab_size)
            lexical_biases_concat = mx.sym.reshape(data=lexical_biases_concat, shape=(-1, self.target_vocab_size))
            logits = mx.sym.broadcast_add(lhs=logits, rhs=lexical_biases_concat,
                                          name='%s_plus_lex_bias' % C.LOGITS_NAME)

        return logits

    def decode_iter(self,
                    source_encoded: mx.sym.Symbol,
                    source_seq_len: int,
                    source_length: mx.sym.Symbol,
                    target: mx.sym.Symbol,
                    target_seq_len: int,
                    source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns sampled sequences of words and associated probabilities for those words.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_seq_len: Maximum source sequence length.
        :param source_length: Lengths of source sequences. Shape: (batch_size,).
        :param target: Target sequence. Shape: (batch_size, target_seq_len).
        :param target_seq_len: Maximum target sequence length.
        :return: Probabilities of next-word predictions for target sequence.
                 Shape: (batch_size * target_seq_len, target_vocab_size)

                 Threshold that determines to use the ground truth words or previous model predictions as inputs.
                 Shape: (1,)

        """

        # process encoder states
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')

        # slice target words
        # target: target_seq_len * (batch_size,)
        target = mx.sym.split(data=target, num_outputs=target_seq_len, axis=1, squeeze_axis=True)

        # get recurrent attention function conditioned on source
        attention_func = self.attention.on(source_encoded_batch_major, source_length, source_seq_len)
        attention_state = self.attention.get_initial_state(source_length, source_seq_len)

        # initialize decoder states
        # hidden: (batch_size, rnn_num_hidden)
        # layer_states: List[(batch_size, state_num_hidden]
        state = self.compute_init_states(source_encoded, source_length)

        self.rnn.reset()

        probs_all = []

        threshold = self._get_sampling_threshold()
        rnd_val = mx.sym.random_uniform(low=0, high=1.0, shape=1)[0]

        for seq_idx in range(target_seq_len):
            if seq_idx == 0:
                input_words = target[seq_idx]
            else:
                # choose input words depending on tossing a biased coin (p = rnd_val)
                # if rnd_val < threshold, then use ground truth
                # otherwise we use model predictions made at the previous time step.
                # XXX threshold decreases over time
                input_words = mx.sym.where(rnd_val < threshold,
                                           mx.sym.cast(target[seq_idx], dtype='int32'), sampled_words)

            trg_emb, _, _ = self.embedding.encode(input_words, None, 1)
            state, attention_state = self._step(trg_emb,
                                                state,
                                                attention_func,
                                                attention_state,
                                                seq_idx)

            # (batch_size, target_vocab_size)
            logit = mx.sym.FullyConnected(data=state.hidden, num_hidden=self.target_vocab_size,
                                          weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

            prob = mx.sym.softmax(logit)
            sampled_words = mx.sym.sample_multinomial(prob)
            sampled_words = mx.sym.BlockGrad(sampled_words)

            probs_all.append(prob)

        # probs: (batch_size, target_seq_len, target_vocab_size)
        probs = mx.sym.concat(*probs_all, dim=1, name='probs')
        # probs: (batch_size * target_seq_len, target_vocab_size)
        probs = mx.sym.reshape(data=probs, shape=(-1, self.target_vocab_size))

        return [probs, threshold]

    def decode_mrt(self,
                   source_encoded: mx.sym.Symbol,
                   source_seq_len: int,
                   source_length: mx.sym.Symbol,
                   labels: mx.sym.Symbol,
                   target_seq_len: int,
                   max_target_seq_len_ratio: int,
                   is_sample: mx.sym.Symbol,
                   entropy_reg: float=0.0001,
                   grad_scale: float=1.0,
                   source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns sampled sequences of words and associated probabilities for those words.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_seq_len: Maximum source sequence length.
        :param source_length: Lengths of source sequences. Shape: (batch_size,).
        :param labels: Target sequence. Shape: (batch_size, target_seq_len).
        :param target_seq_len: Maximum target sequence length.
        :param max_target_seq_len_ratio: Ratio of the generated sequences' maximum length w.r.t. the original maximum target sequence length.
        :param is_sample: An RNN chooses sampled words as next inputs if is_sample == 1, otherwise true targets.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len)
        :return: sampled words
                 Shape: (batch_size, max_target_seq_len)

                 sampling distribution of the sampled words
                 Shape: (batch_size, max_target_seq_len, target_vocab_size)

        """

        # process encoder states
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')

        # get recurrent attention function conditioned on source
        attention_func = self.attention.on(source_encoded_batch_major, source_length, source_seq_len)
        attention_state = self.attention.get_initial_state(source_length, source_seq_len)

        # initialize decoder states
        # hidden: (batch_size, rnn_num_hidden)
        # layer_states: List[(batch_size, state_num_hidden]
        state = self.compute_init_states(source_encoded, source_length)

        self.rnn.reset()

        sampled_words = []
        sample_probs = []
        sample_masks = []

        # The decoder is allowed to generate longer sentences
        # than the ground truths.
        max_target_seq_len = int(target_seq_len * max_target_seq_len_ratio)
        eos_pos_init = max_target_seq_len-1     # initial position of EOS

        # padding the label matrix
        pad_matrix = mx.sym.tile(mx.sym.expand_dims(mx.sym.ones_like(source_length), axis=1),
                                 reps=(1, max_target_seq_len - target_seq_len))
        labels = mx.sym.concat(*[labels, C.PAD_ID * pad_matrix], dim=1)

        split_labels = mx.sym.split(labels, num_outputs=max_target_seq_len, axis=1,
                                    squeeze_axis=True, name='split_labels')

        # The position of the EOS token in sampled sentences
        # intialized by the last position and the positions are updated once EOS
        # tokens generated.
        eos_pos = mx.sym.cast(eos_pos_init*mx.sym.ones_like(source_length), dtype='int32')

        # aux variable containing all 1s; not changed over time.
        all_ones = mx.sym.cast(mx.sym.ones_like(source_length), dtype='int32')

        # input words are initialized by BOS
        input_words = C.BOS_ID*mx.sym.cast(mx.sym.ones_like(source_length), dtype='int32')

        for seq_idx in range(max_target_seq_len):
            embedded, _, _ = self.embedding.encode(input_words, None, 1)
            state, attention_state = self._step(embedded,
                                                state,
                                                attention_func,
                                                attention_state,
                                                seq_idx)

            logit = mx.sym.FullyConnected(data=state.hidden, num_hidden=self.target_vocab_size,
                                          weight=self.cls_w, bias=self.cls_b)

            prob = mx.sym.softmax(logit)
            next_words = mx.sym.where(is_sample > 0,
                                      mx.sym.sample_multinomial(prob),
                                      mx.sym.cast(split_labels_[seq_idx], dtype='int32'))

            is_eos = (next_words == C.EOS_ID)       # check if the generated tokens are EOS or not.
            cond = eos_pos == eos_pos_init          # check whether the EOS token's positions are updated or not
            eos_pos = mx.sym.where(is_eos * cond,
                                   seq_idx * all_ones,      # true: updating 'eos_pos' with the current time step
                                   eos_pos)                 # false: keeping 'eos_pos' previously generated

            # if EOS tokens are generated at previous time steps, we don't care
            # what the decoder RNN generates afterwards.
            mask = eos_pos >= seq_idx

            sampled_words.append(next_words)
            sample_probs.append(prob)
            sample_masks.append(mask)

            input_words = next_words

        # sampled_words: (max_target_seq_len * batch_size,)
        sampled_words = mx.sym.concat(*sampled_words, dim=0, name='sampled_words')
        # sample_probs: ((max_target_seq_len * batch_size), target_vocab_size)
        sample_probs = mx.sym.concat(*sample_probs, dim=0, name='sample_probs')
        # sample_masks: (max_target_seq_len * batch_size,)
        sample_masks = mx.sym.concat(*sample_masks, dim=0, name='sample_masks')

        # apply masking not to propagate errors unnecessarily
        # TODO modifiy LogPolicy to take the mask matrix as an input argument.
        sample_probs = mx.sym.broadcast_mul(sample_probs,
                                            mx.sym.cast(mx.sym.reshape(sample_masks,
                                                                        shape=(-1, 1)),
                                                        dtype='float32'),
                                            out=sample_probs)

        sampled_words = mx.sym.Custom(action=sampled_words, prob=sample_probs, is_sampled=is_sample,
                                      name='log_policy', op_type='LogPolicy',
                                      entropy_reg=entropy_reg, scale=grad_scale)

        # reshape the outcomes
        sampled_words = mx.sym.reshape(sampled_words, shape=(max_target_seq_len, -1))
        sample_probs = mx.sym.reshape(sample_probs,
                                      shape=(max_target_seq_len, -1, self.target_vocab_size))

        # batch_size * target_seq_len
        sampled_words = mx.sym.swapaxes(mx.sym.cast(sampled_words,
                                                    dtype='float32'), dim1=0, dim2=1)
        # batch_size * target_seq_len * target_vocab_size
        sample_probs = mx.sym.swapaxes(sample_probs, dim1=0, dim2=1)

        return mx.sym.Group([sampled_words, sample_probs])

    def predict(self,
                word_id_prev: mx.sym.Symbol,
                state_prev: DecoderState,
                attention_func: Callable,
                attention_state_prev: attentions.AttentionState,
                source_lexicon: Optional[mx.sym.Symbol] = None,
                softmax_temperature: Optional[float] = None) -> Tuple[mx.sym.Symbol,
                                                                      DecoderState,
                                                                      attentions.AttentionState]:
        """
        Given previous word id, attention function, previous hidden state and RNN layer states,
        returns Softmax predictions (not a loss symbol), next hidden state, and next layer
        states. Used for inference.

        :param word_id_prev: Previous target word id. Shape: (1,).
        :param state_prev: Previous decoder state consisting of hidden and layer states.
        :param attention_func: Attention function to produce context vector.
        :param attention_state_prev: Previous attention state.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len).
        :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
        :return: (predicted next-word distribution, decoder state, attention state).
        """
        # target side embedding
        word_vec_prev, _, _ = self.embedding.encode(word_id_prev, None, 1)

        # state.hidden: (batch_size, rnn_num_hidden)
        # attention_state.dynamic_source: (batch_size, source_seq_len, coverage_num_hidden)
        # attention_state.probs: (batch_size, source_seq_len)
        state, attention_state = self._step(word_vec_prev,
                                            state_prev,
                                            attention_func,
                                            attention_state_prev)

        # logits: (batch_size, target_vocab_size)
        logits = mx.sym.FullyConnected(data=state.hidden, num_hidden=self.target_vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

        if source_lexicon is not None:
            assert self.lexicon is not None
            # lex_bias: (batch_size, 1, target_vocab_size)
            lex_bias = self.lexicon.calculate_lex_bias(source_lexicon, attention_state.probs)
            # lex_bias: (batch_size, target_vocab_size)
            lex_bias = mx.sym.reshape(data=lex_bias, shape=(-1, self.target_vocab_size))
            logits = mx.sym.broadcast_add(lhs=logits, rhs=lex_bias, name='%s_plus_lex_bias' % C.LOGITS_NAME)

        if softmax_temperature is not None:
            logits /= softmax_temperature

        softmax_out = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)
        return softmax_out, state, attention_state

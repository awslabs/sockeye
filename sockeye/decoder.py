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

"""
Decoders for sequence-to-sequence models.
"""
import logging
from abc import abstractmethod
from itertools import islice
from typing import Dict, List, Optional, Tuple, Union, Type

import mxnet as mx

from . import constants as C
from . import layers
from . import transformer

logger = logging.getLogger(__name__)
DecoderConfig = Union[transformer.TransformerConfig]


def get_decoder(config: DecoderConfig, inference_only: bool = False,
                prefix: str = '', dtype: str = C.DTYPE_FP32) -> 'Decoder':
    return Decoder.get_decoder(config, inference_only, prefix, dtype)


class Decoder(mx.gluon.Block):
    """
    Generic decoder interface.
    A decoder needs to implement code to decode a target sequence known in advance (decode_sequence),
    and code to decode a single word given its decoder state (decode_step).
    The latter is typically used for inference graphs in beam search.
    For the inference module to be able to keep track of decoder's states
    a decoder provides methods to return initial states (init_states), state variables and their shapes.
    """

    __registry = {}  # type: Dict[Type[DecoderConfig], Tuple[Type['Decoder'], str]]

    @classmethod
    def register(cls, config_type: Type[DecoderConfig], suffix: str):
        """
        Registers decoder type for configuration. Suffix is appended to decoder prefix.

        :param config_type: Configuration type for decoder.
        :param suffix: String to append to decoder prefix.

        :return: Class decorator.
        """
        def wrapper(target_cls):
            cls.__registry[config_type] = (target_cls, suffix)
            return target_cls

        return wrapper

    @classmethod
    def get_decoder(cls, config: DecoderConfig, inference_only: bool, prefix: str, dtype: str) -> 'Decoder':
        """
        Creates decoder based on config type.

        :param config: Decoder config.
        :param inference_only: Create a decoder that is only used for inference.
        :param prefix: Prefix to prepend for decoder.
        :param dtype: Data type for weights.

        :return: Decoder instance.
        """
        config_type = type(config)
        if config_type not in cls.__registry:
            raise ValueError('Unsupported decoder configuration %s' % config_type.__name__)
        decoder_cls, suffix = cls.__registry[config_type]
        # TODO: move final suffix/prefix construction logic into config builder
        return decoder_cls(config=config, inference_only=inference_only, prefix=prefix + suffix, dtype=dtype)  # type: ignore

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def state_structure(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def init_state_from_encoder(self,
                                encoder_outputs: mx.nd.NDArray,
                                encoder_valid_length: Optional[mx.nd.NDArray] = None,
                                target_embed: Optional[mx.nd.NDArray] = None) -> List[mx.nd.NDArray]:
        raise NotImplementedError()

    @abstractmethod
    def decode_seq(self, inputs: mx.nd.NDArray, states: List[mx.nd.NDArray]):
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param inputs: Encoded source: (batch_size, source_encoded_max_length, encoder_depth).
        :param states: List of initial states, as given by init_state_from_encoder().
        :return: Decoder output. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_hidden(self):
        raise NotImplementedError()


@Decoder.register(transformer.TransformerConfig, C.TRANSFORMER_DECODER_PREFIX)
class TransformerDecoder(Decoder, mx.gluon.HybridBlock):
    """
    Transformer decoder as in Vaswani et al, 2017: Attention is all you need.
    In training, computation scores for each position of the known target sequence are computed in parallel,
    yielding most of the speedup.
    At inference time, the decoder block is evaluated again and again over a maximum length input sequence that is
    initially filled with zeros and grows during beam search with predicted tokens. Appropriate masking at every
    time-step ensures correct self-attention scores and is updated with every step.

    :param config: Transformer configuration.
    :param prefix: Name prefix for symbols of this decoder.
    :param inference_only: Only use the model for inference enabling some optimizations,
                           such as disabling the auto-regressive mask.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 prefix: str = C.TRANSFORMER_DECODER_PREFIX,
                 inference_only: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        Decoder.__init__(self)
        mx.gluon.HybridBlock.__init__(self, prefix=prefix)
        self.config = config
        self.inference_only = inference_only
        with self.name_scope():
            self.pos_embedding = layers.PositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                             num_embed=self.config.model_size,
                                                             max_seq_len=self.config.max_seq_len_target,
                                                             prefix=C.TARGET_POSITIONAL_EMBEDDING_PREFIX,
                                                             scale_up_input=True,
                                                             scale_down_positions=False)
            self.autoregressive_bias = transformer.AutoRegressiveBias(prefix="autoregressive_bias_")

            self.layers = mx.gluon.nn.HybridSequential()
            for i in range(config.num_layers):
                self.layers.add(transformer.TransformerDecoderBlock(config, prefix="%d_" % i, dtype=dtype,
                                                                    inference_only=self.inference_only))

            self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                     dropout=config.dropout_prepost,
                                                                     prefix="final_process_",
                                                                     num_hidden=self.config.model_size)

    def state_structure(self) -> str:
        """
        Returns the structure of states used for manipulation of the states.
        Each state is either labeled 's' for step, 'b' for source_mask, 'd' for decoder, or 'e' for encoder.
        """
        structure = ''
        if self.inference_only:
            structure += C.STEP_STATE + C.BIAS_STATE + C.ENCODER_STATE * self.config.num_layers
        else:
            structure += C.STEP_STATE + C.ENCODER_STATE + C.BIAS_STATE

        total_num_states = sum(layer.num_state_tensors for layer in self.layers)
        structure += C.DECODER_STATE * total_num_states

        return structure

    def init_state_from_encoder(self,
                                encoder_outputs: mx.nd.NDArray,
                                encoder_valid_length: Optional[mx.nd.NDArray] = None,
                                target_embed: Optional[mx.nd.NDArray] = None) -> List[mx.nd.NDArray]:
        """
        Returns the initial states given encoder output. States for teacher-forced training are encoder outputs
        and a valid length mask for encoder outputs.
        At inference, this method returns the following state tuple:
        valid length bias, step state,
        [projected encoder attention keys, projected encoder attention values] * num_layers,
        [autoregressive state dummies] * num_layers.

        :param encoder_outputs: Encoder outputs. Shape: (batch, source_length, encoder_dim).
        :param encoder_valid_length: Valid lengths of encoder outputs. Shape: (batch,).
        :param target_embed: Target-side embedding layer output. Shape: (batch, target_length, target_embedding_dim).
        :return: Initial states.
        """
        if target_embed is None:  # Inference: initial step = 0. Shape: (batch_size, 1)
            steps = mx.nd.zeros_like(encoder_valid_length).expand_dims(axis=1, inplace=True)
        else:  # Training: steps up to target length. Shape: (1, target_length)
            steps = mx.nd.contrib.arange_like(target_embed, axis=1).expand_dims(axis=0, inplace=True)

        if self.inference_only:
            # Encoder projection caching, therefore we don't pass the encoder_outputs
            states = [steps, encoder_valid_length]

            for layer in self.layers:
                enc_att_kv = layer.enc_attention.ff_kv(encoder_outputs)
                states.append(mx.nd.transpose(enc_att_kv, axes=(1, 0, 2)))
        else:
            # NO encoder projection caching
            states = [steps, mx.nd.transpose(encoder_outputs, axes=(1, 0, 2)), encoder_valid_length]

        _batch_size = encoder_outputs.shape[0]
        _ctx = encoder_outputs.context
        _dtype = encoder_outputs.dtype
        dummy_autoregr_states = [mx.nd.zeros(layer.get_states_shape(_batch_size), ctx=_ctx, dtype=_dtype)
                                 for layer in self.layers
                                 for _ in range(layer.num_state_tensors)]

        states += dummy_autoregr_states
        return states

    def decode_seq(self, inputs: mx.nd.NDArray, states: List[mx.nd.NDArray]):
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param inputs: Encoded source: (batch_size, source_encoded_max_length, encoder_depth).
        :param states: List of initial states, as given by init_state_from_encoder().
        :return: Decoder output. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        outputs, _ = self.forward(inputs, states)
        return outputs

    def hybrid_forward(self, F, step_input, states):
        mask = None
        if self.inference_only:
            steps, source_valid_length, *other = states
            source_encoded = None  # use constant pre-computed key value projections from the states
            enc_att_kv = other[:self.config.num_layers]
            autoregr_states = other[self.config.num_layers:]
        else:
            if any(layer.needs_mask for layer in self.layers):
                mask = self.autoregressive_bias(step_input)  # mask: (1, length, length)
            steps, source_encoded, source_valid_length, *autoregr_states = states
            enc_att_kv = [None for _ in range(self.config.num_layers)]

        if any(layer.num_state_tensors > 1 for layer in self.layers):
            # separates autoregressive states by layer
            states_iter = iter(autoregr_states)
            autoregr_states = [list(islice(states_iter, 0, layer.num_state_tensors)) for layer in self.layers]

        # (batch_size * heads, query_length)
        source_valid_length = layers.prepare_source_valid_lengths(F, source_valid_length, step_input,
                                                                  num_heads=self.config.attention_heads)

        # target: (batch_size, length, model_size)
        target = self.pos_embedding(step_input, steps)
        # (length, batch_size, model_size)
        target = F.transpose(target, axes=(1, 0, 2))

        if self.config.dropout_prepost > 0.0:
            target = F.Dropout(data=target, p=self.config.dropout_prepost)

        new_autoregr_states = []
        for layer, layer_autoregr_state, layer_enc_att_kv in zip(self.layers, autoregr_states, enc_att_kv):
            target, new_layer_autoregr_state = layer(target,
                                                     mask,
                                                     source_encoded,
                                                     source_valid_length,
                                                     layer_autoregr_state,
                                                     layer_enc_att_kv)

            new_autoregr_states += [*new_layer_autoregr_state]

        target = self.final_process(target, None)
        target = F.transpose(target, axes=(1, 0, 2))

        # Inference: increment steps by 1 (discarded in training)
        steps = steps + 1

        if self.inference_only:
            # pass in cached encoder states
            encoder_attention_keys_values = states[2:2 + self.config.num_layers]
            new_states = [steps, states[1]] + encoder_attention_keys_values + new_autoregr_states
        else:
            encoder_outputs = states[1]
            encoder_valid_length = states[2]
            new_states = [steps, encoder_outputs, encoder_valid_length] + new_autoregr_states

        return target, new_states

    def get_num_hidden(self):
        return self.config.model_size

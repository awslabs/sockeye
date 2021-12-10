# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Dict, List, Optional, Tuple, Type, Union

import torch as pt

from . import constants as C
from . import layers_pt
from . import transformer_pt
from .transformer_pt import TransformerConfig

logger = logging.getLogger(__name__)
DecoderConfig = Union[TransformerConfig, 'sockeye.transformer.TransformerConfig']  # type: ignore


def pytorch_get_decoder(config: DecoderConfig, inference_only: bool = False) -> 'PyTorchDecoder':
    # TODO: while we still have both transformer.TransformerConfig and transformer_pt.TransformerConfig,
    # this leads to unexpected behaviors when loading models. We can re-introduce once MXNet is removed
    #return PyTorchDecoder.get_decoder(config, inference_only)
    return PyTorchTransformerDecoder(config, inference_only=inference_only)


class PyTorchDecoder(pt.nn.Module):
    """
    Generic decoder interface.
    A decoder needs to implement code to decode a target sequence known in advance (decode_sequence),
    and code to decode a single word given its decoder state (decode_step).
    The latter is typically used for inference graphs in beam search.
    For the inference module to be able to keep track of decoder's states
    a decoder provides methods to return initial states (init_states), state variables and their shapes.
    """

    __registry = {}  # type: Dict[Type[DecoderConfig], Type['PyTorchDecoder']]

    @classmethod
    def register(cls, config_type: Type[DecoderConfig]):
        """
        Registers decoder type for configuration. Suffix is appended to decoder prefix.

        :param config_type: Configuration type for decoder.

        :return: Class decorator.
        """
        def wrapper(target_cls):
            cls.__registry[config_type] = target_cls
            return target_cls

        return wrapper

    @classmethod
    def get_decoder(cls, config: DecoderConfig, inference_only: bool) -> 'PyTorchDecoder':
        """
        Creates decoder based on config type.

        :param config: Decoder config.
        :param inference_only: Create a decoder that is only used for inference.

        :return: Decoder instance.
        """
        config_type = type(config)
        if config_type not in cls.__registry:
            raise ValueError('Unsupported decoder configuration %s' % config_type.__name__)
        decoder_cls = cls.__registry[config_type]
        return decoder_cls(config=config, inference_only=inference_only)  # type: ignore

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def state_structure(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def init_state_from_encoder(self,
                                encoder_outputs: pt.Tensor,
                                encoder_valid_length: Optional[pt.Tensor] = None,
                                target_embed: Optional[pt.Tensor] = None) -> List[pt.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def decode_seq(self, inputs: pt.Tensor, states: List[pt.Tensor]) -> pt.Tensor:
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

    @abstractmethod
    def weights_from_mxnet_block(self, block_mx):
        raise NotImplementedError()


@PyTorchDecoder.register(TransformerConfig)
class PyTorchTransformerDecoder(PyTorchDecoder):
    """
    Transformer decoder as in Vaswani et al, 2017: Attention is all you need.
    In training, computation scores for each position of the known target sequence are computed in parallel,
    yielding most of the speedup.
    At inference time, the decoder block is evaluated again and again over a maximum length input sequence that is
    initially filled with zeros and grows during beam search with predicted tokens. Appropriate masking at every
    time-step ensures correct self-attention scores and is updated with every step.

    :param config: Transformer configuration.
    :param inference_only: Only use the model for inference enabling some optimizations,
                           such as disabling the auto-regressive mask.
    """

    def __init__(self, config: TransformerConfig, inference_only: bool = False) -> None:
        PyTorchDecoder.__init__(self)
        pt.nn.Module.__init__(self)
        self.config = config
        self.inference_only = inference_only
        self.pos_embedding = layers_pt.PyTorchPositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                                   num_embed=self.config.model_size,
                                                                   max_seq_len=self.config.max_seq_len_target,
                                                                   scale_up_input=True,
                                                                   scale_down_positions=False)
        self.autoregressive_mask = transformer_pt.AutoRegressiveMask()

        self.layers = pt.nn.ModuleList(  # using ModuleList because we have additional inputs
            transformer_pt.PyTorchTransformerDecoderBlock(config, inference_only=self.inference_only)
            for _ in range(config.num_layers))

        self.final_process = transformer_pt.PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                           dropout=config.dropout_prepost,
                                                                           num_hidden=self.config.model_size)
        if self.config.dropout_prepost > 0.0:
            self.dropout = pt.nn.Dropout(p=self.config.dropout_prepost, inplace=inference_only)

    def state_structure(self) -> str:
        """
        Returns the structure of states used for manipulation of the states.
        Each state is either labeled 's' for step, 'b' for source_mask, 'd' for decoder, or 'e' for encoder.
        """
        structure = ''
        if self.inference_only:
            structure += C.STEP_STATE + C.MASK_STATE + C.ENCODER_STATE * self.config.num_layers
        else:
            structure += C.STEP_STATE + C.ENCODER_STATE + C.MASK_STATE

        total_num_states = sum(layer.num_state_tensors for layer in self.layers)
        structure += C.DECODER_STATE * total_num_states

        return structure

    def init_state_from_encoder(self,
                                encoder_outputs: pt.Tensor,
                                encoder_valid_length: Optional[pt.Tensor] = None,
                                target_embed: Optional[pt.Tensor] = None) -> List[pt.Tensor]:
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
        source_max_len = encoder_outputs.size()[1]
        if target_embed is None:  # Inference: initial step = 0. Shape: (batch_size, 1)
            steps = pt.zeros_like(encoder_valid_length).unsqueeze(1)
            # (batch * heads, 1, source_max_len)
            source_mask = layers_pt.prepare_source_length_mask(encoder_valid_length, self.config.attention_heads,
                                                               source_max_len)
            # Shape: (batch, heads, 1, src_max_len)
            source_mask = source_mask.view(-1, self.config.attention_heads, 1, source_max_len)
        else:  # Training: steps up to target length. Shape: (1, target_length)
            target_length = target_embed.size()[1]
            steps = pt.arange(0, target_length, device=target_embed.device).unsqueeze(0)
            # (batch * heads, 1, source_max_len)
            source_mask = layers_pt.prepare_source_length_mask(encoder_valid_length, self.config.attention_heads,
                                                               source_max_len)
            source_mask = source_mask.repeat(1, target_length, 1)  # Shape: (batch * heads, trg_max_len, src_max_len)

            # Shape: (batch, heads, trg_max_len, src_max_len)
            source_mask = source_mask.view(-1, self.config.attention_heads, target_length, source_max_len)

        if self.inference_only:
            # Encoder projection caching, therefore we don't pass the encoder_outputs
            states = [steps, source_mask]
            for layer in self.layers:
                enc_att_kv = layer.enc_attention.ff_kv(encoder_outputs).transpose(1, 0)
                states.append(enc_att_kv)
        else:
            # NO encoder projection caching
            states = [steps, encoder_outputs.transpose(1, 0), source_mask]

        _batch_size = encoder_outputs.size()[0]
        _device = encoder_outputs.device
        _dtype = encoder_outputs.dtype
        dummy_autoregr_states = [pt.zeros(layer.get_states_shape(_batch_size), device=_device, dtype=_dtype)
                                 for layer in self.layers
                                 for _ in range(layer.num_state_tensors)]

        states += dummy_autoregr_states
        return states

    def decode_seq(self, inputs: pt.Tensor, states: List[pt.Tensor]) -> pt.Tensor:
        """
        Decodes a sequence of embedded target words and returns sequence of last decoder
        representations for each time step.

        :param inputs: Encoded source: (batch_size, source_encoded_max_length, encoder_depth).
        :param states: List of initial states, as given by init_state_from_encoder().
        :return: Decoder output. Shape: (batch_size, target_embed_max_length, decoder_depth).
        """
        outputs, _ = self.forward(inputs, states)
        return outputs

    def forward(self, step_input: pt.Tensor, states: List[pt.Tensor]) -> Tuple[pt.Tensor, List[pt.Tensor]]:
        target_mask = None
        if self.inference_only:
            steps, source_mask, *other = states
            source_encoded = None  # use constant pre-computed key value projections from the states
            enc_att_kv = other[:self.config.num_layers]
            autoregr_states = other[self.config.num_layers:]
        else:
            if any(layer.needs_mask for layer in self.layers):
                target_mask = self.autoregressive_mask(step_input)  # mask: (length, length)
            steps, source_encoded, source_mask, *autoregr_states = states
            enc_att_kv = [None for _ in range(self.config.num_layers)]

        if any(layer.num_state_tensors > 1 for layer in self.layers):
            # separates autoregressive states by layer
            states_iter = iter(autoregr_states)
            autoregr_states = [list(islice(states_iter, 0, layer.num_state_tensors)) for layer in self.layers]  # type: ignore

        batch, heads, target_max_len, source_max_len = source_mask.size()
        source_mask_view = source_mask.view(batch * heads, target_max_len, source_max_len)

        # target: (batch_size, length, model_size)
        target = self.pos_embedding(step_input, steps)
        # (length, batch_size, model_size)
        target = target.transpose(1, 0)

        if self.config.dropout_prepost > 0.0:
            target = self.dropout(target)

        new_autoregr_states = []  # type: List[pt.Tensor]
        for layer, layer_autoregr_state, layer_enc_att_kv in zip(self.layers, autoregr_states, enc_att_kv):
            target, new_layer_autoregr_state = layer(target=target,
                                                     target_mask=target_mask,
                                                     source=source_encoded,
                                                     source_mask=source_mask_view,
                                                     autoregr_states=layer_autoregr_state,
                                                     enc_att_kv=layer_enc_att_kv)

            new_autoregr_states += [*new_layer_autoregr_state]

        target = self.final_process(target)
        target = target.transpose(1, 0)

        # Inference: increment steps by 1 (discarded in training)
        steps = steps + 1

        if self.inference_only:
            # pass in cached encoder states
            encoder_attention_keys_values = states[2:2 + self.config.num_layers]
            new_states = [steps, states[1]] + encoder_attention_keys_values + new_autoregr_states
        else:
            new_states = [steps, states[1], states[2]] + new_autoregr_states

        return target, new_states

    def get_num_hidden(self):
        return self.config.model_size

    def weights_from_mxnet_block(self, block_mx: 'TransformerDecoder'):  # type: ignore
        self.pos_embedding.weights_from_mxnet_block(block_mx.pos_embedding)
        for i, l in enumerate(self.layers):
            l.weights_from_mxnet_block(block_mx.layers[i])
        self.final_process.weights_from_mxnet_block(block_mx.final_process)


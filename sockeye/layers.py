# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import math
from typing import Dict, Optional, Tuple, Union, List, Iterator, Sequence
from abc import ABC, abstractmethod

import mxnet as mx
import numpy as np

from .config import Config
from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


def activation(data: mx.sym.Symbol, act_type: str) -> mx.sym.Symbol:
    """
    Apply custom or standard activation.

    Custom activation types include:
     - Swish-1, also called Sigmoid-Weighted Linear Unit (SiLU): Ramachandran et
       al. (https://arxiv.org/pdf/1710.05941.pdf), Elfwing et al.
       (https://arxiv.org/pdf/1702.03118.pdf)
     - Gaussian Error Linear Unit (GELU): Hendrycks and Gimpel
       (https://arxiv.org/pdf/1606.08415.pdf)

    :param data: input Symbol of any shape.
    :param act_type: Type of activation.
    :return: output Symbol with same shape as input.
    """
    # TODO: Contribute these to MXNet?  For now it appears that registered activation types must be implemented in C++.
    if act_type == C.SWISH1:
        return data * mx.sym.Activation(data, act_type="sigmoid")
    elif act_type == C.GELU:
        # Approximation of x * gaussian_cdf(x) used by Hendrycks and Gimpel
        return 0.5 * data * (1 + mx.sym.Activation((math.sqrt(2 / math.pi) * (data + (0.044715 * (data**3)))),
                                                   act_type="tanh"))
    elif act_type == C.NO_ACTIVATION:
        return data
    else:
        return mx.sym.Activation(data, act_type=act_type)


class Layer(ABC):

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the layer if such a restriction exists.
        """
        return None

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this layer.
        """
        raise NotImplementedError()


class EncoderLayer(Layer):
    """
    Generic encoder layer interface for a layer which takes the sequence of previous hidden states and sequence lengths
    and produces a new sequence hidden states, potentially changing the sequence length.
    """

    @abstractmethod
    def encode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        att_dict: Optional[Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param source_encoded: Input data of size (batch_size, seq_len, num_hidden).
        :param source_encoded_lengths: Vector with sequence lengths of size (batch_size,).
        :param source_encoded_max_length: Maximum sequence length.
        :param att_dict: A dictionary of attention matrices used for visualization.
            Each matrix must be of size (batch_size, source_length, source_length).
        :return: Encoded versions of the input data, the new sequence lenghts and the new maximum length.
        """
        pass

    def att_names(self) -> List[str]:
        """Names of attention matrices produced by this layer."""
        return []

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        :return: The size of the encoded sequence.
        """
        return seq_len


class DecoderLayer(Layer):
    """
    Generic decoder layer interface.
    """

    # TODO: do we need all of the arguments?
    @abstractmethod
    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Run the layer on the entire sequence.

        :param source_encoded: Encoded source layer: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target_encoded: Input data. Shape: (batch_size, seq_len, num_hidden).
        :param target_encoded_lengths: Vector with sequence lengths. Shape: (batch_size,).
        :param target_encoded_max_length: Maximum sequence length.
        :param target_autoregressive_bias: The auto-regressive bias.
        :return: hidden_data, Shape: (batch_size, seq_len, num_hidden).
        """
        pass

    # TODO: potentially define a DecoderAttDict class with source and self_att members which are dicts?!
    @abstractmethod
    def decode_step(self,
                    step: int,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    target: mx.sym.Symbol,
                    states: Sequence[mx.sym.Symbol],
                    att_dict: Dict[str, Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        """
        Run the decoder layer for a single position given the current step, the previous embedded target word,
        and previous decoder layer states.

        :param step: Global step of inference procedure, starts with 1.
        :param source_encoded: Encoded source layer: (source_encoded_max_length, batch_size, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :param target: Shape: (batch_size, input_num_hidden).
        :param states: Arbitrary layer states.
        :param att_dict: A dictionary of attention matrices used for visualization with separate entries for source and
        self attention {'self': Dict, 'source': Dict}. Each source attention matrix must be of size
        (batch_size, 1, source_length) and each self-attention of size (batch_size, 1, step).
        :return: Step result of the layer (batch_size, num_hidden) and a list of new layer states.
        """
        pass

    def att_names(self) -> List[str]:
        """Names of attention matrices produced by this layer."""
        return []

    def self_att_names(self) -> List[str]:
        """Names of self attention matrices produced by this layer."""
        return []

    def reset(self):
        """
        Reset decoder method. Used for inference.
        """
        pass

    def num_states(self, step: int) -> int:
        """
        The number of input states at the given step.
        :param step:
        :return:
        """
        return 0

    def state_variables(self, step: int) -> Sequence[mx.sym.Symbol]:
        """
        Returns the list of symbolic variables for this decoder to be used during inference.

        :param step: Current target sequence length.
        :return: List of symbolic variables.
        """
        return []

    def init_states(self,
                    batch_size: int,
                    source_encoded: Sequence[mx.sym.Symbol],
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> Sequence[mx.sym.Symbol]:
        """
        Returns a list of symbolic states that represent the initial states of this decoder.
        Used for inference.

        :param batch_size: The batch size.
        :param source_encoded: Encoded source. Shape: (batch_size, source_encoded_max_length, encoder_depth).
        :param source_encoded_lengths: Lengths of encoded source sequences. Shape: (batch_size,).
        :param source_encoded_max_length: Size of encoder time dimension.
        :return: List of symbolic initial states.
        """
        return []

    def state_shapes(self,
                     batch_size: int,
                     target_max_length: int,
                     source_encoded_max_length: int,
                     source_encoded_num_hidden: int) -> List[mx.io.DataDesc]:
        """
        Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
        Used for inference.

        :param batch_size: Batch size during inference.
        :param source_encoded_max_length: Size of encoder time dimension.
        :param source_encoded_num_hidden: Depth of encoded source.
        :return: List of shape descriptions.
        """
        return []


class LayerConfig(Config):
    """
    A layer config object used for serializing layer parameters. Each layer config object also defines how an encoder
    or decoder layer are created from the given parameters.
    """

    @abstractmethod
    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        pass

    @abstractmethod
    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        pass


class SharedEncoderDecoderLayer(DecoderLayer, EncoderLayer):
    """
    A layer which does not depend on the source hidden states. On the target side it will be applied on the target
    hidden states and on the source side on the source hidden states. Using this class as a base class only the combined
    `process_sequence` method needs to be implemented.
    """

    @abstractmethod
    def process_sequence(self,
                         data: mx.sym.Symbol,
                         lengths: mx.sym.Symbol,
                         max_length: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Process either the encoder or the decoder hidden states.
        :param data: Encoded source layer: (source_encoded_max_length, batch_size, encoder_depth).
        :param lengths: Lengths of hidden sequences. Shape: (batch_size,).
        :param max_length: Maximum length.
        :return: Encoded versions of the input data, the new sequence lenghts and the new maximum length.
        """
        pass

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        return self.process_sequence(source_encoded, source_encoded_lengths, source_encoded_max_length)

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        new_target_encoded, new_target_encoded_lengths, new_target_encoded_max_length = self.process_sequence(
            target_encoded, target_encoded_lengths, target_encoded_max_length)
        assert new_target_encoded_lengths is target_encoded_lengths, C.SEQUENCE_LENGTH_MUST_NOT_CHANGE_MSG
        assert new_target_encoded_max_length == target_encoded_max_length, C.SEQUENCE_LENGTH_MUST_NOT_CHANGE_MSG
        return new_target_encoded


def layers_with_states_iter(
        layers: List[DecoderLayer],
        step: int,
        layer_states_flat: Sequence[mx.sym.Symbol]) -> Iterator[Tuple[DecoderLayer, Sequence[mx.sym.Symbol]]]:
    """
    A generator for layers and corresponding layer states from a flat list of all layer states.
    :param layers: A list of layers.
    :param step: The current decoder step.
    :param layer_states_flat: A flat list of layer states across all layers, e.g. [l1_state1, l1_state2, l2_state1, ..].
    :return: A generator of tuples of decoder layers and their states.
    """
    state_idx = 0
    for layer in layers:
        if layer.num_states(step) != 0:
            layer_states = layer_states_flat[state_idx:state_idx + layer.num_states(step)]
            state_idx += layer.num_states(step)
        else:
            layer_states = []
        yield layer, layer_states


class EncoderLayerChain(EncoderLayer):

    def __init__(self, layers: List[EncoderLayer]) -> None:
        assert len(layers) >= 1, "At least one layer needed in layer chain."
        self.layers = layers

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        for layer in self.layers:
            source_encoded, source_encoded_lengths, source_encoded_max_length = layer.encode_sequence(
                source_encoded, source_encoded_lengths, source_encoded_max_length, att_dict)
        return source_encoded, source_encoded_lengths, source_encoded_max_length

    def att_names(self):
        att_names = []
        for layer in self.layers:
            att_names.extend(layer.att_names())
        return att_names

    def get_num_hidden(self) -> int:
        return self.layers[-1].get_num_hidden()


class NestedDecoderLayer(DecoderLayer):
    """
    A decoder layer which combines several other decoder sub-layers.
    """

    @property
    @abstractmethod
    def layers(self) -> List[DecoderLayer]:
        pass

    def layers_with_states_iter(self,
                                step: int,
                                layer_states_flat: Sequence[mx.sym.Symbol]) -> Iterator[Tuple[DecoderLayer,
                                                                                          Sequence[mx.sym.Symbol]]]:
        return layers_with_states_iter(self.layers, step, layer_states_flat)

    def att_names(self) -> List[str]:
        att_names = []
        for layer in self.layers:
            att_names.extend(layer.att_names())
        return att_names

    def self_att_names(self) -> List[str]:
        att_names = []
        for layer in self.layers:
            att_names.extend(layer.self_att_names())
        return att_names

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def num_states(self, step) -> int:
        return sum(layer.num_states(step) for layer in self.layers)

    def state_variables(self, step: int) -> Sequence[mx.sym.Symbol]:
        return [var for layer in self.layers for var in layer.state_variables(step)]

    def state_shapes(self,
                     batch_size: int,
                     step: int,
                     source_encoded_max_length: int,
                     source_encoded_num_hidden: int) -> List[mx.io.DataDesc]:
        return [state_shape for layer in self.layers for state_shape in layer.state_shapes(batch_size,
                                                                                           step,
                                                                                           source_encoded_max_length,
                                                                                           source_encoded_num_hidden)]

    def init_states(self,
                    batch_size: int,
                    source_encoded: Sequence[mx.sym.Symbol],
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> Sequence[mx.sym.Symbol]:
        return [init_state for layer in self.layers for init_state in layer.init_states(batch_size,
                                                                                        source_encoded,
                                                                                        source_encoded_lengths,
                                                                                        source_encoded_max_length)]

    def get_max_seq_len(self) -> Optional[int]:
        return min((layer.get_max_seq_len() for layer in self.layers if layer.get_max_seq_len() is not None),
                   default=None)


class DecoderLayerChain(NestedDecoderLayer):

    def __init__(self, layers: List[DecoderLayer]) -> None:
        assert len(layers) >= 1, "At least one layer needed in layer chain."
        self._layers = layers

    @property
    def layers(self) -> List[DecoderLayer]:
        return self._layers

    def decode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol, target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        for layer in self.layers:
            target_encoded = layer.decode_sequence(source_encoded,
                                                   source_encoded_lengths,
                                                   source_encoded_max_length,
                                                   target_encoded,
                                                   target_encoded_lengths,
                                                   target_encoded_max_length,
                                                   target_autoregressive_bias)

        return target_encoded

    def decode_step(self, step: int, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int, target: mx.sym.Symbol, layer_states_flat: Sequence[mx.sym.Symbol],
                    att_dict) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        new_layer_states_flat = []  # type: List[mx.sym.Symbol]
        for layer, layer_states in self.layers_with_states_iter(step, layer_states_flat):
            target, new_layer_states = layer.decode_step(step, source_encoded, source_encoded_lengths,
                                                         source_encoded_max_length, target, layer_states, att_dict)
            new_layer_states_flat.extend(new_layer_states)

        return target, new_layer_states_flat

    def get_num_hidden(self) -> int:
        return self._layers[-1].get_num_hidden()


class StatelessBlock(ABC):
    """
    A block which does not require to keep any state during inference time so that we are able to call it one
    timestep at a time. This means that the block can be run as

        concat(block(data[:,t,:]) for t in range(seq_len), dim=1)

    Namely, at time t we only need the data point at time t from the previous layer.
    """

    @abstractmethod
    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        """
        Compute the block for the entire sequence.

        :param data: Input data, Shape: (batch_size, seq_len, input_num_hidden).
        :param lengths: Optional vector with sequence lengths, Shape: (batch_size,).
        :param max_length: Optional maximum sequence length.
        :return: new data (batch_size, seq_len, num_hidden).
        """
        pass

    def step(self, step: int, data: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Compute the block for a single time step.

        :param step: The current step.
        :param data: Data for a single time step. Shape: (batch_size, 1, input_num_hidden).
        :return: Shape: (batch_size, 1, num_hidden).
        """
        return self.__call__(data)

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this block.
        """
        pass

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the block if such a restriction exists.
        """
        return None


class StatelessBlockLayer(SharedEncoderDecoderLayer):
    """
    A stateless block layer which can act as both a encoder or a decoder layer, applying the block to either the
    source or target hidden state.
    """

    def __init__(self, block: StatelessBlock) -> None:
        self.block = block

    def process_sequence(self,
                         data: mx.sym.Symbol,
                         lengths: mx.sym.Symbol,
                         max_length: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        return self.block(data, lengths, max_length), lengths, max_length

    def get_num_hidden(self) -> int:
        return self.block.get_num_hidden()

    def decode_step(self,
                    step: int,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    target: mx.sym.Symbol,
                    states: Sequence[mx.sym.Symbol],
                    att_dict: Dict[str, Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        # (batch_size, 1, num_hidden)
        target = mx.sym.expand_dims(target, axis=1)

        # (batch_size, 1, num_hidden_block)
        hidden = self.block.step(step, target)

        # (batch_size, num_hidden_block)
        hidden = mx.sym.reshape(hidden, shape=(0, -1))
        return hidden, []

    def get_max_seq_len(self):
        return self.block.get_max_seq_len()


class FeedForwardBlock(StatelessBlock):
    """
    Position-wise feed-forward network with activation and dropout.
    """

    def __init__(self,
                 num_hidden: int,
                 dropout: float = 0.0,
                 act_type: str = C.RELU,
                 prefix: str = "") -> None:
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.prefix = prefix
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)

    def _pre_activation_num_hidden(self):
        if self.act_type == C.GLU:
            return 2 * self.num_hidden
        else:
            return self.num_hidden

    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        """
        Apply the feed-forward layer.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        h = mx.sym.FullyConnected(data=data, num_hidden=self._pre_activation_num_hidden(),
                                  weight=self.w_i2h, bias=self.b_i2h,
                                  flatten=False, name=self.prefix + "ff")

        if self.act_type == C.GLU:
            # GLU
            # two times: (batch_size, seq_len, num_hidden)

            # pylint: disable=unbalanced-tuple-unpacking
            gate_a, gate_b = mx.sym.split(h, num_outputs=2, axis=2)
            # (batch_size, seq_len, num_hidden)
            h = mx.sym.broadcast_mul(gate_a,
                                     mx.sym.Activation(data=gate_b, act_type="sigmoid"))
        else:
            h = activation(h, act_type=self.act_type)
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        return h

    def get_num_hidden(self) -> int:
        return self.num_hidden


class FeedForwardLayerConfig(LayerConfig):

    def __init__(self,
                 num_hidden: int,
                 dropout: float,
                 act_type: str = C.RELU) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.act_type = act_type
        self.dropout = dropout

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return self.create_block_layer(prefix)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return self.create_block_layer(prefix)

    def create_block_layer(self, prefix: str) -> StatelessBlockLayer:
        return StatelessBlockLayer(FeedForwardBlock(self.num_hidden,
                                                    self.dropout,
                                                    self.act_type,
                                                    prefix + "ff_"))


class LinearBlock(StatelessBlock):
    """
    Linear projection followed by dropout.
    """

    def __init__(self,
                 num_hidden: int,
                 dropout: float,
                 no_bias: bool,
                 prefix: str) -> None:
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.no_bias = no_bias
        self.prefix = prefix
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = None if no_bias else mx.sym.Variable('%si2h_bias' % prefix)

    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        """
        Apply the linear projection.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        bias = None if self.no_bias else self.b_i2h
        h = mx.sym.FullyConnected(data=data, num_hidden=self.num_hidden, weight=self.w_i2h, bias=bias,
                                  no_bias=self.no_bias, flatten=False)
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        return h

    def get_num_hidden(self) -> int:
        return self.num_hidden


class LinearLayerConfig(LayerConfig):

    def __init__(self, num_hidden: int, dropout: float, no_bias=False) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.no_bias = no_bias

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return self.create_block_layer(prefix)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return self.create_block_layer(prefix)

    def create_block_layer(self, prefix: str) -> StatelessBlockLayer:
        return StatelessBlockLayer(LinearBlock(self.num_hidden, self.dropout, no_bias=self.no_bias,
                                               prefix=prefix + "linear_"))


class ActivationBlock(StatelessBlock):

    def __init__(self,
                 act_type: str,
                 num_hidden: int) -> None:
        self.act_type = act_type
        self.num_hidden = num_hidden

    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        """
        Apply the activation function.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        data = activation(data, act_type=self.act_type)
        return data

    def get_num_hidden(self) -> int:
        return self.num_hidden


class ActivationLayerConfig(LayerConfig):

    def __init__(self, act_type: str = C.RELU) -> None:
        super().__init__()
        self.act_type = act_type

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return StatelessBlockLayer(ActivationBlock(num_hidden=input_num_hidden, act_type=self.act_type))

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return StatelessBlockLayer(ActivationBlock(num_hidden=input_num_hidden, act_type=self.act_type))


class DropoutBlock(StatelessBlock):
    """
    Position-wise feed-forward network with activation.
    """

    def __init__(self,
                 dropout: float,
                 num_hidden: int) -> None:
        self.dropout = dropout
        self.num_hidden = num_hidden

    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        """
        Position-wise feed-forward network with activation.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        if self.dropout > 0.0:
            data = mx.sym.Dropout(data, p=self.dropout)
        return data

    def get_num_hidden(self) -> int:
        return self.num_hidden


class DropoutLayerConfig(LayerConfig):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return StatelessBlockLayer(DropoutBlock(num_hidden=input_num_hidden, dropout=self.dropout))

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return StatelessBlockLayer(DropoutBlock(num_hidden=input_num_hidden, dropout=self.dropout))


class LearnedAdditivePositionalEmbeddings(StatelessBlock):

    def __init__(self,
                 num_embed: int,
                 dropout: float,
                 max_seq_len: int,
                 prefix: str) -> None:
        self.num_embed = num_embed
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.prefix = prefix
        self.embed_weight = mx.sym.Variable(prefix + "weight")

    def __call__(self, data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        assert max_length is not None, "max_length needed for positional embeddings."
        # (1, source_seq_len)
        positions = mx.sym.expand_dims(data=mx.sym.arange(start=0, stop=max_length, step=1), axis=0)

        # (1, source_seq_len, num_embed)
        pos_embedding = mx.sym.Embedding(data=positions,
                                         input_dim=self.max_seq_len,
                                         weight=self.embed_weight,
                                         output_dim=self.num_embed,
                                         name=self.prefix + "pos_embed")
        data = mx.sym.broadcast_add(data, pos_embedding, name="%s_add" % self.prefix)
        if self.dropout > 0.0:
            data = mx.sym.Dropout(data, p=self.dropout)
        return data

    def step(self, step: int, data: mx.sym.Symbol) -> mx.sym.Symbol:
        position = step - 1
        position = position * mx.sym.reshape(
            mx.sym.slice_axis(mx.sym.reshape(mx.sym.ones_like(data), shape=(0, -1)), axis=1, begin=0, end=1),
            shape=(-1))
        pos_embedding = mx.sym.Embedding(data=position,
                                         input_dim=self.max_seq_len,
                                         weight=self.embed_weight,
                                         output_dim=self.num_embed,
                                         name=self.prefix + "pos_embed")
        pos_embedding = mx.sym.expand_dims(pos_embedding, axis=1)
        return mx.sym.broadcast_add(data, pos_embedding, name="%s_add" % self.prefix)

    def get_num_hidden(self) -> int:
        return self.num_embed

    def get_max_seq_len(self):
        return self.max_seq_len


class LearnedPositionalEmbeddingsLayerConfig(LayerConfig):

    def __init__(self,
                 num_embed: int,
                 dropout: float,
                 max_seq_len: int) -> None:
        super().__init__()
        self.num_embed = num_embed
        self.dropout = dropout
        self.max_seq_len = max_seq_len

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return self.create_layer(prefix)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return self.create_layer(prefix)

    def create_layer(self, prefix: str) -> StatelessBlockLayer:
        return StatelessBlockLayer(LearnedAdditivePositionalEmbeddings(num_embed=self.num_embed,
                                                                       dropout=self.dropout,
                                                                       max_seq_len=self.max_seq_len,
                                                                       prefix=prefix + "pos_embed_"))


# TODO: share the code with the encoder!
class AdditiveSinCosPositionalEmbeddings(StatelessBlock):
    """
    Takes an encoded sequence and adds fixed positional embeddings as in Vaswani et al, 2017 to it.

    :param num_embed: Embedding size.
    :param prefix: Name prefix for symbols of this encoder.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    """

    def __init__(self,
                 num_embed: int,
                 dropout: float,
                 prefix: str,
                 scale_up_input: bool,
                 scale_down_positions: bool) -> None:
        utils.check_condition(num_embed % 2 == 0, "Positional embeddings require an even embedding size it "
                                                  "is however %d." % num_embed)
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions
        self.num_embed = num_embed
        self.dropout = dropout
        self.prefix = prefix

    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None) -> mx.sym.Symbol:
        assert max_length is not None, "max_length needed for positional embeddings."
        # add positional embeddings to data
        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        positions = mx.sym.BlockGrad(mx.symbol.Custom(length=max_length,
                                                      depth=self.num_embed,
                                                      name="%spositional_encodings" % self.prefix,
                                                      op_type='positional_encodings'))

        if self.scale_down_positions:
            positions = positions * (self.num_embed ** -0.5)

        embedding = mx.sym.broadcast_add(data, positions)
        if self.dropout > 0.0:
            embedding = mx.sym.Dropout(embedding, p=self.dropout)
        return embedding

    def step(self, step: int, data: mx.sym.Symbol) -> mx.sym.Symbol:
        position = step - 1
        # (batch_size, num_hidden) -> (batch_size,)
        positions = position * mx.sym.reshape(
            mx.sym.slice_axis(mx.sym.reshape(mx.sym.ones_like(data), shape=(0, -1)), axis=1, begin=0, end=1),
            shape=(-1))
        # (batch_size, 1)
        positions = mx.sym.expand_dims(positions, axis=1)
        # (num_embed,)
        channels = mx.sym.arange(0, self.num_embed // 2)
        # (1, num_embed,)
        scaling = mx.sym.expand_dims(1. / mx.sym.pow(10000, (2 * channels) / self.num_embed), axis=0)

        # (batch_size, num_embed/2)
        scaled_positions = mx.sym.dot(positions, scaling)

        sin = mx.sym.sin(scaled_positions)
        cos = mx.sym.cos(scaled_positions)

        # (batch_size, num_embed)
        pos_embedding = mx.sym.concat(sin, cos, dim=1)

        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        if self.scale_down_positions:
            pos_embedding = pos_embedding * (self.num_embed ** -0.5)

        # (batch_size, 1, num_embed)
        pos_embedding = mx.sym.expand_dims(pos_embedding, axis=1)
        return mx.sym.broadcast_add(data, pos_embedding, name="%s_add" % self.prefix)

    def get_num_hidden(self) -> int:
        return self.num_embed


class SinCosPositionalEmbeddingsLayerConfig(LayerConfig):

    def __init__(self,
                 num_embed: int,
                 dropout: float = 0.0,
                 scale_inputs: bool = True) -> None:
        super().__init__()
        self.num_embed = num_embed
        self.dropout = dropout
        if scale_inputs:
            # Transformer default (seems to work better for the transformer)
            self.scale_up_input = True
            self.scale_down_positions = False
        else:
            self.scale_up_input = False
            self.scale_down_positions = True

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return self.create_layer(prefix)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return self.create_layer(prefix)

    def create_layer(self, prefix) -> StatelessBlockLayer:
        return StatelessBlockLayer(AdditiveSinCosPositionalEmbeddings(num_embed=self.num_embed,
                                                                      dropout=self.dropout,
                                                                      prefix=prefix + "pos_embed_",
                                                                      scale_up_input=self.scale_up_input,
                                                                      scale_down_positions=self.scale_down_positions))


def get_autoregressive_bias(max_length: int, name: str) -> mx.sym.Symbol:
    """
    Returns bias/mask to ensure position i can only attend to positions <i.

    :param max_length: Sequence length.
    :param name: Name of symbol.
    :return: Bias symbol of shape (1, max_length, max_length).
    """
    return mx.sym.BlockGrad(mx.symbol.Custom(length=max_length,
                                             name=name,
                                             op_type='auto_regressive_bias'))


class AutoRegressiveBias(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, length, length) with cells above the main diagonal
    set to a large negative value, e.g.
    length=4

    0 1 1 1
    0 0 1 1   * LARGE_NEGATIVE_VALUE
    0 0 0 1
    0 0 0 0
    """

    def __init__(self, length: int, dtype:str, ctx: mx.Context) -> None:
        super().__init__()
        self.bias = self.get_bias(length, dtype, ctx)

    @staticmethod
    def get_bias(length: int, dtype: str, ctx: mx.Context):
        # matrix with lower triangle and main diagonal set to 0, upper triangle set to 1
        upper_triangle = np.triu(np.ones((length, length), dtype=dtype), k=1)
        # (1, length, length)
        bias = -C.LARGE_VALUES[dtype] * np.reshape(upper_triangle, (1, length, length))
        return mx.nd.array(bias, ctx=ctx)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("auto_regressive_bias")
class AutoRegressiveBiasProp(mx.operator.CustomOpProp):

    def __init__(self, length: str, dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.length = int(length)
        self.dtype = dtype

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.length)], []

    def infer_type(self, in_type):
        return [], [np.dtype(self.dtype).type], []

    def create_operator(self, ctx, shapes, dtypes):
        return AutoRegressiveBias(length=self.length, dtype=self.dtype, ctx=ctx)


class MultiHeadSourceAttentionDecoderLayer(DecoderLayer):

    def __init__(self, num_hidden, att_num_hidden, heads: int, dropout: float, dropout_attention: float,
                 prefix: str = "") -> None:
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.att_num_hidden = att_num_hidden if att_num_hidden is not None else num_hidden
        self.dropout = dropout
        self.att = MultiHeadAttention(prefix=prefix, heads=heads, depth_att=self.att_num_hidden, depth_out=num_hidden,
                                      dropout=dropout_attention)

    def att_names(self):
        return [self.prefix + ("h%d" % i) for i in range(1, self.att.heads+1)]

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        contexts, probs = self.att(target_encoded, source_encoded, source_encoded_lengths)

        if self.dropout > 0.0:
            contexts = mx.sym.Dropout(contexts, p=self.dropout)
        return contexts

    def decode_step(self,
                    step: int,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    target: mx.sym.Symbol,
                    states: Sequence[mx.sym.Symbol],
                    att_dict: Dict[str, Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        # (batch_size, 1, num_hidden)
        target = mx.sym.expand_dims(target, axis=1)

        context, probs = self.att(target, source_encoded, source_encoded_lengths)

        # probs is of shape (batch, heads, 1, target_length)
        for head_idx, head_prob in enumerate(mx.sym.split(probs, axis=1, squeeze_axis=True,
                                                          num_outputs=self.att.heads), 1):
            name = self.prefix + ("h%d" % head_idx)
            att_dict["source"][name] = mx.sym.identity(head_prob, name=name)

        # (batch_size, num_hidden)
        context = mx.sym.reshape(context, shape=(0, -1))
        return context, []

    def get_num_hidden(self) -> int:
        return self.num_hidden


class MultiHeadSourceAttentionLayerConfig(LayerConfig):
    def __init__(self,
                 heads: int = 8,
                 dropout: float = 0.0,
                 dropout_attention: Optional[float] = None,
                 num_hidden: int = None,
                 att_num_hidden: Optional[int] = None) -> None:
        super().__init__()
        assert num_hidden is not None
        self.num_hidden = num_hidden
        self.att_num_hidden = att_num_hidden if att_num_hidden is not None else num_hidden
        self.dropout = dropout
        self.dropout_attention = dropout_attention if dropout_attention is not None else dropout
        self.heads = heads

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        raise NotImplementedError("Source attention is only availabe on the decoder side.")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return MultiHeadSourceAttentionDecoderLayer(num_hidden=self.num_hidden,
                                                    att_num_hidden=self.att_num_hidden,
                                                    heads=self.heads,
                                                    dropout=self.dropout,
                                                    dropout_attention=self.dropout_attention,
                                                    prefix=prefix + "mh_att_")


class MultiHeadSelfAttentionEncoderLayer(EncoderLayer):

    def __init__(self, num_hidden, att_num_hidden: Optional[int], heads: int, dropout: float, dropout_attention: float,
                 prefix: str) -> None:
        if att_num_hidden is None:
            att_num_hidden = num_hidden
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.att = MultiHeadSelfAttention(prefix=prefix,
                                          dropout=dropout_attention,
                                          heads=heads,
                                          depth_att=att_num_hidden,
                                          depth_out=num_hidden)

    def encode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        att_dict: Optional[Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        contexts, probs = self.att(source_encoded, source_encoded_lengths)

        if att_dict is not None:
            # probs is of shape (batch, heads, source_length, source_length)
            for head_idx, head_prob in enumerate(mx.sym.split(probs, axis=1, squeeze_axis=True,
                                                              num_outputs=self.att.heads), 1):
                name = self.prefix + ("h%d" % head_idx)
                att_dict[name] = mx.sym.identity(head_prob, name=name)

        if self.dropout > 0.0:
            contexts = mx.sym.Dropout(contexts, p=self.dropout)
        return contexts, source_encoded_lengths, source_encoded_max_length

    def att_names(self):
        return [self.prefix + ("h%d" % i) for i in range(1, self.att.heads+1)]

    def get_num_hidden(self) -> int:
        return self.num_hidden


class MultiHeadSelfAttentionDecoderLayer(DecoderLayer):

    def __init__(self, num_hidden, att_num_hidden: int, heads: int, dropout: float, dropout_attention: float,
                 prefix: str = "") -> None:
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.att_num_hidden = att_num_hidden if att_num_hidden is not None else num_hidden
        self.dropout = dropout
        self.att = MultiHeadSelfAttention(prefix=prefix,
                                          dropout=dropout_attention,
                                          heads=heads,
                                          depth_att=self.att_num_hidden,
                                          depth_out=num_hidden)

    def decode_sequence(self,
                        source_encoded: Sequence[mx.sym.Symbol],
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:

        contexts, _ = self.att(target_encoded, bias=target_autoregressive_bias)
        if self.dropout > 0.0:
            contexts = mx.sym.Dropout(contexts, p=self.dropout)
        return contexts

    def decode_step(self,
                    step: int,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    target: mx.sym.Symbol,
                    states: Sequence[mx.sym.Symbol],
                    att_dict: Dict[str, Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        target = mx.sym.expand_dims(target, axis=1)

        if step > 1:
            prev_keys, prev_values = states
            cache = {'k': prev_keys, 'v': prev_values}
        else:
            cache = {'k': None, 'v': None}

        context, probs = self.att(target, cache=cache)

        new_states = [cache['k'], cache['v']]  # type: Sequence[mx.sym.Symbol]

        # Fill the attention dictionary
        # probs has shape (batch, heads, 1, target_length)
        for head_idx, head_prob in enumerate(mx.sym.split(probs, axis=1, squeeze_axis=True,
                                                          num_outputs=self.att.heads), 1):
            name = self.prefix + ("h%d" % head_idx)
            att_dict["self"][name] = mx.sym.identity(head_prob, name=name)

        # context: (batch_size, 1, dv) -> (batch_size, num_hidden)
        context = mx.sym.reshape(context, shape=(0, -1))

        # note: self.att has updated the cache

        return context, new_states

    def self_att_names(self):
        return [self.prefix + ("h%d" % i) for i in range(1, self.att.heads+1)]

    def num_states(self, step):
        if step == 1:
            return 0
        else:
            return 2

    def state_variables(self, step: int):
        if step == 1:
            return []
        else:
            return [mx.sym.Variable("%sself_att_state0" % self.prefix),
                    mx.sym.Variable("%sself_att_state1" % self.prefix)]

    def state_shapes(self,
                     batch_size: int,
                     step: int,
                     source_encoded_max_length: int,
                     source_encoded_num_hidden: int):
        if step == 1:
            return []
        else:
            return [mx.io.DataDesc(name="%sself_att_state0" % self.prefix,
                                   shape=(batch_size,
                                          (step - 1),
                                          self.att_num_hidden),
                                   layout=C.BATCH_MAJOR),
                    mx.io.DataDesc(name="%sself_att_state1" % self.prefix,
                                   shape=(batch_size,
                                          (step - 1),
                                          self.att_num_hidden),
                                   layout=C.BATCH_MAJOR)]

    def init_states(self,
                    batch_size: int,
                    source_encoded: Sequence[mx.sym.Symbol],
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int):
        return []

    def get_num_hidden(self) -> int:
        return self.num_hidden


class MultiHeadSelfAttentionLayerConfig(LayerConfig):

    def __init__(self,
                 heads: int = 8,
                 dropout: float = 0.0,
                 dropout_attention: Optional[float] = None,
                 num_hidden: int = None,
                 att_num_hidden: Optional[int] = None) -> None:
        super().__init__()
        assert num_hidden is not None, "num_hidden required"
        self.num_hidden = num_hidden
        self.att_num_hidden = att_num_hidden if att_num_hidden is not None else num_hidden
        self.heads = heads
        self.dropout = dropout
        self.dropout_attention = dropout_attention if dropout_attention is not None else dropout

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        # TODO: can we simplify this? (e.g. by using all attributes of the config object)
        return MultiHeadSelfAttentionEncoderLayer(num_hidden=self.num_hidden,
                                                  att_num_hidden=self.att_num_hidden,
                                                  dropout=self.dropout,
                                                  dropout_attention=self.dropout_attention,
                                                  heads=self.heads,
                                                  prefix=prefix + "mh_self_att_")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return MultiHeadSelfAttentionDecoderLayer(num_hidden=self.num_hidden,
                                                  att_num_hidden=self.att_num_hidden,
                                                  dropout=self.dropout,
                                                  dropout_attention=self.dropout_attention,
                                                  heads=self.heads,
                                                  prefix=prefix + "mh_self_att_")


# TODO: make sure the number of hidden units does not change!
class ResidualEncoderLayer(EncoderLayer):
    def __init__(self, layers: List[EncoderLayer]) -> None:
        self.layers = layers

    def encode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        att_dict: Optional[Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        new_source_encoded = source_encoded
        for layer in self.layers:
            new_source_encoded, new_source_encoded_lengths, new_source_encoded_max_length = layer.encode_sequence(
                new_source_encoded,
                source_encoded_lengths,
                source_encoded_max_length,
                att_dict)
            assert source_encoded_max_length == new_source_encoded_max_length, C.SEQUENCE_LENGTH_MUST_NOT_CHANGE_MSG
            assert source_encoded_lengths is new_source_encoded_lengths, C.SEQUENCE_LENGTH_MUST_NOT_CHANGE_MSG

        return source_encoded + new_source_encoded, source_encoded_lengths, source_encoded_max_length

    def att_names(self):
        att_names = []
        for layer in self.layers:
            att_names.extend(layer.att_names())
        return att_names

    def get_num_hidden(self) -> int:
        return self.layers[-1].get_num_hidden()


# TODO: potentially add a projection layer (for when the shapes don't match up). Alternative: check that the input num hidden matches the output num_hidden (maybe add a get_input_num_hidden())
# TODO: consider inheriting from both NestedDecoderLayer and SharedEncoderDecoderLayer to just have a single implementation
class ResidualDecoderLayer(NestedDecoderLayer):

    def __init__(self, layers: List[DecoderLayer]) -> None:
        self._layers = layers

    @property
    def layers(self) -> List[DecoderLayer]:
        return self._layers

    def decode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol, target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        target_encoded_input = target_encoded
        for layer in self.layers:
            target_encoded = layer.decode_sequence(source_encoded, source_encoded_lengths, source_encoded_max_length,
                                                   target_encoded, target_encoded_lengths, target_encoded_max_length,
                                                   target_autoregressive_bias)

        return target_encoded_input + target_encoded

    def decode_step(self, step: int, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int, target: mx.sym.Symbol, layer_states_flat: Sequence[mx.sym.Symbol],
                    att_dict: Dict[str, Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:

        new_layer_states_flat = []  # type: List[mx.sym.Symbol]
        target_input = target

        for layer, layer_states in self.layers_with_states_iter(step, layer_states_flat):
            target, new_layer_states = layer.decode_step(step, source_encoded, source_encoded_lengths,
                                                         source_encoded_max_length, target, layer_states, att_dict)
            new_layer_states_flat.extend(new_layer_states)
        return target_input + target, new_layer_states_flat

    def get_num_hidden(self) -> int:
        return self._layers[-1].get_num_hidden()


class ResidualLayerConfig(LayerConfig):
    def __init__(self, layer_configs: List[LayerConfig]) -> None:
        super().__init__()
        self.layer_configs = layer_configs

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        layers = []
        original_input_num_hidden = input_num_hidden
        for idx, layer in enumerate(self.layer_configs):
            new_layer = layer.create_encoder_layer(input_num_hidden, "%sres%d_" % (prefix, idx))
            input_num_hidden = new_layer.get_num_hidden()
            layers.append(new_layer)
        utils.check_condition(original_input_num_hidden == input_num_hidden,
                              "The input and output number of hidden units of the residual connection must match (%d vs %d)" % (
                              original_input_num_hidden, input_num_hidden))
        return ResidualEncoderLayer(layers)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        layers = []
        original_input_num_hidden = input_num_hidden
        for idx, layer in enumerate(self.layer_configs):
            new_layer = layer.create_decoder_layer(input_num_hidden, "%sres%d_" % (prefix, idx))
            input_num_hidden = new_layer.get_num_hidden()
            layers.append(new_layer)
        utils.check_condition(original_input_num_hidden == input_num_hidden,
                              "The input and output number of hidden units of the residual connection must match (%d vs %d)" % (
                              original_input_num_hidden, input_num_hidden))
        return ResidualDecoderLayer(layers)


# TODO: make this a block!?
class HighwayLayer:

    def __init__(self, num_hidden: int, gate_input: str, gated: str, prefix: str) -> None:
        self.gate_input = gate_input
        self.gated = gated
        self.ff = FeedForwardBlock(num_hidden=num_hidden,
                                   dropout=0.0,
                                   act_type=C.SIGMOID,
                                   prefix=prefix)

    def highway(self, input: mx.sym.Symbol, gate_input: mx.sym.Symbol, input_lengths: mx.sym.Symbol,
                input_max_length: int, output: mx.sym.Symbol):
        if self.gate_input == 'input':
            gate = self.ff(gate_input, input_lengths, input_max_length)
        elif self.gate_input == 'output':
            gate = self.ff(output, input_lengths, input_max_length)
        elif self.gate_input == 'both':
            gate = self.ff(mx.sym.concat(gate_input, output, dim=2), input_lengths, input_max_length)
        else:
            raise ValueError("unknown gate input %s" % self.gate_input)
        if self.gated == "both":
            return gate * input + (1. - gate) * output
        elif self.gated == "output":
            return input + gate * output
        else:
            raise ValueError("unknown gate method %s" % self.gated)

    def highway_step(self, step: int, input: mx.sym.Symbol, gate_input: mx.sym.Symbol, output: mx.sym.Symbol):
        if self.gate_input == 'input':
            gate = self.ff.step(step, gate_input)
        elif self.gate_input == 'output':
            gate = self.ff.step(step, output)
        elif self.gate_input == 'both':
            gate = self.ff.step(step, mx.sym.concat(gate_input, output, dim=1))
        else:
            raise ValueError("unknown gate input %s" % self.gate_input)
        if self.gated == "both":
            return gate * input + (1. - gate) * output
        elif self.gated == "output":
            return input + gate * output
        else:
            raise ValueError("unknown gate method %s" % self.gated)


class HighwayEncoderLayer(EncoderLayer, HighwayLayer):

    def __init__(self,
                 layers: List[EncoderLayer],
                 gate_input: str,
                 gated: str,
                 prefix: str = "") -> None:
        super().__init__(num_hidden=layers[-1].get_num_hidden(), gate_input=gate_input, gated=gated, prefix=prefix)
        self._layers = layers

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        # TODO: make sure input num hidden equals output num hidden
        highway_input = source_encoded
        gate_input = source_encoded

        new_source_encoded = source_encoded
        for layer in self._layers:
            new_source_encoded = layer.encode_sequence(new_source_encoded, source_encoded_lengths,
                                                       source_encoded_max_length, att_dict)[0]

        return self.highway(highway_input,
                            gate_input,
                            source_encoded_lengths,
                            source_encoded_max_length,
                            new_source_encoded), source_encoded_lengths, source_encoded_max_length

    def att_names(self):
        att_names = []
        for layer in self._layers:
            att_names.extend(layer.att_names())
        return att_names

    def get_num_hidden(self) -> int:
        return self._layers[-1].get_num_hidden()


class HighwayDecoderLayer(NestedDecoderLayer, HighwayLayer):

    def __init__(self,
                 layers: List[DecoderLayer],
                 gate_input: str,
                 gated: str,
                 prefix: str = "") -> None:
        # TODO: make sure input num hidden equals output num hidden
        super().__init__(num_hidden=layers[-1].get_num_hidden(), gate_input=gate_input, gated=gated, prefix=prefix)
        self._layers = layers
        num_hidden = layers[-1].get_num_hidden()

    @property
    def layers(self) -> List[DecoderLayer]:
        return self._layers

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        highway_input = target_encoded
        gate_input = target_encoded

        target_encoded_input = target_encoded
        for layer in self.layers:
            target_encoded = layer.decode_sequence(source_encoded, source_encoded_lengths, source_encoded_max_length,
                                                   target_encoded, target_encoded_lengths, target_encoded_max_length,
                                                   target_autoregressive_bias)

        return self.highway(highway_input,
                            gate_input,
                            target_encoded_lengths,
                            target_encoded_max_length,
                            target_encoded)

    def decode_step(self, step: int, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int, target: mx.sym.Symbol, layer_states_flat: Sequence[mx.sym.Symbol],
                    att_dict: Dict[str, Dict[str, mx.sym.Symbol]]) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        highway_input = target
        gate_input = target

        new_layer_states_flat = []  # type: List[mx.sym.Symbol]

        for layer, layer_states in self.layers_with_states_iter(step, layer_states_flat):
            target, new_layer_states = layer.decode_step(step, source_encoded, source_encoded_lengths,
                                                         source_encoded_max_length, target, layer_states, att_dict)
            new_layer_states_flat.extend(new_layer_states)

        return self.highway_step(step, highway_input, gate_input, target), new_layer_states_flat

    def get_num_hidden(self) -> int:
        return self._layers[-1].get_num_hidden()


class HighwayLayerConfig(LayerConfig):

    def __init__(self, layer_configs: List[LayerConfig], gate_input="input", gated: str = "both") -> None:
        super().__init__()
        self.layer_configs = layer_configs
        self.gate_input = gate_input
        self.gated = gated

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        layers = []
        for idx, layer_config in enumerate(self.layer_configs):
            layer = layer_config.create_encoder_layer(input_num_hidden, "%shighway%d_" % (prefix, idx))
            input_num_hidden = layer.get_num_hidden()
            layers.append(layer)
        return HighwayEncoderLayer(layers=layers, gate_input=self.gate_input,
                                   gated=self.gated,
                                   prefix=prefix + "highway_")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        layers = []
        for idx, layer_config in enumerate(self.layer_configs):
            layer = layer_config.create_decoder_layer(input_num_hidden, "%s_highway%d" % (prefix, idx))
            input_num_hidden = layer.get_num_hidden()
            layers.append(layer)
        return HighwayDecoderLayer(layers=layers, gate_input=self.gate_input,
                                   gated=self.gated,
                                   prefix=prefix + "highway_")


# TODO: implement this as a stateless layer instead?!
class IdentityEncoderLayer(EncoderLayer):
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        return source_encoded, source_encoded_lengths, source_encoded_max_length

    def get_num_hidden(self) -> int:
        return self.num_hidden


class IdentityDecoderLayer(DecoderLayer):
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden

    def decode_sequence(self, source_encoded: Sequence[mx.sym.Symbol], source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol, target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        return target_encoded

    def decode_step(self, step: int, source_encoded: Sequence[mx.sym.Symbol], source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int, target: mx.sym.Symbol, states: Sequence[mx.sym.Symbol], att_dict) -> Tuple[mx.sym.Symbol, Sequence[mx.sym.Symbol]]:
        return target, []

    def get_num_hidden(self) -> int:
        return self.num_hidden


class IdentityLayerConfig(LayerConfig):
    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return IdentityEncoderLayer(num_hidden=input_num_hidden)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return IdentityDecoderLayer(num_hidden=input_num_hidden)


class LayerNormalization(StatelessBlock):
    """
    Implements Ba et al, Layer Normalization (https://arxiv.org/abs/1607.06450).

    :param prefix: Optional prefix of layer name.
    :param scale: Optional variable for scaling of shape (num_hidden,). Will be created if None.
    :param shift: Optional variable for shifting of shape (num_hidden,). Will be created if None.
    :param scale_init: Initial value of scale variable if scale is None. Default 1.0.
    :param shift_init: Initial value of shift variable if shift is None. Default 0.0.
    """
    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'layernorm',
                 scale: Optional[mx.sym.Symbol] = None,
                 shift: Optional[mx.sym.Symbol] = None,
                 scale_init: float = 1.0,
                 shift_init: float = 0.0) -> None:
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.scale = scale if scale is not None else mx.sym.Variable('%s_gamma' % prefix,
                                                                     init=mx.init.Constant(value=scale_init))
        self.shift = shift if shift is not None else mx.sym.Variable('%s_beta' % prefix,
                                                                     init=mx.init.Constant(value=shift_init))

    def __call__(self,
                 data: mx.sym.Symbol,
                 lengths: Optional[mx.sym.Symbol] = None,
                 max_length: Optional[int] = None,
                 eps: float = 1e-06) -> mx.sym.Symbol:
        """
        Normalizes hidden units of data as follows:

        data = scale * (data - mean) / sqrt(var + eps) + shift

        Normalization is performed over the last dimension of the input data.

        :param data: Data to normalize. Shape: (d0, ..., dn, num_hidden).
        :param eps: Variance epsilon.
        :return: inputs_norm: Normalized inputs. Shape: (d0, ..., dn, num_hidden).
        """
        return mx.sym.LayerNorm(data=data, gamma=self.scale, beta=self.shift, axis=-1,
                                eps=eps, output_mean_var=False, name=self.prefix)

    def get_num_hidden(self):
        return self.num_hidden


class LayerNormalizationLayerConfig(LayerConfig):
    def __init__(self):
        super().__init__()

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> EncoderLayer:
        return self.create_layer(input_num_hidden, prefix)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> DecoderLayer:
        return self.create_layer(input_num_hidden, prefix)

    def create_layer(self, num_hidden: int, prefix) -> StatelessBlockLayer:
        return StatelessBlockLayer(LayerNormalization(num_hidden=num_hidden,
                                                      prefix=prefix + "norm_"))


class LHUC:
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    :param weight: Optional parameter vector.
    :param prefix: Optional prefix for created parameters (if not given as weight).
    """
    def __init__(self,
                 num_hidden: int,
                 weight: Optional[mx.sym.Symbol] = None,
                 prefix: str = "") -> None:
        self.num_hidden = num_hidden
        self.prefix = prefix
        if weight is None:
            self.params = mx.sym.Variable(self.prefix + C.LHUC_NAME,
                                          shape=(self.num_hidden,),
                                          init=mx.init.Uniform(0.1),
                                          dtype="float32")
        else:
            self.params = weight

    def __call__(self,
                 inputs: mx.sym.Symbol,
                 name: Optional[str] = None) -> mx.sym.Symbol:

        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight_vector = 2 * mx.sym.Activation(data=self.params, act_type="sigmoid")
        out = mx.sym.broadcast_mul(weight_vector, inputs, name=name)

        return out


class WeightNormalization:
    """
    Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
    For a given tensor the normalization is done per hidden dimension.

    :param weight: Weight tensor of shape: (num_hidden, d1, d2, ...).
    :param num_hidden: Size of the first dimension.
    :param ndim: The total number of dimensions of the weight tensor.
    :param prefix: The prefix used for naming.
    """

    def __init__(self, weight, num_hidden, ndim=2, prefix: str = '') -> None:
        self.prefix = prefix
        self.weight = weight
        self.num_hidden = num_hidden
        self.scale = mx.sym.Variable("%swn_scale" % prefix,
                                     shape=tuple([num_hidden] + [1] * (ndim - 1)),
                                     init=mx.init.Constant(value=1.0))

    def __call__(self, weight: Optional[mx.nd.NDArray] = None, scale: Optional[mx.nd.NDArray] = None) -> mx.sym.Symbol:
        """
        Normalize each hidden dimension and scale afterwards

        :return: A weight normalized weight tensor.
        """
        if weight is None and scale is None:
            return mx.sym.broadcast_mul(lhs=mx.sym.L2Normalization(self.weight, mode='instance'),
                                        rhs=self.scale, name="%swn_scale" % self.prefix)
        else:
            assert isinstance(weight, mx.nd.NDArray)
            assert isinstance(scale, mx.nd.NDArray)
            return mx.nd.broadcast_mul(lhs=mx.nd.L2Normalization(weight, mode='instance'), rhs=scale)


class OutputLayer:
    """
    Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.

    :param hidden_size: Decoder hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight_normalization: Whether to apply weight normalization.
    :param prefix: Prefix used for naming.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 weight: Optional[mx.sym.Symbol],
                 weight_normalization: bool,
                 prefix: str = C.DEFAULT_OUTPUT_LAYER_PREFIX) -> None:
        self.vocab_size = vocab_size
        self.prefix = prefix

        if weight is None:
            self.w = mx.sym.Variable("%sweight" % self.prefix, shape=(vocab_size, hidden_size))
        else:
            self.w = weight

        self.weight_normalization = weight_normalization
        if weight_normalization:
            logger.info("Normalizing output layer weights.")
            self.weight_norm = WeightNormalization(self.w,
                                                   num_hidden=vocab_size,
                                                   ndim=2,
                                                   prefix=self.prefix)
            self.w = self.weight_norm()

        self.b = mx.sym.Variable("%sbias" % self.prefix)

    def __call__(self,
                 hidden: Union[mx.sym.Symbol, mx.nd.NDArray],
                 weight: Optional[mx.nd.NDArray] = None,
                 bias: Optional[mx.nd.NDArray] = None):
        """
        Linear transformation to vocab size. Returns logits.

        :param hidden: Decoder representation for n elements. Shape: (n, self.num_hidden).
        :return: Logits. Shape(n, self.vocab_size).
        """
        if isinstance(hidden, mx.sym.Symbol):
            # TODO dropout?
            return mx.sym.FullyConnected(data=hidden,
                                         num_hidden=self.vocab_size,
                                         weight=self.w,
                                         bias=self.b,
                                         flatten=False,
                                         name=C.LOGITS_NAME)

        # Equivalent NDArray implementation (requires passed weights/biases)
        assert isinstance(hidden, mx.nd.NDArray)
        utils.check_condition(weight is not None and bias is not None,
                              "OutputLayer NDArray implementation requires passing weight and bias NDArrays.")

        return mx.nd.FullyConnected(data=hidden,
                                    num_hidden=bias.shape[0],
                                    weight=weight,
                                    bias=bias,
                                    flatten=False)


def split_heads(x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with head dimension folded into batch and depth divided by the number of heads.

    :param x: Symbol of shape (batch, length, depth).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch * heads, length, depth_per_heads).
    """
    # (batch, length, heads, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(0, -1, heads, depth_per_head))
    # (batch, heads, length, depth/heads)
    x = mx.sym.transpose(data=x, axes=(0, 2, 1, 3))
    # (batch * heads, length, depth/heads)
    return mx.sym.reshape(data=x, shape=(-3, -1, depth_per_head))


def combine_heads(x: mx.sym.Symbol, depth_per_head: int, heads: int) -> mx.sym.Symbol:
    """
    Returns a symbol with both batch & length, and head & depth dimensions combined.

    :param x: Symbol of shape (batch * heads, length, depth_per_head).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch, length, depth).
    """
    # (batch, heads, length, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(-4, -1, heads, 0, depth_per_head))
    # (batch, length, heads, depth_per_head)
    x = mx.sym.transpose(x, axes=(0, 2, 1, 3))
    # (batch, length, depth)
    return mx.sym.reshape(x, shape=(-1, 0, depth_per_head * heads))


def broadcast_to_heads(x: mx.sym.Symbol, num_heads: int, ndim: int, fold_heads: bool = True) -> mx.sym.Symbol:
    """
    Broadcasts batch-major input of shape (batch, d1 ... dn-1) to (batch*heads, d1 ... dn-1).

    :param x: Batch-major input. Shape: (batch, d1 ... dn-1).
    :param num_heads: Number of heads.
    :param ndim: Number of dimensions in x.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :return: Tensor with each sample repeated heads-many times.
             Shape: (batch * heads, d1 ... dn-1) if fold_heads == True, (batch, heads, d1 ... dn-1) else.
    """
    dims = [0] * (ndim - 1)
    # x: (batch, 1)
    x = mx.sym.expand_dims(x, axis=1)
    # x: (batch, heads, dims...)
    x = mx.sym.broadcast_to(x, shape=[0, num_heads] + dims)
    if fold_heads:
        # (batch * heads, dims...)
        return mx.sym.reshape(x, shape=[-3] + dims)
    else:
        # x: (batch, heads, dims...)
        return x


def dot_attention(queries: mx.sym.Symbol,
                  keys: mx.sym.Symbol,
                  values: mx.sym.Symbol,
                  lengths: Optional[mx.sym.Symbol] = None,
                  dropout: float = 0.0,
                  bias: Optional[mx.sym.Symbol] = None,
                  prefix: Optional[str] = ''):
    """
    Computes dot attention for a set of queries, keys, and values.

    :param queries: Attention queries. Shape: (n, lq, d).
    :param keys: Attention keys. Shape: (n, lk, d).
    :param values: Attention values. Shape: (n, lk, dv).
    :param lengths: Optional sequence lengths of the keys. Shape: (n,).
    :param dropout: Dropout probability.
    :param bias: Optional 3d bias tensor.
    :param prefix: Optional prefix
    :return: 'Context' vectors for each query. Shape: (n, lq, dv).
    """
    # (n, lq, lk)
    logits = mx.sym.batch_dot(lhs=queries, rhs=keys, transpose_b=True, name='%sdot' % prefix)

    if lengths is not None:
        # mask lk dimension
        # (lk, n, lq)
        logits = mx.sym.transpose(data=logits, axes=(2, 0, 1))
        logits = mx.sym.SequenceMask(data=logits,
                                     use_sequence_length=True,
                                     sequence_length=lengths,
                                     value=C.LARGE_NEGATIVE_VALUE)
        # (n, lq, lk)
        logits = mx.sym.transpose(data=logits, axes=(1, 2, 0))

    if bias is not None:
        logits = mx.sym.broadcast_add(logits, bias, name='%sbias_add' % prefix)

    probs = mx.sym.softmax(logits, axis=-1)
    probs = mx.sym.Dropout(probs, p=dropout) if dropout > 0.0 else probs

    # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
    return mx.sym.batch_dot(lhs=probs, rhs=values, name='%scontexts' % prefix), probs


class MultiHeadAttentionBase:
    """
    Base class for Multi-head attention.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """
    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        self.prefix = prefix
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.dropout = dropout
        self.depth_per_head = self.depth // self.heads

        self.w_h2o = mx.sym.Variable("%sh2o_weight" % prefix)

    def _attend(self,
                queries: mx.sym.Symbol,
                keys: mx.sym.Symbol,
                values: mx.sym.Symbol,
                lengths: Optional[mx.sym.Symbol] = None,
                bias: Optional[mx.sym.Symbol] = None) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (batch_size, query_max_length, depth).
        :param keys: Keys. Shape: (batch_size, memory_max_length, depth).
        :param values: Values. Shape: (batch_size, memory_max_length, depth).
        :param lengths: Optional lengths of keys. Shape: (batch_size,).
        :param bias: Optional 3d bias.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth) and attention probabilities.
            Shape: (batch_size, query_max_length, memory_max_length).
        """
        # scale by sqrt(depth_per_head)
        queries = queries * (self.depth_per_head ** -0.5)

        # (batch*heads, length, depth/heads)
        queries = split_heads(queries, self.depth_per_head, self.heads)
        keys = split_heads(keys, self.depth_per_head, self.heads)
        values = split_heads(values, self.depth_per_head, self.heads)
        lengths = broadcast_to_heads(lengths, self.heads, ndim=1, fold_heads=True) if lengths is not None else lengths

        # (batch*heads, query_max_length, depth_per_head)
        contexts, probs = dot_attention(queries, keys, values,
                                        lengths=lengths, dropout=self.dropout, bias=bias, prefix=self.prefix)

        # (batch, query_max_length, depth)
        contexts = combine_heads(contexts, self.depth_per_head, self.heads)

        # contexts: (batch, query_max_length, output_depth)
        contexts = mx.sym.FullyConnected(data=contexts,
                                         weight=self.w_h2o,
                                         no_bias=True,
                                         num_hidden=self.depth_out,
                                         flatten=False)

        return contexts, probs


class MultiHeadSelfAttention(MultiHeadAttentionBase):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """
    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout)
        self.w_i2h = mx.sym.Variable("%si2h_weight" % prefix)

    def __call__(self,
                 inputs: mx.sym.Symbol,
                 input_lengths: Optional[mx.sym.Symbol] = None,
                 bias: Optional[mx.sym.Symbol] = None,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a symbol of shape (batch, max_length, output_depth).

        :param inputs: Input Data. Shape: (batch, max_length, input_depth).
        :param input_lengths: Optional lengths of inputs to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param cache: Optional dictionary of previously computed keys and values.
        :return: Symbol of shape (batch, max_length, output_depth).
        """
        # combined: (batch, max_length, depth * 3)
        combined = mx.sym.FullyConnected(data=inputs,
                                         weight=self.w_i2h,
                                         no_bias=True,
                                         num_hidden=self.depth * 3,
                                         flatten=False,
                                         name="%sqkv_transform" % self.prefix)
        # split into query, keys and values
        # (batch, max_length, depth)
        # pylint: disable=unbalanced-tuple-unpacking
        queries, keys, values = mx.sym.split(data=combined, num_outputs=3, axis=2)

        if cache is not None:
            # append new keys & values to cache, update the cache
            keys = cache['k'] = keys if cache['k'] is None else mx.sym.concat(cache['k'], keys, dim=1)
            values = cache['v'] = values if cache['v'] is None else mx.sym.concat(cache['v'], values, dim=1)

        return self._attend(queries,
                            keys,
                            values,
                            lengths=input_lengths,
                            bias=bias)


class MultiHeadAttention(MultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param prefix: Attention prefix.
    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """

    def __init__(self,
                 prefix: str,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(prefix, depth_att, heads, depth_out, dropout)
        self.w_q2h = mx.sym.Variable("%sq2h_weight" % prefix)
        self.w_k2h = mx.sym.Variable("%sk2h_weight" % prefix)
        self.w_v2h = mx.sym.Variable("%sv2h_weight" % prefix)

    def __call__(self,
                 queries: mx.sym.Symbol,
                 memory: mx.sym.Symbol,
                 memory_lengths: Optional[mx.sym.Symbol] = None,
                 bias: Optional[mx.sym.Symbol] = None) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Query tensor. Shape: (batch, query_max_length, input_depth).
        :param memory: Memory data to attend to. Shape: (batch, memory_max_length, input_depth).
        :param memory_lengths: Optional lengths of memory to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :return: Symbol of shape (batch, query_seq_len, output_depth).
        """
        # (batch, query_max_length, depth)
        queries = mx.sym.FullyConnected(data=queries,
                                        weight=self.w_q2h,
                                        no_bias=True,
                                        num_hidden=self.depth,
                                        flatten=False,
                                        name="%sq_transform" % self.prefix)

        # (batch, memory_max_length, depth)
        keys = mx.sym.FullyConnected(data=memory,
                                     weight=self.w_k2h,
                                     no_bias=True,
                                     num_hidden=self.depth,
                                     flatten=False,
                                     name="%sk_transform" % self.prefix)

        # (batch, memory_max_length, depth)
        values = mx.sym.FullyConnected(data=memory,
                                       weight=self.w_v2h,
                                       no_bias=True,
                                       num_hidden=self.depth,
                                       flatten=False,
                                       name="%sv_transform" % self.prefix)

        return self._attend(queries,
                            keys,
                            values,
                            bias=bias,
                            lengths=memory_lengths)


class ProjectedDotAttention:
    """
    Dot attention layer for queries independent from keys/values.

    :param prefix: Attention prefix.
    :param num_hidden: Attention depth / number of hidden units.
    """

    def __init__(self,
                 prefix: str,
                 num_hidden) -> None:
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.w_q2h = mx.sym.Variable("%sq2h_weight" % prefix)
        self.b_q2h = mx.sym.Variable("%sq2h_bias" % prefix)
        self.w_kv2h = mx.sym.Variable("%skv2h_weight" % prefix)
        self.b_kv2h = mx.sym.Variable("%skv2h_bias" % prefix)

    def __call__(self,
                 queries: mx.sym.Symbol,
                 memory: mx.sym.Symbol,
                 memory_lengths: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Apply project, apply dot attention and return new context vectors.

        :param queries: Symbol of shape (batch, queries_max_length, input_num_hidden).
        :param memory: Symbol of shape (batch, memory_max_length, input_num_hidden).
        :param memory_lengths: Symbol of shape (batch, 1).
        :return: Symbol of shape (batch, queries_max_length, num_hidden).
        """
        # (batch, memory_max_length, num_hidden * 2)
        combined = mx.sym.FullyConnected(data=memory,
                                         weight=self.w_kv2h,
                                         bias=self.b_kv2h,
                                         num_hidden=self.num_hidden * 2,
                                         flatten=False,
                                         name="%skv_transform" % self.prefix)

        # split into keys and values
        # pylint: disable=unbalanced-tuple-unpacking
        keys, values = mx.sym.split(data=combined, num_outputs=2, axis=2)

        # (batch, queries_max_length, num_hidden)
        queries = mx.sym.FullyConnected(data=queries,
                                        weight=self.w_q2h,
                                        bias=self.b_q2h,
                                        num_hidden=self.num_hidden,
                                        flatten=False,
                                        name="%sq_transform" % self.prefix)
        # scale by sqrt(num_hidden)
        queries = queries * (self.num_hidden ** -0.5)

        # (batch, queries_max_length, num_hidden)
        contexts, probs = dot_attention(queries, keys, values, memory_lengths)

        return contexts


class PlainDotAttention:
    """
    Dot attention layer for queries independent from keys/values.
    """

    def __call__(self,
                 queries: mx.sym.Symbol,
                 memory: mx.sym.Symbol,
                 memory_lengths: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns a symbol of shape (batch, max_length, output_depth).

        :param queries: Symbol of shape (batch, queries_max_length, input_depth).
        :param memory: Symbol of shape (batch, memory_max_length, input_depth).
        :param memory_lengths: Symbol of shape (batch, 1).
        :return: Symbol of shape (batch, queries_max_length, output_depth).
        """

        # (batch*heads, queries_max_length, depth_per_head)
        contexts, probs = dot_attention(queries, memory, memory, memory_lengths)

        return contexts


class PositionalEncodings(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, max_seq_len, num_embed)
    with positional encodings as in Vaswani et al, 2017.

    :param length: Maximum sequence length.
    :param depth: Embedding size.
    """

    def __init__(self, length: int, depth: int) -> None:
        super().__init__()
        self.encodings = self.get_encodings(length, depth)

    @staticmethod
    def get_encodings(length, depth) -> np.ndarray:
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
        # interleave: (1, length, num_embed)
        encodings = np.hstack([sin, cos]).reshape(1, length, depth)
        return encodings

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.encodings)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("positional_encodings")
class PositionalEncodingsProp(mx.operator.CustomOpProp):

    def __init__(self, length: str, depth: str) -> None:
        super().__init__()
        self.length = int(length)
        self.depth = int(depth)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.depth)], []

    def infer_type(self, in_type):
        return [], [np.float32], []

    def create_operator(self, ctx, shapes, dtypes):
        return PositionalEncodings(length=self.length, depth=self.depth)

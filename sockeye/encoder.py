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
Encoders for sequence-to-sequence models.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from math import ceil, floor
from typing import Callable, List, Optional, Tuple, Union, Dict

import mxnet as mx

from . import config
from . import constants as C
from . import convolution
from . import rnn
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


ImageEncoderConfig = None


def get_encoder(config: 'EncoderConfig', prefix: str = '') -> 'Encoder':
    if isinstance(config, RecurrentEncoderConfig):
        return get_recurrent_encoder(config, prefix)
    elif isinstance(config, transformer.TransformerConfig):
        return get_transformer_encoder(config, prefix)
    elif isinstance(config, ConvolutionalEncoderConfig):
        return get_convolutional_encoder(config, prefix)
    else:
        from .image_captioning.encoder import ImageLoadedCnnEncoderConfig, \
            get_image_cnn_encoder
        ImageEncoderConfig = ImageLoadedCnnEncoderConfig

        if isinstance(config, ImageLoadedCnnEncoderConfig):
            return get_image_cnn_encoder(config)
        else:
            raise ValueError("Unsupported encoder configuration")


class RecurrentEncoderConfig(config.Config):
    """
    Recurrent encoder configuration.

    :param rnn_config: RNN configuration.
    :param conv_config: Optional configuration for convolutional embedding.
    :param reverse_input: Reverse embedding sequence before feeding into RNN.
    :param dtype: Data type.
    """

    def __init__(self,
                 rnn_config: rnn.RNNConfig,
                 conv_config: Optional['ConvolutionalEmbeddingConfig'] = None,
                 reverse_input: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.rnn_config = rnn_config
        self.conv_config = conv_config
        self.reverse_input = reverse_input
        self.dtype = dtype


class ConvolutionalEncoderConfig(config.Config):
    """
    Convolutional encoder configuration.

    :param cnn_config: CNN configuration.
    :param num_layers: The number of convolutional layers on top of the embeddings.
    :param positional_embedding_type: The type of positional embedding.
    :param dtype: Data type.
    """

    def __init__(self,
                 num_embed: int,
                 max_seq_len_source: int,
                 cnn_config: convolution.ConvolutionConfig,
                 num_layers: int,
                 positional_embedding_type: str,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.num_embed = num_embed
        self.num_layers = num_layers
        self.cnn_config = cnn_config
        self.max_seq_len_source = max_seq_len_source
        self.positional_embedding_type = positional_embedding_type
        self.dtype = dtype


def get_recurrent_encoder(config: RecurrentEncoderConfig, prefix: str) -> 'Encoder':
    """
    Returns an encoder stack with a bi-directional RNN, and a variable number of uni-directional forward RNNs.

    :param config: Configuration for recurrent encoder.
    :param prefix: Prefix for variable names.
    :return: Encoder instance.
    """
    # TODO give more control on encoder architecture
    encoder_seq = EncoderSequence([], config.dtype)

    if config.conv_config is not None:
        encoder_seq.append(ConvolutionalEmbeddingEncoder, config=config.conv_config,
                           prefix=prefix + C.CHAR_SEQ_ENCODER_PREFIX)
        if config.conv_config.add_positional_encoding:
            # If specified, add positional encodings to segment embeddings
            encoder_seq.append(AddSinCosPositionalEmbeddings,
                               num_embed=config.conv_config.num_embed,
                               scale_up_input=False,
                               scale_down_positions=False,
                               prefix="%s%sadd_positional_encodings" % (prefix, C.CHAR_SEQ_ENCODER_PREFIX))
        encoder_seq.append(ConvertLayout, infer_hidden=True, target_layout=C.TIME_MAJOR)
    else:
        encoder_seq.append(ConvertLayout, target_layout=C.TIME_MAJOR, num_hidden=0)

    if config.reverse_input:
        encoder_seq.append(ReverseSequence, infer_hidden=True)

    if config.rnn_config.residual:
        utils.check_condition(config.rnn_config.first_residual_layer >= 2,
                              "Residual connections on the first encoder layer are not supported")

    # One layer bi-directional RNN:
    encoder_seq.append(BiDirectionalRNNEncoder,
                       rnn_config=config.rnn_config.copy(num_layers=1),
                       prefix=prefix + C.BIDIRECTIONALRNN_PREFIX,
                       layout=C.TIME_MAJOR)

    if config.rnn_config.num_layers > 1:
        # Stacked uni-directional RNN:
        # Because we already have a one layer bi-rnn we reduce the num_layers as well as the first_residual_layer.
        remaining_rnn_config = config.rnn_config.copy(num_layers=config.rnn_config.num_layers - 1,
                                                      first_residual_layer=config.rnn_config.first_residual_layer - 1)
        encoder_seq.append(RecurrentEncoder,
                           rnn_config=remaining_rnn_config,
                           prefix=prefix + C.STACKEDRNN_PREFIX,
                           layout=C.TIME_MAJOR)

    encoder_seq.append(ConvertLayout, infer_hidden=True, target_layout=C.BATCH_MAJOR)

    return encoder_seq


def get_convolutional_encoder(config: ConvolutionalEncoderConfig, prefix: str) -> 'Encoder':
    """
    Creates a convolutional encoder.

    :param config: Configuration for convolutional encoder.
    :param prefix: Prefix for variable names.
    :return: Encoder instance.
    """
    encoder_seq = EncoderSequence([], dtype=config.dtype)
    cls, encoder_params = _get_positional_embedding_params(config.positional_embedding_type,
                                                           config.num_embed,
                                                           max_seq_len=config.max_seq_len_source,
                                                           fixed_pos_embed_scale_up_input=False,
                                                           fixed_pos_embed_scale_down_positions=True,
                                                           prefix=prefix + C.SOURCE_POSITIONAL_EMBEDDING_PREFIX)
    encoder_seq.append(cls, **encoder_params)
    encoder_seq.append(ConvolutionalEncoder, config=config)
    return encoder_seq


def get_transformer_encoder(config: transformer.TransformerConfig, prefix: str) -> 'Encoder':
    """
    Returns a Transformer encoder, consisting of an embedding layer with
    positional encodings and a TransformerEncoder instance.

    :param config: Configuration for transformer encoder.
    :param prefix: Prefix for variable names.
    :return: Encoder instance.
    """
    encoder_seq = EncoderSequence([], dtype=config.dtype)
    cls, encoder_params = _get_positional_embedding_params(config.positional_embedding_type,
                                                           config.model_size,
                                                           config.max_seq_len_source,
                                                           fixed_pos_embed_scale_up_input=True,
                                                           fixed_pos_embed_scale_down_positions=False,
                                                           prefix=prefix + C.SOURCE_POSITIONAL_EMBEDDING_PREFIX)
    encoder_seq.append(cls, **encoder_params)
    if config.conv_config is not None:
        encoder_seq.append(ConvolutionalEmbeddingEncoder, config=config.conv_config,
                           prefix=prefix + C.CHAR_SEQ_ENCODER_PREFIX)

    encoder_seq.append(TransformerEncoder, config=config, prefix=prefix + C.TRANSFORMER_ENCODER_PREFIX)

    return encoder_seq


class Encoder(ABC):
    """
    Generic encoder interface.

    :param dtype: Data type.
    """

    @abstractmethod
    def __init__(self, dtype):
        logger.info('{}.{} dtype: {}'.format(self.__module__, self.__class__.__name__, dtype))
        self.dtype = dtype

    @abstractmethod
    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        pass

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this encoder.
        """
        raise NotImplementedError()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        :return: The size of the encoded sequence.
        """
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        return None


class ConvertLayout(Encoder):
    """
    Converts batch major data to time major by swapping the first dimension and setting the __layout__ attribute.

    :param target_layout: The target layout to convert to (C.BATCH_MAJOR or C.TIMEMAJOR).
    :param num_hidden: The number of hidden units of the previous encoder.
    :param dtype: Data type.
    """

    def __init__(self, target_layout: str, num_hidden: int, dtype: str = C.DTYPE_FP32) -> None:
        assert target_layout == C.BATCH_MAJOR or target_layout == C.TIME_MAJOR
        super().__init__(dtype)
        self.num_hidden = num_hidden
        self.target_layout = target_layout

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        with mx.AttrScope(__layout__=self.target_layout):
            return mx.sym.swapaxes(data=data, dim1=0, dim2=1), data_length, seq_len

    def get_num_hidden(self) -> int:
        return self.num_hidden


class ReverseSequence(Encoder):
    """
    Reverses the input sequence. Requires time-major layout.

    :param dtype: Data type.
    """

    def __init__(self, num_hidden: int, dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(dtype)
        self.num_hidden = num_hidden

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        data = mx.sym.SequenceReverse(data=data, sequence_length=data_length, use_sequence_length=True)
        return data, data_length, seq_len

    def get_num_hidden(self):
        return self.num_hidden


class FactorConfig(config.Config):

    def __init__(self, vocab_size: int, num_embed: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed


class EmbeddingConfig(config.Config):

    def __init__(self,
                 vocab_size: int,
                 num_embed: int,
                 dropout: float,
                 factor_configs: Optional[List[FactorConfig]] = None,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.dropout = dropout
        self.factor_configs = factor_configs
        self.num_factors = 1
        if self.factor_configs is not None:
            self.num_factors += len(self.factor_configs)
        self.dtype = dtype


class Embedding(Encoder):
    """
    Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.

    :param config: Embedding config.
    :param prefix: Name prefix for symbols of this encoder.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new one.
    :param is_source: Whether this is the source embedding instance. Default: False.
    """

    def __init__(self,
                 config: EmbeddingConfig,
                 prefix: str,
                 embed_weight: Optional[mx.sym.Symbol] = None,
                 is_source: bool = False) -> None:
        super().__init__(config.dtype)
        self.config = config
        self.prefix = prefix
        self.embed_weight = embed_weight
        self.is_source = is_source

        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable(prefix + "weight",
                                                shape=(self.config.vocab_size, self.config.num_embed))

        self.embed_factor_weights = []  # type: List[mx.sym.Symbol]
        if self.config.factor_configs is not None:
            # Factors weights aren't shared so they're not passed in and we create them here.
            for i, fc in enumerate(self.config.factor_configs):
                self.embed_factor_weights.append(mx.sym.Variable(prefix + "factor%d_weight" % i,
                                                                 shape=(fc.vocab_size, fc.num_embed)))

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        factor_embeddings = []  # type: List[mx.sym.Symbol]
        if self.is_source:
            data, *data_factors = mx.sym.split(data=data,
                                               num_outputs=self.config.num_factors,
                                               axis=2,
                                               squeeze_axis=True, name=self.prefix + "factor_split")

            if self.config.factor_configs is not None:
                for i, (factor_data, factor_config, factor_weight) in enumerate(zip(data_factors,
                                                                                    self.config.factor_configs,
                                                                                    self.embed_factor_weights)):
                    factor_embeddings.append(mx.sym.Embedding(data=factor_data,
                                                              input_dim=factor_config.vocab_size,
                                                              weight=factor_weight,
                                                              output_dim=factor_config.num_embed,
                                                              name=self.prefix + "factor%d_embed" % i))

        embedding = mx.sym.Embedding(data=data,
                                     input_dim=self.config.vocab_size,
                                     weight=self.embed_weight,
                                     output_dim=self.config.num_embed,
                                     name=self.prefix + "embed")

        if self.config.factor_configs is not None:
            embedding = mx.sym.concat(embedding, *factor_embeddings, dim=2, name=self.prefix + "embed_plus_factors")

        if self.config.dropout > 0:
            embedding = mx.sym.Dropout(data=embedding, p=self.config.dropout, name="source_embed_dropout")

        return embedding, data_length, seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.num_embed


class PassThroughEmbeddingConfig(config.Config):

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 0
        self.num_embed = 0
        self.num_factors = 1


class PassThroughEmbedding(Encoder):
    """
    This is an embedding which passes through an input symbol without doing any operation.

    :param config: PassThroughEmbeddingConfig config.
    """

    def __init__(self,
                 config: PassThroughEmbeddingConfig) -> None:
        self.config = config

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int = 0) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        return data, data_length, seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return 0


class PositionalEncoder(Encoder):
    @abstractmethod
    def encode_positions(self,
                         positions: mx.sym.Symbol,
                         data: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Add positional encodings to the data using the provided positions.
        :param positions: (batch_size,)
        :param data: (batch_size, num_embed)
        :return: (batch_size, num_embed)
        """
        pass


class AddSinCosPositionalEmbeddings(PositionalEncoder):
    """
    Takes an encoded sequence and adds fixed positional embeddings as in Vaswani et al, 2017 to it.

    :param num_embed: Embedding size.
    :param prefix: Name prefix for symbols of this encoder.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    :param dtype: Data type.
    """

    def __init__(self,
                 num_embed: int,
                 prefix: str,
                 scale_up_input: bool,
                 scale_down_positions: bool,
                 dtype: str = C.DTYPE_FP32) -> None:
        utils.check_condition(num_embed % 2 == 0, "Positional embeddings require an even embedding size it "
                                                  "is however %d." % num_embed)
        super().__init__(dtype)
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions
        self.num_embed = num_embed
        self.prefix = prefix

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        :param data: (batch_size, source_seq_len, num_embed)
        :param data_length: (batch_size,)
        :param seq_len: sequence length.
        :return: (batch_size, source_seq_len, num_embed)
        """
        # add positional embeddings to data
        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        positions = mx.sym.BlockGrad(mx.symbol.Custom(length=seq_len,
                                                      depth=self.num_embed,
                                                      name="%spositional_encodings" % self.prefix,
                                                      op_type='positional_encodings'))

        if self.scale_down_positions:
            positions = positions * (self.num_embed ** -0.5)

        embedding = mx.sym.broadcast_add(data, positions)
        return embedding, data_length, seq_len

    def encode_positions(self,
                         positions: mx.sym.Symbol,
                         data: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        :param positions: (batch_size,)
        :param data: (batch_size, num_embed)
        :return: (batch_size, num_embed)
        """
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

        return mx.sym.broadcast_add(data, pos_embedding, name="%s_add" % self.prefix)

    def get_num_hidden(self) -> int:
        return self.num_embed


class AddLearnedPositionalEmbeddings(PositionalEncoder):
    """
    Takes an encoded sequence and adds positional embeddings to it, which are learned jointly. Note that this will
    limited the maximum sentence length during decoding.

    :param num_embed: Embedding size.
    :param max_seq_len: Maximum sequence length.
    :param prefix: Name prefix for symbols of this encoder.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new one.
    :param dtype: Data type.
    """

    def __init__(self,
                 num_embed: int,
                 max_seq_len: int,
                 prefix: str,
                 embed_weight: Optional[mx.sym.Symbol] = None,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(dtype)
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.prefix = prefix
        if embed_weight is not None:
            self.embed_weight = embed_weight
        else:
            self.embed_weight = mx.sym.Variable(prefix + "weight")

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        :param data: (batch_size, source_seq_len, num_embed)
        :param data_length: (batch_size,)
        :param seq_len: sequence length.
        :return: (batch_size, source_seq_len, num_embed)
        """

        # (1, source_seq_len)
        positions = mx.sym.expand_dims(data=mx.sym.arange(start=0, stop=seq_len, step=1), axis=0)

        # (1, source_seq_len, num_embed)
        pos_embedding = mx.sym.Embedding(data=positions,
                                         input_dim=self.max_seq_len,
                                         weight=self.embed_weight,
                                         output_dim=self.num_embed,
                                         name=self.prefix + "pos_embed")
        return mx.sym.broadcast_add(data, pos_embedding, name="%s_add" % self.prefix), data_length, seq_len

    def encode_positions(self,
                         positions: mx.sym.Symbol,
                         data: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        :param positions: (batch_size,)
        :param data: (batch_size, num_embed)
        :return: (batch_size, num_embed)
        """

        # (batch_size, source_seq_len, num_embed)
        pos_embedding = mx.sym.Embedding(data=positions,
                                         input_dim=self.max_seq_len,
                                         weight=self.embed_weight,
                                         output_dim=self.num_embed,
                                         name=self.prefix + "pos_embed")
        return mx.sym.broadcast_add(data, pos_embedding, name="%s_add" % self.prefix)

    def get_num_hidden(self) -> int:
        return self.num_embed

    def get_max_seq_len(self) -> Optional[int]:
        # we can only support sentences as long as the maximum length during training.
        return self.max_seq_len


class NoOpPositionalEmbeddings(PositionalEncoder):
    """
    Simple NoOp pos embedding. It does not modify the data, but avoids lots of if statements.

    :param dtype: Data type.
    """

    def __init__(self, num_embed, dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(dtype)
        self.num_embed = num_embed

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        return data, data_length, seq_len

    def encode_positions(self,
                         positions: mx.sym.Symbol,
                         data: mx.sym.Symbol) -> mx.sym.Symbol:
        return data

    def get_num_hidden(self) -> int:
        return self.num_embed


def _get_positional_embedding_params(positional_embedding_type: str,
                                     num_embed: int,
                                     max_seq_len: int,
                                     fixed_pos_embed_scale_up_input: bool = False,
                                     fixed_pos_embed_scale_down_positions: bool = False,
                                     prefix: str = '') -> Tuple[Callable, Dict]:
    if positional_embedding_type == C.FIXED_POSITIONAL_EMBEDDING:
        return AddSinCosPositionalEmbeddings, dict(num_embed=num_embed,
                                                   scale_up_input=fixed_pos_embed_scale_up_input,
                                                   scale_down_positions=fixed_pos_embed_scale_down_positions,
                                                   prefix=prefix)
    elif positional_embedding_type == C.LEARNED_POSITIONAL_EMBEDDING:
        return AddLearnedPositionalEmbeddings, dict(num_embed=num_embed,
                                                    max_seq_len=max_seq_len,
                                                    prefix=prefix)
    elif positional_embedding_type == C.NO_POSITIONAL_EMBEDDING:
        return NoOpPositionalEmbeddings, dict(num_embed=num_embed)
    else:
        raise ValueError("Unknown positional embedding type %s" % positional_embedding_type)


def get_positional_embedding(positional_embedding_type: str,
                             num_embed: int,
                             max_seq_len: int,
                             fixed_pos_embed_scale_up_input: bool = False,
                             fixed_pos_embed_scale_down_positions: bool = False,
                             prefix: str = '') -> PositionalEncoder:
    cls, encoder_params = _get_positional_embedding_params(positional_embedding_type,
                                                           num_embed,
                                                           max_seq_len,
                                                           fixed_pos_embed_scale_up_input,
                                                           fixed_pos_embed_scale_down_positions,
                                                           prefix)
    return cls(**encoder_params)


class EncoderSequence(Encoder):
    """
    A sequence of encoders is itself an encoder.

    :param encoders: List of encoders.
    :param dtype: Data type.
    """

    def __init__(self, encoders: List[Encoder], dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(dtype)
        self.encoders = encoders

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        for encoder in self.encoders:
            data, data_length, seq_len = encoder.encode(data, data_length, seq_len)
        return data, data_length, seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.encoders[-1].get_num_hidden()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        for encoder in self.encoders:
            seq_len = encoder.get_encoded_seq_len(seq_len)
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        max_seq_len = min((encoder.get_max_seq_len()
                           for encoder in self.encoders if encoder.get_max_seq_len() is not None), default=None)
        return max_seq_len

    def append(self, cls, infer_hidden: bool = False, **kwargs) -> Encoder:
        """
        Extends sequence with new Encoder. 'dtype' gets passed into Encoder instance if not present in parameters
        and supported by specific Encoder type.

        :param cls: Encoder type.
        :param infer_hidden: If number of hidden should be inferred from previous encoder.
        :param kwargs: Named arbitrary parameters for Encoder.

        :return: Instance of Encoder.
        """
        params = dict(kwargs)
        if infer_hidden:
            params['num_hidden'] = self.get_num_hidden()

        sig_params = inspect.signature(cls.__init__).parameters
        if 'dtype' in sig_params and 'dtype' not in kwargs:
            params['dtype'] = self.dtype
        encoder = cls(**params)
        self.encoders.append(encoder)
        return encoder


class RecurrentEncoder(Encoder):
    """
    Uni-directional (multi-layered) recurrent encoder.

    :param rnn_config: RNN configuration.
    :param prefix: Prefix for variable names.
    :param layout: Data layout.
    """

    def __init__(self,
                 rnn_config: rnn.RNNConfig,
                 prefix: str = C.STACKEDRNN_PREFIX,
                 layout: str = C.TIME_MAJOR) -> None:
        super().__init__(rnn_config.dtype)
        self.rnn_config = rnn_config
        self.layout = layout
        self.rnn = rnn.get_stacked_rnn(rnn_config, prefix)

    def encode(self,
               data: mx.sym.Symbol,
               data_length: Optional[mx.sym.Symbol],
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        outputs, _ = self.rnn.unroll(seq_len, inputs=data, merge_outputs=True, layout=self.layout)

        return outputs, data_length, seq_len

    def get_rnn_cells(self):
        """
        Returns RNNCells used in this encoder.
        """
        return [self.rnn]

    def get_num_hidden(self):
        """
        Return the representation size of this encoder.
        """
        return self.rnn_config.num_hidden


class BiDirectionalRNNEncoder(Encoder):
    """
    An encoder that runs a forward and a reverse RNN over input data.
    States from both RNNs are concatenated together.

    :param rnn_config: RNN configuration.
    :param prefix: Prefix for variable names.
    :param layout: Data layout.
    :param encoder_class: Recurrent encoder class to use.
    """

    def __init__(self,
                 rnn_config: rnn.RNNConfig,
                 prefix=C.BIDIRECTIONALRNN_PREFIX,
                 layout=C.TIME_MAJOR,
                 encoder_class: Callable = RecurrentEncoder) -> None:
        utils.check_condition(rnn_config.num_hidden % 2 == 0,
                              "num_hidden must be a multiple of 2 for BiDirectionalRNNEncoders.")
        super().__init__(rnn_config.dtype)
        self.rnn_config = rnn_config
        self.internal_rnn_config = rnn_config.copy(num_hidden=rnn_config.num_hidden // 2)
        if layout[0] == 'N':
            logger.warning("Batch-major layout for encoder input. Consider using time-major layout for faster speed")

        # time-major layout as _encode needs to swap layout for SequenceReverse
        self.forward_rnn = encoder_class(rnn_config=self.internal_rnn_config,
                                         prefix=prefix + C.FORWARD_PREFIX,
                                         layout=C.TIME_MAJOR)
        self.reverse_rnn = encoder_class(rnn_config=self.internal_rnn_config,
                                         prefix=prefix + C.REVERSE_PREFIX,
                                         layout=C.TIME_MAJOR)
        self.layout = layout
        self.prefix = prefix

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data (data, data_length, seq_len).
        """
        if self.layout[0] == 'N':
            data = mx.sym.swapaxes(data=data, dim1=0, dim2=1)
        data = self._encode(data, data_length, seq_len)
        if self.layout[0] == 'N':
            data = mx.sym.swapaxes(data=data, dim1=0, dim2=1)
        return data, data_length, seq_len

    def _encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Bidirectionally encodes time-major data.
        """
        # (seq_len, batch_size, num_embed)
        data_reverse = mx.sym.SequenceReverse(data=data, sequence_length=data_length,
                                              use_sequence_length=True)
        # (seq_length, batch, cell_num_hidden)
        hidden_forward, _, _ = self.forward_rnn.encode(data, data_length, seq_len)
        # (seq_length, batch, cell_num_hidden)
        hidden_reverse, _, _ = self.reverse_rnn.encode(data_reverse, data_length, seq_len)
        # (seq_length, batch, cell_num_hidden)
        hidden_reverse = mx.sym.SequenceReverse(data=hidden_reverse, sequence_length=data_length,
                                                use_sequence_length=True)
        # (seq_length, batch, 2 * cell_num_hidden)
        hidden_concat = mx.sym.concat(hidden_forward, hidden_reverse, dim=2, name="%s_rnn" % self.prefix)

        return hidden_concat

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.rnn_config.num_hidden

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        return self.forward_rnn.get_rnn_cells() + self.reverse_rnn.get_rnn_cells()


class ConvolutionalEncoder(Encoder):
    """
    Encoder that uses convolution instead of recurrent connections, similar to Gehring et al. 2017.

    :param config: Configuration for convolutional encoder.
    :param prefix: Name prefix for operations in this encoder.
    """

    def __init__(self,
                 config: ConvolutionalEncoderConfig,
                 prefix: str = C.CNN_ENCODER_PREFIX) -> None:
        super().__init__(config.dtype)
        self.config = config

        # initialize the weights of the linear transformation required for the residual connections
        self.i2h_weight = mx.sym.Variable('%si2h_weight' % prefix)

        # initialize the layers of blocks containing a convolution and a GLU, since
        # every layer is shared over all encode calls
        self.layers = [convolution.ConvolutionBlock(
            config.cnn_config,
            pad_type='centered',
            prefix="%s%d_" % (prefix, i)) for i in range(config.num_layers)]

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data with a stack of Convolution+GLU blocks given sequence lengths of individual examples
        and maximum sequence length.

        :param data: Input data. Shape: (batch_size, seq_len, input_num_hidden).
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded version of the data.
        """
        # data: (batch_size, seq_len, num_hidden)
        data = mx.sym.FullyConnected(data=data,
                                     num_hidden=self.config.cnn_config.num_hidden,
                                     no_bias=True,
                                     flatten=False,
                                     weight=self.i2h_weight)

        # Multiple layers with residual connections:
        for layer in self.layers:
            data = data + layer(data, data_length, seq_len)
        return data, data_length, seq_len

    def get_num_hidden(self) -> int:
        return self.config.cnn_config.num_hidden


class TransformerEncoder(Encoder):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    :param prefix: Name prefix for operations in this encoder.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 prefix: str = C.TRANSFORMER_ENCODER_PREFIX) -> None:
        super().__init__(config.dtype)
        self.config = config
        self.prefix = prefix
        self.layers = [transformer.TransformerEncoderBlock(
            config, prefix="%s%d_" % (prefix, i)) for i in range(config.num_layers)]
        self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                 dropout=config.dropout_prepost,
                                                                 prefix="%sfinal_process_" % prefix)

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data data, data_length, seq_len.
        """
        data = utils.cast_conditionally(data, self.dtype)
        if self.config.dropout_prepost > 0.0:
            data = mx.sym.Dropout(data=data, p=self.config.dropout_prepost)

        # (batch_size * heads, 1, max_length)
        bias = mx.sym.expand_dims(transformer.get_variable_length_bias(lengths=data_length,
                                                                       max_length=seq_len,
                                                                       num_heads=self.config.attention_heads,
                                                                       fold_heads=True,
                                                                       name="%sbias" % self.prefix), axis=1)
        bias = utils.cast_conditionally(bias, self.dtype)
        for i, layer in enumerate(self.layers):
            # (batch_size, seq_len, config.model_size)
            data = layer(data, bias)
        data = self.final_process(data=data, prev=None)
        data = utils.uncast_conditionally(data, self.dtype)
        return data, data_length, seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size


class ConvolutionalEmbeddingConfig(config.Config):
    """
    Convolutional embedding encoder configuration.

    :param num_embed: Input embedding size.
    :param output_dim: Output segment embedding size.
    :param max_filter_width: Maximum filter width for convolutions.
    :param num_filters: Number of filters of each width.
    :param pool_stride: Stride for pooling layer after convolutions.
    :param num_highway_layers: Number of highway layers for segment embeddings.
    :param dropout: Dropout probability.
    :param add_positional_encoding: Dropout probability.
    :param dtype: Data type.
    """

    def __init__(self,
                 num_embed: int,
                 output_dim: int = None,
                 max_filter_width: int = 8,
                 num_filters: Tuple[int, ...] = (200, 200, 250, 250, 300, 300, 300, 300),
                 pool_stride: int = 5,
                 num_highway_layers: int = 4,
                 dropout: float = 0.0,
                 add_positional_encoding: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.num_embed = num_embed
        self.output_dim = output_dim
        self.max_filter_width = max_filter_width
        self.num_filters = num_filters
        self.pool_stride = pool_stride
        self.num_highway_layers = num_highway_layers
        self.dropout = dropout
        self.add_positional_encoding = add_positional_encoding
        if self.output_dim is None:
            self.output_dim = sum(self.num_filters)
        self.dtype = dtype


class ConvolutionalEmbeddingEncoder(Encoder):
    """
    An encoder developed to map a sequence of character embeddings to a shorter sequence of segment
    embeddings using convolutional, pooling, and highway layers.  More generally, it maps a sequence
    of input embeddings to a sequence of span embeddings.

    * "Fully Character-Level Neural Machine Translation without Explicit Segmentation"
      Jason Lee; Kyunghyun Cho; Thomas Hofmann (https://arxiv.org/pdf/1610.03017.pdf)

    :param config: Convolutional embedding config.
    :param prefix: Name prefix for symbols of this encoder.
    """

    def __init__(self,
                 config: ConvolutionalEmbeddingConfig,
                 prefix: str = C.CHAR_SEQ_ENCODER_PREFIX) -> None:
        utils.check_condition(len(config.num_filters) == config.max_filter_width,
                              "num_filters must have max_filter_width elements.")
        super().__init__(config.dtype)
        self.num_embed = config.num_embed
        self.output_dim = config.output_dim
        self.max_filter_width = config.max_filter_width
        self.num_filters = config.num_filters[:]
        self.pool_stride = config.pool_stride
        self.num_highway_layers = config.num_highway_layers
        self.prefix = prefix
        self.dropout = config.dropout
        self.add_positional_encoding = config.add_positional_encoding

        self.conv_weight = {filter_width: mx.sym.Variable("%s%s%d%s" % (self.prefix, "conv_", filter_width, "_weight"))
                            for filter_width in range(1, self.max_filter_width + 1)}
        self.conv_bias = {filter_width: mx.sym.Variable("%s%s%d%s" % (self.prefix, "conv_", filter_width, "_bias"))
                          for filter_width in range(1, self.max_filter_width + 1)}

        self.project_weight = mx.sym.Variable(self.prefix + "project_weight")
        self.project_bias = mx.sym.Variable(self.prefix + "project_bias")

        self.gate_weight = [mx.sym.Variable("%s%s%d%s" % (self.prefix, "gate_", i, "_weight"))
                            for i in range(self.num_highway_layers)]
        self.gate_bias = [mx.sym.Variable("%s%s%d%s" % (self.prefix, "gate_", i, "_bias"))
                          for i in range(self.num_highway_layers)]

        self.transform_weight = [mx.sym.Variable("%s%s%d%s" % (self.prefix, "transform_", i, "_weight"))
                                 for i in range(self.num_highway_layers)]
        self.transform_bias = [mx.sym.Variable("%s%s%d%s" % (self.prefix, "transform_", i, "_bias"))
                               for i in range(self.num_highway_layers)]

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data data, data_length, seq_len.
        """
        total_num_filters = sum(self.num_filters)
        encoded_seq_len = self.get_encoded_seq_len(seq_len)

        # (batch_size, channel=1, seq_len, num_embed)
        data = mx.sym.Reshape(data=data, shape=(-1, 1, seq_len, self.num_embed))

        # Convolution filters of width 1..N
        conv_outputs = []
        for filter_width, num_filter in enumerate(self.num_filters, 1):
            # "half" padding: output length == input length
            pad_before = ceil((filter_width - 1) / 2)
            pad_after = floor((filter_width - 1) / 2)
            # (batch_size, channel=1, seq_len + (filter_width - 1), num_embed)
            padded = mx.sym.pad(data=data,
                                mode="constant",
                                constant_value=0,
                                pad_width=(0, 0, 0, 0, pad_before, pad_after, 0, 0))
            # (batch_size, num_filter, seq_len, num_scores=1)
            conv = mx.sym.Convolution(data=padded,
                                      # cudnn_tune="off",
                                      kernel=(filter_width, self.num_embed),
                                      num_filter=num_filter,
                                      weight=self.conv_weight[filter_width],
                                      bias=self.conv_bias[filter_width])
            conv = mx.sym.Activation(data=conv, act_type="relu")
            conv_outputs.append(conv)
        # (batch_size, total_num_filters, seq_len, num_scores=1)
        conv_concat = mx.sym.concat(*conv_outputs, dim=1)

        # Max pooling with stride
        uncovered = seq_len % self.pool_stride
        if uncovered > 0:
            pad_after = self.pool_stride - uncovered
            # (batch_size, total_num_filters, seq_len + pad_to_final_stride, num_scores=1)
            conv_concat = mx.sym.pad(data=conv_concat,
                                     mode="constant",
                                     constant_value=0,
                                     pad_width=(0, 0, 0, 0, 0, pad_after, 0, 0))
        # (batch_size, total_num_filters, seq_len/stride, num_scores=1)
        pool = mx.sym.Pooling(data=conv_concat,
                              pool_type="max",
                              kernel=(self.pool_stride, 1),
                              stride=(self.pool_stride, 1))
        # (batch_size, total_num_filters, seq_len/stride)
        pool = mx.sym.reshape(data=pool,
                              shape=(-1, total_num_filters, encoded_seq_len))
        # (batch_size, seq_len/stride, total_num_filters)
        pool = mx.sym.swapaxes(data=pool, dim1=1, dim2=2)
        if self.dropout > 0:
            pool = mx.sym.Dropout(data=pool, p=self.dropout)

        # Raw segment embeddings reshaped for highway network
        # (batch_size * seq_len/stride, total_num_filters)
        seg_embedding = mx.sym.Reshape(data=pool, shape=(-3, total_num_filters))

        # Projection layer if requested output dimension is different from total number of filters
        # (TransformerEncoder compatibility, not in original paper)
        if self.output_dim != total_num_filters:
            # (batch_size * seq_len/stride, outut_dim)
            seg_embedding = mx.sym.FullyConnected(data=seg_embedding,
                                                  num_hidden=self.output_dim,
                                                  weight=self.project_weight,
                                                  bias=self.project_bias)
            seg_embedding = mx.sym.Activation(data=seg_embedding, act_type="relu")
            if self.dropout > 0:
                seg_embedding = mx.sym.Dropout(data=seg_embedding, p=self.dropout)

        # Highway network
        for i in range(self.num_highway_layers):
            # Gate
            gate = mx.sym.FullyConnected(data=seg_embedding,
                                         num_hidden=self.output_dim,
                                         weight=self.gate_weight[i],
                                         bias=self.gate_bias[i])
            gate = mx.sym.Activation(data=gate, act_type="sigmoid")
            if self.dropout > 0:
                gate = mx.sym.Dropout(data=gate, p=self.dropout)
            # Transform
            transform = mx.sym.FullyConnected(data=seg_embedding,
                                              num_hidden=self.output_dim,
                                              weight=self.transform_weight[i],
                                              bias=self.transform_bias[i])
            transform = mx.sym.Activation(data=transform, act_type="relu")
            if self.dropout > 0:
                transform = mx.sym.Dropout(data=transform, p=self.dropout)
            # Connection
            seg_embedding = gate * transform + (1 - gate) * seg_embedding
        # (batch_size, seq_len/stride, outut_dim) aka
        # (batch_size, encoded_seq_len, num_segment_emded)
        seg_embedding = mx.sym.Reshape(data=seg_embedding,
                                       shape=(-1, encoded_seq_len, self.output_dim))

        # Dropout on final segment embeddings
        if self.dropout > 0:
            seg_embedding = mx.sym.Dropout(data=seg_embedding, p=self.dropout)

        # Ceiling function isn't differentiable so this will throw errors if we
        # attempt to compute gradients.  Fortunately we aren't updating inputs
        # so we can just block the backward pass here.
        encoded_data_length = mx.sym.BlockGrad(mx.sym.ceil(data_length / self.pool_stride))

        return seg_embedding, encoded_data_length, encoded_seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.output_dim

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        return int(ceil(seq_len / self.pool_stride))


EncoderConfig = Union[RecurrentEncoderConfig, transformer.TransformerConfig, ConvolutionalEncoderConfig]
if ImageEncoderConfig is not None:
    EncoderConfig = Union[EncoderConfig, ImageEncoderConfig]  # type: ignore
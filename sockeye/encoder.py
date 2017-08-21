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
import logging
from math import ceil, floor
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import mxnet as mx

from sockeye.config import Config
from . import constants as C
from . import rnn
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


def get_encoder(config: Config, fused: bool, embed_weight: Optional[mx.sym.Symbol] = None):
    if isinstance(config, RecurrentEncoderConfig):
        return get_recurrent_encoder(config, fused, embed_weight)
    elif isinstance(config, transformer.TransformerConfig):
        return get_transformer_encoder(config, embed_weight)
    else:
        raise ValueError("Unsupported encoder configuration")


class RecurrentEncoderConfig(Config):
    """
    Recurrent encoder configuration.

    :param vocab_size: Source vocabulary size.
    :param num_embed: Size of embedding layer.
    :param embed_dropout: Dropout probability on embedding layer.
    :param rnn_config: RNN configuration.
    :param conv_config: Optional configuration for convolutional embedding.
    :param reverse_input: Reverse embedding sequence before feeding into RNN.
    """

    def __init__(self,
                 vocab_size: int,
                 num_embed: int,
                 embed_dropout: float,
                 rnn_config: rnn.RNNConfig,
                 conv_config: Optional['ConvolutionalEmbeddingConfig'] = None,
                 reverse_input: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.embed_dropout = embed_dropout
        self.rnn_config = rnn_config
        self.conv_config = conv_config
        self.reverse_input = reverse_input


def get_recurrent_encoder(config: RecurrentEncoderConfig, fused: bool,
                          embed_weight: Optional[mx.sym.Symbol] = None) -> 'Encoder':
    """
    Returns a recurrent encoder with embedding, batch2time-major conversion, and bidirectional RNN.
    If num_layers > 1, adds additional uni-directional RNNs.

    :param config: Configuration for recurrent encoder.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new one.
    :return: Encoder instance.
    """
    # TODO give more control on encoder architecture
    encoders = list()  # type: List[Encoder]

    encoders.append(Embedding(num_embed=config.num_embed,
                              vocab_size=config.vocab_size,
                              prefix=C.SOURCE_EMBEDDING_PREFIX,
                              dropout=config.embed_dropout,
                              embed_weight=embed_weight))

    if config.conv_config is not None:
        encoders.append(ConvolutionalEmbeddingEncoder(config.conv_config))

    encoders.append(BatchMajor2TimeMajor())
    
    if config.reverse_input:
        encoders.append(ReverseSequence())
    
    if config.rnn_config.residual:
        utils.check_condition(config.rnn_config.first_residual_layer >= 2,
                              "Residual connections on the first encoder layer are not supported")

    encoder_class = FusedRecurrentEncoder if fused else RecurrentEncoder
    # One layer bi-directional RNN:
    encoders.append(BiDirectionalRNNEncoder(rnn_config=config.rnn_config.copy(num_layers=1),
                                            prefix=C.BIDIRECTIONALRNN_PREFIX,
                                            layout=C.TIME_MAJOR))

    if config.rnn_config.num_layers > 1:
        # Stacked uni-directional RNN:
        # Because we already have a one layer bi-rnn we reduce the num_layers as well as the first_residual_layer.
        remaining_rnn_config = config.rnn_config.copy(num_layers=config.rnn_config.num_layers - 1,
                                                      first_residual_layer=config.rnn_config.first_residual_layer - 1)
        encoders.append(encoder_class(rnn_config=remaining_rnn_config,
                                      prefix=C.STACKEDRNN_PREFIX,
                                      layout=C.TIME_MAJOR))

    return EncoderSequence(encoders)


def get_transformer_encoder(config: transformer.TransformerConfig,
                            embed_weight: Optional[mx.sym.Symbol] = None) -> 'Encoder':
    """
    Returns a Transformer encoder, consisting of an embedding layer with
    positional encodings and a TransformerEncoder instance.

    :param config: Configuration for transformer encoder.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new one.
    :return: Encoder instance.
    """
    encoders = list()  # type: List[Encoder]
    encoders.append(Embedding(num_embed=config.model_size,
                              vocab_size=config.vocab_size,
                              prefix=C.SOURCE_EMBEDDING_PREFIX,
                              dropout=config.dropout_residual,
                              embed_weight=embed_weight,
                              add_positional_encoding=config.positional_encodings))
    if config.conv_config is not None:
        encoders.append(ConvolutionalEmbeddingEncoder(config.conv_config))

    encoders.append(TransformerEncoder(config))
    encoders.append(BatchMajor2TimeMajor())

    return EncoderSequence(encoders)


class Encoder(ABC):
    """
    Generic encoder interface.
    """

    @abstractmethod
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
        pass

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        raise NotImplementedError()

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        return []

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        return seq_len


class BatchMajor2TimeMajor(Encoder):
    """
    Converts batch major data to time major.
    """

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
        with mx.AttrScope(__layout__=C.TIME_MAJOR):
            return mx.sym.swapaxes(data=data, dim1=0, dim2=1), data_length, seq_len


class ReverseSequence(Encoder):
    """
    Reverses the input sequence. Requires time-major layout.
    """

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        data = mx.sym.SequenceReverse(data=data, sequence_length=data_length, use_sequence_length=True)
        return data, data_length, seq_len


class Embedding(Encoder):
    """
    Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.

    :param num_embed: Embedding size.
    :param vocab_size: Source vocabulary size.
    :param prefix: Name prefix for symbols of this encoder.
    :param dropout: Dropout probability.
    :param embed_weight: Optionally use an existing embedding matrix instead of creating a new one.
    :param add_positional_encoding: If true, adds positional encodings to embedding.
    """

    def __init__(self,
                 num_embed: int,
                 vocab_size: int,
                 prefix: str,
                 dropout: float,
                 embed_weight: Optional[mx.sym.Symbol] = None,
                 add_positional_encoding: bool = False) -> None:
        self.num_embed = num_embed
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.dropout = dropout
        if embed_weight is not None:
            self.embed_weight = embed_weight
        else:
            self.embed_weight = mx.sym.Variable(prefix + "weight")
        self.add_positional_encoding = add_positional_encoding

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
        embedding = mx.sym.Embedding(data=data,
                                     input_dim=self.vocab_size,
                                     weight=self.embed_weight,
                                     output_dim=self.num_embed,
                                     name=self.prefix + "embed")
        if self.add_positional_encoding:
            embedding = mx.sym.broadcast_add(embedding,
                                             self.get_positional_encoding(length=seq_len,
                                                                          depth=self.num_embed,
                                                                          name="%spositional_encodings" % self.prefix),
                                             name='%sadd_positional_encodings' % self.prefix)
        if self.dropout > 0:
            embedding = mx.sym.Dropout(data=embedding, p=self.dropout, name="source_embed_dropout")
        return embedding, data_length, seq_len

    @staticmethod
    def get_positional_encoding(length: int, depth: int, name: str) -> mx.sym.Symbol:
        """
        Returns symbol initialized with positional encodings as in Vaswani et al.

        :param length: Maximum sequence length
        :param depth: Depth.
        :param name: Symbol name.
        :return: Symbol(1, length, depth)
        """
        return mx.sym.BlockGrad(mx.symbol.Custom(length=length, depth=depth, name=name,
                                                 op_type='positional_encodings'))

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.num_embed


class EncoderSequence(Encoder):
    """
    A sequence of encoders is itself an encoder.

    :param encoders: List of encoders.
    """

    def __init__(self, encoders: List[Encoder]) -> None:
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
        if isinstance(self.encoders[-1], BatchMajor2TimeMajor):
            utils.check_condition(len(self.encoders) > 1,
                                  "Cannot return num_hidden from a BatchMajor2TimeMajor encoder only")
            return self.encoders[-2].get_num_hidden()
        else:
            return self.encoders[-1].get_num_hidden()

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        cells = []
        for encoder in self.encoders:
            for cell in encoder.get_rnn_cells():
                cells.append(cell)
        return cells

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        for encoder in self.encoders:
            seq_len = encoder.get_encoded_seq_len(seq_len)
        return seq_len


class RecurrentEncoder(Encoder):
    """
    Uni-directional (multi-layered) recurrent encoder.

    :param rnn_config: RNN configuration.
    :param prefix: Prefix.
    :param layout: Data layout.
    """

    def __init__(self,
                 rnn_config: rnn.RNNConfig,
                 prefix: str = C.STACKEDRNN_PREFIX,
                 layout: str = C.TIME_MAJOR) -> None:
        self.rnn_config = rnn_config
        self.layout = layout
        self.rnn = rnn.get_stacked_rnn(rnn_config, prefix)

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


class FusedRecurrentEncoder(RecurrentEncoder):
    """
    Uni-directional (multi-layered) recurrent encoder.

    :param rnn_config: RNN configuration.
    :param prefix: Prefix.
    :param layout: Data layout.
    """

    def __init__(self,
                 rnn_config: rnn.RNNConfig,
                 prefix: str = C.STACKEDRNN_PREFIX,
                 layout: str = C.TIME_MAJOR) -> None:
        super().__init__(rnn_config, prefix, layout)
        logger.warning("%s: FusedRNNCell uses standard MXNet Orthogonal initializer w/ rand_type=uniform", prefix)
        self.rnn = mx.rnn.FusedRNNCell(self.rnn_config.num_hidden,
                                       num_layers=self.rnn_config.num_layers,
                                       mode=self.rnn_config.cell_type,
                                       bidirectional=False,
                                       dropout=self.rnn_config.dropout,
                                       forget_bias=self.rnn_config.forget_bias,
                                       prefix=prefix)


class BiDirectionalRNNEncoder(Encoder):
    """
    An encoder that runs a forward and a reverse RNN over input data.
    States from both RNNs are concatenated together.

    :param rnn_config: RNN configuration.
    :param prefix: Prefix.
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
        self.config = config
        self.prefix = prefix
        self.layers = [transformer.TransformerEncoderBlock(
            config, prefix="%s%d_" % (prefix, i)) for i in range(config.num_layers)]

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
        for i, layer in enumerate(self.layers):
            # (batch_size, seq_len, config.model_size)
            data = layer(data, data_length, seq_len)
        return data, data_length, seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size


class ConvolutionalEmbeddingConfig(Config):
    """
    Convolutional embedding encoder configuration.

    :param num_embed: Input embedding size.
    :param output_dim: Output segment embedding size.
    :param max_filter_width: Maximum filter width for convolutions.
    :param num_filters: Number of filters of each width.
    :param pool_stride: Stride for pooling layer after convolutions.
    :param num_highway_layers: Number of highway layers for segment embeddings.
    :param dropout: Dropout probability.
    """

    def __init__(self,
                 num_embed: int,
                 output_dim: int = None,
                 max_filter_width: int = 8,
                 num_filters: Tuple[int, ...] = (200, 200, 250, 250, 300, 300, 300, 300),
                 pool_stride: int = 5,
                 num_highway_layers: int = 4,
                 dropout: float = 0.0,
                 add_positional_encoding: bool = False) -> None:
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

        # If specified, add positional encodings to segment embeddings
        # (TransformerEncoder compatibility, not in original paper)
        if self.add_positional_encoding:
            seg_embedding = mx.sym.broadcast_add(seg_embedding,
                                                 Embedding.get_positional_encoding(
                                                     length=encoded_seq_len,
                                                     depth=self.output_dim,
                                                     name="%spositional_encodings" % self.prefix),
                                                 name='%sadd_positional_encodings' % self.prefix)

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

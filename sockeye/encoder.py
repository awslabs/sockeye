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
Defines Encoder interface and various implementations.
"""
import logging

from math import ceil, floor
from typing import Callable, List, Tuple

import mxnet as mx

import sockeye.constants as C
import sockeye.rnn
import sockeye.utils
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


# TODO break out EncoderConfig to allow use without populating options for full translation model
def get_encoder(config: "ModelConfig",
                forget_bias: float,
                fused: bool) -> 'Encoder':
    """
    Returns an encoder with embedding, batch2time-major conversion, and bidirectional RNN encoder.
    If num_layers > 1, adds uni-directional RNNs.

    :param config: ModelConfig populated by command line args and/or defaults.
    :param forget_bias: Initial value of RNN forget biases.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :return: Encoder instance.
    """
    # TODO give more control on encoder architecture
    encoders = list()

    encoders.append(Embedding(num_embed=config.num_embed_source,
                              vocab_size=config.vocab_source_size,
                              prefix=C.SOURCE_EMBEDDING_PREFIX,
                              dropout=config.dropout))

    if config.encoder == C.RNN_WITH_CONV_EMBED_NAME:
        encoders.append(ConvolutionalEmbeddingEncoder(num_embed=config.num_embed_source,
                                                      max_filter_width=config.conv_embed_max_filter_width,
                                                      num_filters=config.conv_embed_num_filters,
                                                      pool_stride=config.conv_embed_pool_stride,
                                                      num_highway_layers=config.conv_embed_num_highway_layers,
                                                      prefix=C.CHAR_SEQ_ENCODER_PREFIX,
                                                      dropout=config.dropout))

    encoders.append(BatchMajor2TimeMajor())

    encoder_class = FusedRecurrentEncoder if fused else RecurrentEncoder
    encoders.append(BiDirectionalRNNEncoder(num_hidden=config.rnn_num_hidden,
                                            num_layers=1,
                                            dropout=config.dropout,
                                            layout=C.TIME_MAJOR,
                                            cell_type=config.rnn_cell_type,
                                            encoder_class=encoder_class,
                                            forget_bias=forget_bias))

    if config.rnn_num_layers > 1:
        encoders.append(encoder_class(num_hidden=config.rnn_num_hidden,
                                      num_layers=config.rnn_num_layers - 1,
                                      dropout=config.dropout,
                                      layout=C.TIME_MAJOR,
                                      cell_type=config.rnn_cell_type,
                                      residual=config.rnn_residual_connections,
                                      forget_bias=forget_bias))

    return EncoderSequence(encoders)


class Encoder:
    """
    Generic encoder interface.
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
        raise NotImplementedError()

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        raise NotImplementedError()

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        raise NotImplementedError()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        return seq_len


class BatchMajor2TimeMajor(Encoder):
    """
    Converts batch major data to time major
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
        return self._encode(data), data_length, seq_len

    def _encode(self, data: mx.sym.Symbol) -> mx.sym.Symbol:
        with mx.AttrScope(__layout__=C.TIME_MAJOR):
            return mx.sym.swapaxes(data=data, dim1=0, dim2=1)

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return 0

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        return []


class Embedding(Encoder):
    """
    Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.

    :param num_embed: Embedding size.
    :param vocab_size: Source vocabulary size.
    :param prefix: Name prefix for symbols of this encoder.
    :param dropout: Dropout probability.
    """

    def __init__(self, num_embed: int, vocab_size: int, prefix: str, dropout: float):
        self.num_embed = num_embed
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.dropout = dropout
        self.embed_weight = mx.sym.Variable(prefix + "weight")

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
                                     name=self.prefix + 'embed')
        if self.dropout > 0:
            embedding = mx.sym.Dropout(data=embedding, p=self.dropout, name="source_embed_dropout")
        return embedding, data_length, seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.num_embed

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        return []


class EncoderSequence(Encoder):
    """
    A sequence of encoders is itself an encoder.

    :param encoders: List of encoders.
    """

    def __init__(self, encoders: List[Encoder]):
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
    Uni-directional (multi-layered) recurrent encoder
    """

    def __init__(self,
                 num_hidden: int,
                 num_layers: int,
                 prefix: str = C.STACKEDRNN_PREFIX,
                 dropout: float = 0.,
                 layout: str = C.TIME_MAJOR,
                 cell_type: str = C.LSTM_TYPE,
                 residual: bool = False,
                 forget_bias=0.0):
        self.layout = layout
        self.num_hidden = num_hidden
        self.rnn = sockeye.rnn.get_stacked_rnn(cell_type, num_hidden,
                                               num_layers, dropout, prefix,
                                               residual, forget_bias)

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
        return self.num_hidden


class FusedRecurrentEncoder(Encoder):
    """
    Uni-directional (multi-layered) recurrent encoder
    """

    def __init__(self,
                 num_hidden: int,
                 num_layers: int,
                 prefix: str = C.STACKEDRNN_PREFIX,
                 dropout: float = 0.,
                 layout: str = C.TIME_MAJOR,
                 cell_type: str = C.LSTM_TYPE,
                 residual: bool = False,
                 forget_bias=0.0):
        self.layout = layout
        self.num_hidden = num_hidden
        logger.warning("%s: FusedRNNCell uses standard MXNet Orthogonal initializer w/ rand_type=uniform", prefix)
        self.rnn = [mx.rnn.FusedRNNCell(num_hidden,
                                        num_layers=num_layers,
                                        mode=cell_type,
                                        bidirectional=False,
                                        dropout=dropout,
                                        forget_bias=forget_bias,
                                        prefix=prefix)]

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
        outputs = data
        for cell in self.rnn:
            outputs, _ = cell.unroll(seq_len, inputs=outputs, merge_outputs=True, layout=self.layout)

        return outputs, data_length, seq_len

    def get_rnn_cells(self):
        """
        Returns RNNCells used in this encoder.
        """
        return self.rnn

    def get_num_hidden(self):
        """
        Return the representation size of this encoder.
        """
        return self.num_hidden


class BiDirectionalRNNEncoder(Encoder):
    """
    An encoder that runs a forward and a reverse RNN over input data.
    States from both RNNs are concatenated together.

    :param num_hidden: Number of hidden units for final, concatenated encoder states. Must be a multiple of 2.
    :param num_layers: Number of RNN layers.
    :param prefix: Name prefix for symbols of this encoder.
    :param dropout: Dropout probability.
    :param layout: Input data layout. Default: time-major.
    :param cell_type: RNN cell type.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :param forget_bias: Initial value of RNN forget biases.
    """

    def __init__(self,
                 num_hidden: int,
                 num_layers: int,
                 prefix=C.BIDIRECTIONALRNN_PREFIX,
                 dropout: float = 0.,
                 layout=C.TIME_MAJOR,
                 cell_type=C.LSTM_TYPE,
                 encoder_class: Callable = RecurrentEncoder,
                 forget_bias: float = 0.0):
        check_condition(num_hidden % 2 == 0, "num_hidden must be a multiple of 2 for BiDirectionalRNNEncoders.")
        self.num_hidden = num_hidden
        if layout[0] == 'N':
            logger.warning("Batch-major layout for encoder input. Consider using time-major layout for faster speed")

        # time-major layout as _encode needs to swap layout for SequenceReverse
        self.forward_rnn = encoder_class(num_hidden=num_hidden // 2, num_layers=num_layers,
                                         prefix=prefix + C.FORWARD_PREFIX, dropout=dropout,
                                         layout=C.TIME_MAJOR, cell_type=cell_type,
                                         forget_bias=forget_bias)
        self.reverse_rnn = encoder_class(num_hidden=num_hidden // 2, num_layers=num_layers,
                                         prefix=prefix + C.REVERSE_PREFIX, dropout=dropout,
                                         layout=C.TIME_MAJOR, cell_type=cell_type,
                                         forget_bias=forget_bias)
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
        return self.num_hidden

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        return self.forward_rnn.get_rnn_cells() + self.reverse_rnn.get_rnn_cells()


class ConvolutionalEmbeddingEncoder(Encoder):
    """
    An encoder developed to map a sequence of character embeddings to a shorter sequence of segment
    embeddings using convolutional, pooling, and highway layers.  More generally, it maps a sequence
    of input embeddings to a sequence of span embeddings.
        * "Fully Character-Level Neural Machine Translation without Explicit Segmentation"
          Jason Lee; Kyunghyun Cho; Thomas Hofmann (https://arxiv.org/pdf/1610.03017.pdf)

    :param num_embed: Input embedding size.
    :param max_filter_width: Maximum filter width for convolutions.
    :param num_filters: Number of filters of each width.
    :param pool_stride: Stride for pooling layer after convolutions.
    :param num_highway_layers: Number of highway layers for segment embeddings.
    :param prefix: Name prefix for symbols of this encoder.
    :param dropout: Dropout probability.
    """

    def __init__(self,
                 num_embed: int,
                 max_filter_width: int = 8,
                 num_filters: List[int] = None,
                 pool_stride: int = 5,
                 num_highway_layers: int = 4,
                 prefix: str = C.CHAR_SEQ_ENCODER_PREFIX,
                 dropout: float = 0.):
        if not num_filters:
            num_filters = [200, 200, 250, 250, 300, 300, 300, 300]
        check_condition(len(num_filters) == max_filter_width, "num_filters must have max_filter_width elements.")
        self.num_embed = num_embed
        self.max_filter_width = max_filter_width
        self.num_filters = num_filters[:]
        self.pool_stride = pool_stride
        self.num_highway_layers = num_highway_layers
        self.prefix = prefix
        self.dropout = dropout
        self.prefix = prefix

        self.conv_weight = {filter_width: mx.sym.Variable("%s%s%d%s" % (self.prefix, "conv_", filter_width, "_weight"))
                            for filter_width in range(1, self.max_filter_width + 1)}
        self.conv_bias = {filter_width: mx.sym.Variable("%s%s%d%s" % (self.prefix, "conv_", filter_width, "_bias"))
                          for filter_width in range(1, self.max_filter_width + 1)}

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
        :return: Encoded versions of input data (data, data_length, seq_len).
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
                                      #cudnn_tune="off",
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

        # Highway network
        # (batch_size * seq_len/stride, total_num_filters)
        seg_embedding = mx.sym.Reshape(data=pool, shape=(-3, total_num_filters))
        for i in range(self.num_highway_layers):
            # Gate
            gate = mx.sym.FullyConnected(data=seg_embedding,
                                         num_hidden=total_num_filters,
                                         weight=self.gate_weight[i],
                                         bias=self.gate_bias[i])
            gate = mx.sym.Activation(data=gate, act_type="sigmoid")
            if self.dropout > 0:
                gate = mx.sym.Dropout(data=gate, p=self.dropout)
            # Transform
            transform = mx.sym.FullyConnected(data=seg_embedding,
                                              num_hidden=total_num_filters,
                                              weight=self.transform_weight[i],
                                              bias=self.transform_bias[i])
            transform = mx.sym.Activation(data=transform, act_type="relu")
            if self.dropout > 0:
                transform = mx.sym.Dropout(data=transform, p=self.dropout)
            # Connection
            seg_embedding = gate * transform + (1 - gate) * seg_embedding
        # (batch_size, seq_len/stride, total_num_filters) aka
        # (batch_size, encoded_seq_len, num_segment_emded)
        seg_embedding = mx.sym.Reshape(data=seg_embedding,
                                       shape=(-1, encoded_seq_len, total_num_filters))

        # Ceiling function isn't differentiable so this will throw errors if we
        # attempt to compute gradients.  Fortunately we aren't updating inputs
        # so we can just block the backward pass here.
        encoded_data_length = mx.sym.BlockGrad(mx.sym.ceil(data_length / self.pool_stride))

        return seg_embedding, encoded_data_length, encoded_seq_len

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return sum(self.num_filters)

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this encoder.
        """
        return []

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        return ceil(seq_len / self.pool_stride)

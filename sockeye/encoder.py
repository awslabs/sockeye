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
from typing import List

import mxnet as mx

import sockeye.constants as C
import sockeye.rnn
import sockeye.utils

logger = logging.getLogger(__name__)


def get_encoder(num_embed: int,
                vocab_size: int,
                num_layers: int,
                rnn_num_hidden: int,
                cell_type: str,
                residual: bool,
                dropout: float,
                forget_bias: float,
                fused: bool = False) -> 'Encoder':
    """
    Returns an encoder with embedding, batch2time-major conversion, and bidirectional RNN encoder.
    If num_layers > 1, adds uni-directional RNNs.

    :param num_embed: Size of embedding layer.
    :param vocab_size: Source vocabulary size.
    :param num_layers: Number of encoder layers.
    :param rnn_num_hidden: Number of hidden units for RNN cells.
    :param cell_type: RNN cell type.
    :param residual: Whether to add residual connections to multi-layered RNNs.
    :param dropout: Dropout probability for encoders (RNN and embedding).
    :param forget_bias: Initial value of RNN forget biases.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :return: Encoder instance.
    """
    # TODO give more control on encoder architecture
    encoders = list()
    encoders.append(Embedding(num_embed=num_embed,
                              vocab_size=vocab_size,
                              prefix=C.SOURCE_EMBEDDING_PREFIX,
                              dropout=dropout))
    encoders.append(BatchMajor2TimeMajor())

    EncoderClass = FusedRecurrentEncoder if fused else RecurrentEncoder
    encoders.append(BiDirectionalRNNEncoder(num_hidden=rnn_num_hidden,
                                            num_layers=1,
                                            dropout=dropout,
                                            layout=C.TIME_MAJOR,
                                            cell_type=cell_type,
                                            EncoderClass=EncoderClass,
                                            forget_bias=forget_bias))

    if num_layers > 1:
        encoders.append(EncoderClass(num_hidden=rnn_num_hidden,
                                     num_layers=num_layers - 1,
                                     dropout=0.,
                                     layout=C.TIME_MAJOR,
                                     cell_type=cell_type,
                                     residual=residual,
                                     forget_bias=forget_bias))

    return EncoderSequence(encoders)


class Encoder:
    """
    Generic encoder interface.
    """

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.
        
        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
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


class BatchMajor2TimeMajor(Encoder):
    """
    Converts batch major data to time major
    """

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples (data_length) and maximum sequence length (seq_len).

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
        """
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

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
        """
        embedding = mx.sym.Embedding(data=data,
                                     input_dim=self.vocab_size,
                                     weight=self.embed_weight,
                                     output_dim=self.num_embed,
                                     name=self.prefix + 'embed')
        if self.dropout > 0:
            embedding = mx.sym.Dropout(data=embedding, p=self.dropout, name="source_embed_dropout")
        return embedding

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

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
        """
        for encoder in self.encoders:
            data = encoder.encode(data, data_length, seq_len)
        return data

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

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
        """
        outputs, _ = self.rnn.unroll(seq_len, inputs=data, merge_outputs=True, layout=self.layout)

        return outputs

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
                                        forget_bias=forget_bias,
                                        prefix=prefix)]

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
        """
        outputs = data
        for cell in self.rnn:
            outputs, _ = cell.unroll(seq_len, inputs=outputs, merge_outputs=True, layout=self.layout)

        return outputs

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
                 EncoderClass: Encoder = RecurrentEncoder,
                 forget_bias: float = 0.0):
        assert num_hidden % 2 == 0, "num_hidden must be a multiple of 2 for BiDirectionalRNNEncoders."
        self.num_hidden = num_hidden
        if layout[0] == 'N':
            logger.warning("Batch-major layout for encoder input. Consider using time-major layout for faster speed")

        # time-major layout as _encode needs to swap layout for SequenceReverse
        self.forward_rnn = EncoderClass(num_hidden=num_hidden // 2, num_layers=num_layers,
                                        prefix=prefix + C.FORWARD_PREFIX, dropout=dropout,
                                        layout=C.TIME_MAJOR, cell_type=cell_type,
                                        forget_bias=forget_bias)
        self.reverse_rnn = EncoderClass(num_hidden=num_hidden // 2, num_layers=num_layers,
                                        prefix=prefix + C.REVERSE_PREFIX, dropout=dropout,
                                        layout=C.TIME_MAJOR, cell_type=cell_type,
                                        forget_bias=forget_bias)
        self.layout = layout
        self.prefix = prefix

    def encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Input data.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded input data.
        """
        if self.layout[0] == 'N':
            data = mx.sym.swapaxes(data=data, dim1=0, dim2=1)
        data = self._encode(data, data_length, seq_len)
        if self.layout[0] == 'N':
            data = mx.sym.swapaxes(data=data, dim1=0, dim2=1)
        return data

    def _encode(self, data: mx.sym.Symbol, data_length: mx.sym.Symbol, seq_len: int) -> mx.sym.Symbol:
        """
        Bidirectionally encodes time-major data.
        """
        # (seq_len, batch_size, num_embed)
        data_reverse = mx.sym.SequenceReverse(data=data, sequence_length=data_length,
                                              use_sequence_length=True)
        # (seq_length, batch, cell_num_hidden)
        hidden_forward = self.forward_rnn.encode(data, data_length, seq_len)
        # (seq_length, batch, cell_num_hidden)
        hidden_reverse = self.reverse_rnn.encode(data_reverse, data_length, seq_len)
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

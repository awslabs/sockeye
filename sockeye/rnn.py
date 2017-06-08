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

import mxnet as mx

from sockeye import constants as C


def get_stacked_rnn(cell_type: str,
                    num_hidden: int,
                    num_layers: int,
                    dropout: float,
                    prefix: str,
                    residual: bool = False,
                    forget_bias: float = 0.0) -> mx.rnn.SequentialRNNCell:
    """
    Returns (stacked) RNN cell given parameters.

    :param cell_type: RNN cell type.
    :param num_hidden: Number of RNN hidden units.
    :param num_layers: Number of RNN layers.
    :param dropout: Dropout probability on RNN outputs.
    :param prefix: Symbol prefix for RNN.
    :param residual: Whether to add residual connections between multi-layered RNNs.
    :param forget_bias: Initial value of forget biases.
    :return: RNN cell.
    """

    rnn = mx.rnn.SequentialRNNCell()
    for layer in range(num_layers):
        # fhieber: the 'l' in the prefix does NOT stand for 'layer' but for the direction 'l' as in mx.rnn.rnn_cell::517
        # this ensures parameter name compatibility of training w/ FusedRNN and decoding with 'unfused' RNN.
        cell_prefix = "%sl%d_" % (prefix, layer)
        if cell_type == C.LSTM_TYPE:
            cell = mx.rnn.LSTMCell(num_hidden=num_hidden, prefix=cell_prefix, forget_bias=forget_bias)
        elif cell_type == C.GRU_TYPE:
            cell = mx.rnn.GRUCell(num_hidden=num_hidden, prefix=cell_prefix)
        else:
            raise NotImplementedError()
        if residual and layer > 0:
            cell = mx.rnn.ResidualCell(cell)
        rnn.add(cell)

        if dropout > 0.:
            # TODO(fhieber): add pervasive dropout?
            rnn.add(mx.rnn.DropoutCell(dropout, prefix=cell_prefix + "_dropout"))
    return rnn

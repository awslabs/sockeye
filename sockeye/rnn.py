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

from typing import List, Optional

import mxnet as mx

from sockeye import constants as C
from sockeye.layers import LayerNormalization


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
        elif cell_type == C.LNLSTM_TYPE:
            cell = LnLSTMCell(num_hidden=num_hidden, prefix=cell_prefix, forget_bias=forget_bias)
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


class LnLSTMCell(mx.rnn.LSTMCell):
    """
    Long-Short Term Memory (LSTM) network cell with layer normalization.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param forget_bias: bias added to forget gate, default 1.0. Jozefowicz et al. 2015 recommends setting this to 1.0.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lstm_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 forget_bias: float = 1.0,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LnLSTMCell, self).__init__(num_hidden, prefix, params, forget_bias)

        self._norm_layers = list()  # type: List[LayerNormalization]
        for name in ['i', 'f', 'c', 'o', 's']:
            scale = self.params.get('%s_shift' % name, shape=(num_hidden,),
                                    init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name, shape=(num_hidden,),
                                    init=mx.init.Constant(value=norm_scale))
            self._norm_layers.append(
                LayerNormalization(num_hidden, prefix="%s%s" % (self._prefix, name), scale=scale, shift=shift))

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_' % (self._prefix, self._counter)
        i2h = mx.sym.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%si2h' % name)
        h2h = mx.sym.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%sh2h' % name)
        gates = i2h + h2h
        in_gate, forget_gate, in_transform, out_gate = mx.sym.SliceChannel(
            gates, num_outputs=4, name="%sslice" % name)

        in_gate = self._norm_layers[0].normalize(in_gate)
        forget_gate = self._norm_layers[1].normalize(forget_gate)
        in_transform = self._norm_layers[2].normalize(in_transform)
        out_gate = self._norm_layers[3].normalize(out_gate)

        in_gate = mx.sym.Activation(in_gate, act_type="sigmoid",
                                    name='%si' % name)
        forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid",
                                        name='%sf' % name)
        in_transform = mx.sym.Activation(in_transform, act_type="tanh",
                                         name='%sc' % name)
        out_gate = mx.sym.Activation(out_gate, act_type="sigmoid",
                                     name='%so' % name)
        next_c = mx.sym._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate' % name)
        next_c = self._norm_layers[4].normalize(next_c)
        next_h = mx.sym._internal._mul(out_gate, mx.sym.Activation(next_c, act_type="tanh"),
                                       name='%sout' % name)
        return next_h, [next_h, next_c]

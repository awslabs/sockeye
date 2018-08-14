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

# List is needed for mypy, but not used in the code, only in special comments
from typing import Optional, List, Iterable, Tuple  # NOQA pylint: disable=unused-import

import mxnet as mx

import numpy as np
from sockeye.config import Config
from . import layers
from . import constants as C
from . import utils


class RNNConfig(Config):
    """
    RNN configuration.

    :param cell_type: RNN cell type.
    :param num_hidden: Number of RNN hidden units.
    :param num_layers: Number of RNN layers.
    :param dropout_inputs: Dropout probability on RNN inputs (Gal, 2015).
    :param dropout_states: Dropout probability on RNN states (Gal, 2015).
    :param dropout_recurrent: Dropout probability on cell update (Semeniuta, 2016).
    :param residual: Whether to add residual connections between multi-layered RNNs.
    :param first_residual_layer: First layer with a residual connection (1-based indexes).
           Default is to start at the second layer.
    :param forget_bias: Initial value of forget biases.
    :param lhuc: Apply LHUC (Vilar 2018) to the hidden units of the RNN.
    :param dtype: Data type.
    """

    def __init__(self,
                 cell_type: str,
                 num_hidden: int,
                 num_layers: int,
                 dropout_inputs: float,
                 dropout_states: float,
                 dropout_recurrent: float = 0,
                 residual: bool = False,
                 first_residual_layer: int = 2,
                 forget_bias: float = 0.0,
                 lhuc: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.cell_type = cell_type
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_inputs = dropout_inputs
        self.dropout_states = dropout_states
        self.dropout_recurrent = dropout_recurrent
        self.residual = residual
        self.first_residual_layer = first_residual_layer
        self.forget_bias = forget_bias
        self.lhuc = lhuc
        self.dtype = dtype


class SequentialRNNCellParallelInput(mx.rnn.SequentialRNNCell):
    """
    A SequentialRNNCell, where an additional "parallel" input can be given at
    call time and it will be added to the input of each layer
    """

    def __call__(self, inputs, parallel_inputs, states):
        # Adapted copy of mx.rnn.SequentialRNNCell.__call__()
        self._counter += 1
        next_states = []
        pos = 0
        for cell in self._cells:
            assert not isinstance(cell, mx.rnn.BidirectionalCell)
            length = len(cell.state_info)
            state = states[pos:pos + length]
            pos += length
            inputs, state = cell(inputs, parallel_inputs, state)
            next_states.append(state)
        return inputs, sum(next_states, [])


class ParallelInputCell(mx.rnn.ModifierCell):
    """
    A modifier cell that accepts two input vectors and concatenates them before
    calling the original cell. Typically it is used for concatenating the
    normal and the parallel input in a stacked rnn.
    """

    def __call__(self, inputs, parallel_inputs, states):
        concat_inputs = mx.sym.concat(inputs, parallel_inputs)
        output, states = self.base_cell(concat_inputs, states)
        return output, states


class ResidualCellParallelInput(mx.rnn.ResidualCell):
    """
    A ResidualCell, where an additional "parallel" input can be given at call
    time and it will be added to the input of each layer, but not considered
    for the residual connection itself.
    """

    def __call__(self, inputs, parallel_inputs, states):
        concat_inputs = mx.sym.concat(inputs, parallel_inputs)
        output, states = self.base_cell(concat_inputs, states)
        output = mx.symbol.elemwise_add(output, inputs, name="%s_plus_residual" % output.name)
        return output, states


def get_rnn_cell(
        cell_type: str,
        num_hidden: int,
        dropout_inputs: float,
        dropout_states: float,
        dropout_recurrent: float = 0,
        forget_bias: float = 0.0,
        lhuc: bool = False,
        dtype: str = C.DTYPE_FP32,
        prefix: str = ''):
    """
    Create a single rnn cell.
    :param cell_type: RNN cell type.
    :param num_hidden: Number of RNN hidden units.
    :param dropout_inputs: Dropout probability on RNN inputs (Gal, 2015).
    :param dropout_states: Dropout probability on RNN states (Gal, 2015).
    :param dropout_recurrent: Dropout probability on cell update (Semeniuta, 2016).
    :param forget_bias: Initial value of forget biases.
    :param lhuc: Apply LHUC (Vilar 2018) to the hidden units of the RNN.
    :param dtype: Data type.
    :param prefix: Variable name prefix.
    """
    if cell_type == C.LSTM_TYPE:
        if dropout_recurrent > 0.0:
            cell = RecurrentDropoutLSTMCell(num_hidden=num_hidden,
                                            prefix=prefix,
                                            forget_bias=forget_bias,
                                            dropout=dropout_recurrent)
        else:
            cell = mx.rnn.LSTMCell(num_hidden=num_hidden, prefix=prefix, forget_bias=forget_bias)
    elif cell_type == C.LNLSTM_TYPE:
        cell = LayerNormLSTMCell(num_hidden=num_hidden, prefix=prefix, forget_bias=forget_bias)
    elif cell_type == C.LNGLSTM_TYPE:
        cell = LayerNormPerGateLSTMCell(num_hidden=num_hidden, prefix=prefix,
                                        forget_bias=forget_bias)
    elif cell_type == C.GRU_TYPE:
        cell = mx.rnn.GRUCell(num_hidden=num_hidden, prefix=prefix)
    elif cell_type == C.LNGRU_TYPE:
        cell = LayerNormGRUCell(num_hidden=num_hidden, prefix=prefix)
    elif cell_type == C.LNGGRU_TYPE:
        cell = LayerNormPerGateGRUCell(num_hidden=num_hidden, prefix=prefix)
    elif cell_type == "simple":
        cell = JanetCell(num_hidden=num_hidden, prefix=prefix,
                         forget_bias=forget_bias)
    else:
        raise NotImplementedError("Unknown cell type %s" % cell_type)

    if dropout_inputs > 0 or dropout_states > 0:
        cell = VariationalDropoutCell(cell,
                                      dropout_inputs=dropout_inputs,
                                      dropout_states=dropout_states)
    if lhuc:
        cell = LHUCCell(cell, num_hidden, dtype)

    return cell


def get_stacked_rnn(config: RNNConfig,
                    prefix: str,
                    parallel_inputs: bool = False,
                    layers: Optional[Iterable[int]] = None) -> mx.rnn.SequentialRNNCell:
    """
    Returns (stacked) RNN cell given parameters.

    :param config: rnn configuration.
    :param prefix: Symbol prefix for RNN.
    :param parallel_inputs: Support parallel inputs for the stacked RNN cells.
    :param layers: Specify which layers to create as a list of layer indexes.

    :return: RNN cell.
    """

    rnn = mx.rnn.SequentialRNNCell() if not parallel_inputs else SequentialRNNCellParallelInput()
    if not layers:
        layers = range(config.num_layers)
    for layer_idx in layers:
        # fhieber: the 'l' in the prefix does NOT stand for 'layer' but for the direction 'l' as in mx.rnn.rnn_cell::517
        # this ensures parameter name compatibility of training w/ FusedRNN and decoding with 'unfused' RNN.
        cell_prefix = "%sl%d_" % (prefix, layer_idx)
        cell = get_rnn_cell(cell_type=config.cell_type, num_hidden=config.num_hidden,
                            dropout_inputs=config.dropout_inputs, dropout_states=config.dropout_states,
                            dropout_recurrent=config.dropout_recurrent, forget_bias=config.forget_bias,
                            lhuc=config.lhuc, dtype=config.dtype, prefix=cell_prefix)

        # layer_idx is 0 based, whereas first_residual_layer is 1-based
        if config.residual and layer_idx + 1 >= config.first_residual_layer:
            cell = mx.rnn.ResidualCell(cell) if not parallel_inputs else ResidualCellParallelInput(cell)
        elif parallel_inputs:
            cell = ParallelInputCell(cell)

        rnn.add(cell)

    return rnn


class LayerNormLSTMCell(mx.rnn.LSTMCell):
    """
    Long-Short Term Memory (LSTM) network cell with layer normalization across gates.
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
                 prefix: str = 'lnlstm_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 forget_bias: float = 1.0,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormLSTMCell, self).__init__(num_hidden, prefix, params, forget_bias)
        self._iN = layers.LayerNormalization(num_hidden=num_hidden, prefix="%si2h" % self._prefix,
                                             scale=self.params.get('i2h_scale', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_scale)),
                                             shift=self.params.get('i2h_shift', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_shift)))
        self._hN = layers.LayerNormalization(num_hidden=num_hidden, prefix="%sh2h" % self._prefix,
                                             scale=self.params.get('h2h_scale', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_scale)),
                                             shift=self.params.get('h2h_shift', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_shift)))
        self._cN = layers.LayerNormalization(num_hidden=num_hidden, prefix="%sc" % self._prefix,
                                             scale=self.params.get('c_scale', shape=(num_hidden,), init=mx.init.Constant(value=norm_scale)),
                                             shift=self.params.get('c_shift', shape=(num_hidden,), init=mx.init.Constant(value=norm_shift)))

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_' % (self._prefix, self._counter)
        i2h = mx.sym.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%si2h' % name)
        h2h = mx.sym.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%sh2h' % name)
        gates = self._iN(data=i2h) + self._hN(data=h2h + mx.sym.zeros_like(i2h))
        # pylint: disable=unbalanced-tuple-unpacking
        in_gate, forget_gate, in_transform, out_gate = mx.sym.split(gates,
                                                                    num_outputs=4,
                                                                    axis=1,
                                                                    name="%sslice" % name)
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
        next_h = mx.sym._internal._mul(out_gate,
                                       mx.sym.Activation(self._cN(data=next_c), act_type="tanh"),
                                       name='%sout' % name)
        return next_h, [next_h, next_c]


class LayerNormPerGateLSTMCell(mx.rnn.LSTMCell):
    """
    Long-Short Term Memory (LSTM) network cell with layer normalization per gate.
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
                 prefix: str = 'lnglstm_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 forget_bias: float = 1.0,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormPerGateLSTMCell, self).__init__(num_hidden, prefix, params, forget_bias)
        self._norm_layers = list()  # type: List[layers.LayerNormalization]
        for name in ['i', 'f', 'c', 'o', 's']:
            scale = self.params.get('%s_shift' % name,
                                    init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name,
                                    init=mx.init.Constant(value=norm_scale if name != "f" else forget_bias))
            self._norm_layers.append(
                layers.LayerNormalization(prefix="%s%s" % (self._prefix, name), num_hidden=num_hidden,
                                          scale=scale, shift=shift))

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
        # pylint: disable=unbalanced-tuple-unpacking
        in_gate, forget_gate, in_transform, out_gate = mx.sym.split(
            gates, num_outputs=4, name="%sslice" % name)

        in_gate = self._norm_layers[0](data=in_gate)
        forget_gate = self._norm_layers[1](data=forget_gate)
        in_transform = self._norm_layers[2](data=in_transform)
        out_gate = self._norm_layers[3](data=out_gate)

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
        next_h = mx.sym._internal._mul(out_gate,
                                       mx.sym.Activation(self._norm_layers[4].__call__(next_c), act_type="tanh"),
                                       name='%sout' % name)
        return next_h, [next_h, next_c]


class LHUCCell(mx.rnn.ModifierCell):
    """
    Adds a LHUC operation to the output of the cell.
    """
    def __init__(self, base_cell, num_hidden, dtype) -> None:
        super().__init__(base_cell)
        self.num_hidden = num_hidden
        self.lhuc_params = self.params.get(C.LHUC_NAME, shape=(num_hidden,), dtype=dtype, init=mx.init.Uniform(0.1))
        self.lhuc = layers.LHUC(num_hidden, self.lhuc_params)

    def __call__(self, inputs, states):
        output, states = self.base_cell(inputs, states)
        output = self.lhuc(inputs=output)
        return output, states


class RecurrentDropoutLSTMCell(mx.rnn.LSTMCell):
    """
    LSTMCell with recurrent dropout without memory loss as in:
    http://aclanthology.coli.uni-saarland.de/pdf/C/C16/C16-1165.pdf
    """

    def __init__(self, num_hidden, prefix='lstm_', params=None, forget_bias=1.0, dropout: float = 0.0) -> None:
        super().__init__(num_hidden, prefix, params, forget_bias)
        utils.check_condition(dropout > 0.0, "RecurrentDropoutLSTMCell shoud have dropout > 0.0")
        self.dropout = dropout

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
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                          name="%sslice" % name)
        in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid",
                                    name='%si' % name)
        forget_gate = mx.sym.Activation(slice_gates[1], act_type="sigmoid",
                                        name='%sf' % name)
        in_transform = mx.sym.Activation(slice_gates[2], act_type="tanh",
                                         name='%sc' % name)
        if self.dropout > 0.0:
            in_transform = mx.sym.Dropout(in_transform, p=self.dropout, name='%sc_dropout' % name)
        out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid",
                                     name='%so' % name)
        next_c = mx.sym._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate' % name)
        next_h = mx.sym._internal._mul(out_gate, mx.sym.Activation(next_c, act_type="tanh"),
                                       name='%sout' % name)

        return next_h, [next_h, next_c]


class LayerNormGRUCell(mx.rnn.GRUCell):
    """
    Gated Recurrent Unit (GRU) network cell with layer normalization across gates.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lngru_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormGRUCell, self).__init__(num_hidden, prefix, params)
        self._iN = layers.LayerNormalization(
            prefix="%si2h" % self._prefix,
            num_hidden=num_hidden,
            scale=self.params.get('i2h_scale', init=mx.init.Constant(value=norm_scale)),
            shift=self.params.get('i2h_shift', init=mx.init.Constant(value=norm_shift)))
        self._hN = layers.LayerNormalization(
            prefix="%sh2h" % self._prefix,
            num_hidden=num_hidden,
            scale=self.params.get('h2h_scale', init=mx.init.Constant(value=norm_scale)),
            shift=self.params.get('h2h_shift', init=mx.init.Constant(value=norm_shift)))

    def __call__(self, inputs, states):
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = mx.sym.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_i2h" % name)
        h2h = mx.sym.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_h2h" % name)

        i2h = self._iN(data=i2h)
        h2h = self._hN(data=h2h)

        # pylint: disable=unbalanced-tuple-unpacking
        i2h_r, i2h_z, i2h = mx.sym.split(i2h, num_outputs=3, name="%s_i2h_slice" % name)
        h2h_r, h2h_z, h2h = mx.sym.split(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = mx.sym.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                       name="%s_r_act" % name)
        update_gate = mx.sym.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                        name="%s_z_act" % name)

        next_h_tmp = mx.sym.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                       name="%s_h_act" % name)

        next_h = mx.sym._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]


class LayerNormPerGateGRUCell(mx.rnn.GRUCell):
    """
    Gated Recurrent Unit (GRU) network cell with layer normalization per gate.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lnggru_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormPerGateGRUCell, self).__init__(num_hidden, prefix, params)
        self._norm_layers = list()  # type: List[layers.LayerNormalization]
        for name in ['r', 'z', 'o']:
            scale = self.params.get('%s_shift' % name, init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name, init=mx.init.Constant(value=norm_scale))
            self._norm_layers.append(layers.LayerNormalization(
                prefix="%s%s" % (self._prefix, name), num_hidden=num_hidden, scale=scale, shift=shift))

    def __call__(self, inputs, states):
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = mx.sym.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_i2h" % name)
        h2h = mx.sym.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_h2h" % name)

        # pylint: disable=unbalanced-tuple-unpacking
        i2h_r, i2h_z, i2h = mx.sym.split(i2h, num_outputs=3, name="%s_i2h_slice" % name)
        h2h_r, h2h_z, h2h = mx.sym.split(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = mx.sym.Activation(self._norm_layers[0](data=i2h_r + h2h_r),
                                       act_type="sigmoid", name="%s_r_act" % name)
        update_gate = mx.sym.Activation(self._norm_layers[1](data=i2h_z + h2h_z),
                                        act_type="sigmoid", name="%s_z_act" % name)

        next_h_tmp = mx.sym.Activation(self._norm_layers[2](data=i2h + reset_gate * h2h),
                                       act_type="tanh", name="%s_h_act" % name)

        next_h = mx.sym._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]


class VariationalDropoutCell(mx.rnn.ModifierCell):
    """
    Apply Bayesian Dropout on input and states separately. The dropout mask does not change when applied sequentially.

    :param base_cell: Base cell to be modified.
    :param dropout_inputs: Dropout probability for inputs.
    :param dropout_states: Dropout probability for state inputs.
    """

    def __init__(self,
                 base_cell: mx.rnn.BaseRNNCell,
                 dropout_inputs: float,
                 dropout_states: float) -> None:
        super().__init__(base_cell)
        self.dropout_inputs = dropout_inputs
        self.dropout_states = dropout_states
        self.mask_inputs = None
        self.mask_states = None

    def __call__(self, inputs, states):
        if self.dropout_inputs > 0:
            if self.mask_inputs is None:
                self.mask_inputs = mx.sym.Dropout(data=mx.sym.ones_like(inputs), p=self.dropout_inputs)
            inputs = inputs * self.mask_inputs

        if self.dropout_states > 0:
            if self.mask_states is None:
                self.mask_states = mx.sym.Dropout(data=mx.sym.ones_like(states[0]), p=self.dropout_states)
            states[0] = states[0] * self.mask_states

        output, states = self.base_cell(inputs, states)

        return output, states

    def reset(self):
        super(VariationalDropoutCell, self).reset()
        self.mask_inputs = None
        self.mask_states = None


class JanetCell(mx.rnn.BaseRNNCell):
    """Janet cell, as described in:
    https://arxiv.org/pdf/1804.04849.pdf

    Parameters
    ----------
    num_hidden : int
        Number of units in output symbol.
    prefix : str, default 'lstm_'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    forget_bias : bias added to forget gate, default 1.0.
        Jozefowicz et al. 2015 recommends setting this to 1.0
    """
    def __init__(self, num_hidden, prefix='lstm_', params=None, forget_bias=1.0):
        super().__init__(prefix=prefix, params=params)

        self._num_hidden = num_hidden
        self._iW = self.params.get('i2h_weight')
        self._hW = self.params.get('h2h_weight')
        # we add the forget_bias to i2h_bias, this adds the bias to the forget gate activation
        self._iB = self.params.get('i2h_bias', init=mx.init.LSTMBias(forget_bias=forget_bias))
        self._hB = self.params.get('h2h_bias')

    @property
    def state_info(self):
        return [{'shape': (0, self._num_hidden), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ['_f', '_o']

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_'%(self._prefix, self._counter)
        i2h = mx.sym.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden*2,
                                    name='%si2h'%name)
        h2h = mx.sym.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden*2,
                                    name='%sh2h'%name)
        gates = i2h + h2h
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=2,
                                          name="%sslice"%name)
        forget_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid",
                                        name='%sf'%name)
        in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")

        next_h = mx.sym._internal._plus(forget_gate * states[0], (1. - forget_gate) * in_transform,
                                        name='%sstate' % name)

        return next_h, [next_h]


class RecurrentLayerRNNConfig(Config):
    """
    :param num_hidden: Number of RNN hidden units.
    :param dropout_recurrent: Dropout probability on cell update (Semeniuta, 2016).
    :param dropout_inputs: Dropout probability on RNN inputs (Gal, 2015).
    :param dropout_states: Dropout probability on RNN states (Gal, 2015).
    :param cell_type: RNN cell type.
    :param forget_bias: Initial value of forget biases.
    :param lhuc: Apply LHUC (Vilar 2018) to the hidden units of the RNN.
    :param dtype: Data type.
    """

    def __init__(self,
                 num_hidden: int,
                 dropout_recurrent: float = 0.0,
                 dropout_inputs: float = 0.0,
                 dropout_states: float = 0.0,
                 norm_states: bool = True,
                 norm_first_step: bool = True,
                 cell_type: str = C.LSTM_TYPE,
                 forget_bias: float = 0.0,
                 lhuc: bool = False,
                 dtype: str = C.DTYPE_FP32):
        super().__init__()
        self.num_hidden = num_hidden
        # recurrent/inputs/states is for "old" cells and just "dropout" for "new states"
        self.dropout_recurrent = dropout_recurrent
        self.dropout_inputs = dropout_inputs
        self.dropout_states = dropout_states
        self.norm_states = norm_states
        self.norm_first_step = norm_first_step
        self.cell_type = cell_type
        self.forget_bias = forget_bias
        self.lhuc = lhuc
        self.dtype = dtype

    def create_rnn_cell(self, prefix: str):
        cell = get_rnn_cell(cell_type=self.cell_type, num_hidden=self.num_hidden,
                            dropout_inputs=self.dropout_inputs, dropout_states=self.dropout_states,
                            dropout_recurrent=self.dropout_recurrent, forget_bias=self.forget_bias,
                            lhuc=self.lhuc, dtype=self.dtype, prefix=prefix)

        if self.dropout_inputs > 0 or self.dropout_states > 0:
            cell = VariationalDropoutCell(cell,
                                          dropout_inputs=self.dropout_inputs,
                                          dropout_states=self.dropout_states)
        return cell


class RecurrentEncoderLayer(layers.EncoderLayer):

    def __init__(self,
                 rnn_config: RecurrentLayerRNNConfig,
                 prefix: str = ""):
        self.rnn_cell = rnn_config.create_rnn_cell(prefix)
        self.num_hidden = rnn_config.num_hidden

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        outputs, _ = self.rnn_cell.unroll(length=source_encoded_max_length,
                                          inputs=source_encoded,
                                          merge_outputs=True,
                                          layout=C.BATCH_MAJOR)
        return outputs, source_encoded_lengths, source_encoded_max_length

    def get_num_hidden(self) -> int:
        return self.num_hidden


class RecurrentDecoderLayer(layers.DecoderLayer):

    def __init__(self,
                 rnn_config: RecurrentLayerRNNConfig,
                 prefix: str = ""):
        self.prefix = prefix
        self.rnn_cell = rnn_config.create_rnn_cell(prefix)
        self.num_hidden = rnn_config.num_hidden

    def decode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        outputs, _ = self.rnn_cell.unroll(length=target_encoded_max_length,
                                          inputs=target_encoded,
                                          merge_outputs=True,
                                          layout=C.BATCH_MAJOR)

        return outputs

    def decode_step(self, step: int,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int,
                    target: mx.sym.Symbol,
                    states: List[mx.sym.Symbol],
                    att_dict) -> Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        return self.rnn_cell(target, states)

    def reset(self):
        # TODO remove this once mxnet.rnn.ModifierCell.reset() invokes reset() of base_cell
        cell = self.rnn_cell
        if isinstance(cell, mx.rnn.ModifierCell):
            cell.base_cell.reset()
        cell.reset()

    def get_num_hidden(self) -> int:
        return self.num_hidden

    def num_states(self, step: int) -> int:
        return len(self.rnn_cell.state_info)

    def state_variables(self, step: int) -> List[mx.sym.Symbol]:
        return [mx.sym.Variable("%rnn_state_%d" % (self.prefix, i))
                for i, state_info in enumerate(self.rnn_cell.state_info)]

    def init_states(self,
                    batch_size,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> List[mx.sym.Symbol]:
        return [mx.sym.zeros(shape=(batch_size, num_hidden)) for (_, num_hidden) in self.rnn_cell.state_shape]

    def state_shapes(self,
                     batch_size: int,
                     target_max_length: int,
                     source_encoded_max_length: int,
                     source_encoded_num_hidden: int) -> List[mx.io.DataDesc]:
        return [mx.io.DataDesc("%rnn_state_%d" % (self.prefix, i),
                               (batch_size, num_hidden),
                               layout=C.BATCH_MAJOR) for i, (_, num_hidden) in enumerate(self.rnn_cell.state_shape)]


class RecurrentLayerConfig(layers.LayerConfig):
    def __init__(self,
                 num_hidden: int,
                 cell_type: str = C.LSTM_TYPE,
                 dropout_inputs: float = 0.0,
                 dropout_states: float = 0.0,
                 dropout_recurrent: float = 0.0,
                 norm_states: bool = True,
                 norm_first_step: bool = True,
                 forget_bias: float = 0.0):
        super().__init__()
        self.rnn_config = RecurrentLayerRNNConfig(num_hidden=num_hidden,
                                                  dropout_recurrent=dropout_recurrent,
                                                  dropout_inputs=dropout_inputs,
                                                  dropout_states=dropout_states,
                                                  norm_states=norm_states,
                                                  norm_first_step=norm_first_step,
                                                  cell_type=cell_type,
                                                  forget_bias=forget_bias)

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> layers.EncoderLayer:
        return RecurrentEncoderLayer(rnn_config=self.rnn_config, prefix=prefix + "rnn_")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> layers.DecoderLayer:
        return RecurrentDecoderLayer(rnn_config=self.rnn_config, prefix=prefix + "rnn_")


class BidirectionalRecurrentEncoderLayer(layers.EncoderLayer):
    def __init__(self,
                 rnn_config: RecurrentLayerRNNConfig,
                 prefix: str = ""):
        self.prefix = prefix
        utils.check_condition(rnn_config.num_hidden % 2 == 0,
                              "num_hidden must be a multiple of 2 for BiDirectionalRNNEncoders.")
        self.rnn_config = rnn_config
        self.internal_rnn_config = rnn_config.copy(num_hidden=rnn_config.num_hidden // 2)

        self.forward_rnn_cell = self.internal_rnn_config.create_rnn_cell(prefix + C.FORWARD_PREFIX)
        self.backward_rnn_cell = self.internal_rnn_config.create_rnn_cell(prefix + C.REVERSE_PREFIX)

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        # Batch major to time major for sequence reverse
        # (batch_size, seq_len, num_hidden) -> (seq_len, batch_size, num_hidden)
        data = mx.sym.transpose(data=source_encoded, axes=(1, 0, 2))

        # (seq_len, batch_size, num_embed)
        data_reverse = mx.sym.SequenceReverse(data=data,
                                              sequence_length=source_encoded_lengths,
                                              use_sequence_length=True)
        # (seq_length, batch, cell_num_hidden)
        hidden_forward, _ = self.forward_rnn_cell.unroll(length=source_encoded_max_length,
                                                         inputs=data,
                                                         merge_outputs=True,
                                                         layout=C.TIME_MAJOR)
        # (seq_length, batch, cell_num_hidden)
        hidden_reverse, _ = self.backward_rnn_cell.unroll(length=source_encoded_max_length,
                                                          inputs=data_reverse,
                                                          merge_outputs=True,
                                                          layout=C.TIME_MAJOR)

        # (seq_length, batch, cell_num_hidden)
        hidden_reverse = mx.sym.SequenceReverse(data=hidden_reverse,
                                                sequence_length=source_encoded_lengths,
                                                use_sequence_length=True)

        # (seq_length, batch, 2 * cell_num_hidden)
        hidden_concat = mx.sym.concat(hidden_forward, hidden_reverse, dim=2, name="%s_rnn" % self.prefix)

        # Time major to batch major for sequence reverse
        # (seq_len, batch_size, num_hidden) -> (batch_size, seq_len, num_hidden)
        hidden_concat = mx.sym.transpose(data=hidden_concat, axes=(1, 0, 2))
        return hidden_concat, source_encoded_lengths, source_encoded_max_length

    def get_num_hidden(self) -> int:
        return self.rnn_config.num_hidden


class BidirectionalRecurrentLayerConfig(layers.LayerConfig):
    def __init__(self,
                 num_hidden: int,
                 cell_type: str = C.LSTM_TYPE,
                 dropout_inputs: float = 0.0,
                 dropout_states: float = 0.0,
                 dropout_recurrent: float = 0.0,
                 norm_states: bool = True,
                 forget_bias: float = 0.0):
        super().__init__()
        self.rnn_config = RecurrentLayerRNNConfig(num_hidden=num_hidden,
                                                  dropout_recurrent=dropout_recurrent,
                                                  dropout_inputs=dropout_inputs,
                                                  dropout_states=dropout_states,
                                                  norm_states=norm_states,
                                                  cell_type=cell_type,
                                                  forget_bias=forget_bias)

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> layers.EncoderLayer:
        return BidirectionalRecurrentEncoderLayer(rnn_config=self.rnn_config, prefix=prefix + "birnn_")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> layers.DecoderLayer:
        raise ValueError("Bi-directional RNN can only be used on the encoder side.")


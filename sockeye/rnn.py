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

# List is needed for mypy, but not used in the code, only in special comments
from typing import Optional, List, Iterable  # NOQA pylint: disable=unused-import

import mxnet as mx

from sockeye.config import Config
from sockeye.layers import LayerNormalization, LHUC
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


def get_stacked_rnn(config: RNNConfig, prefix: str,
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
        if config.cell_type == C.LSTM_TYPE:
            if config.dropout_recurrent > 0.0:
                cell = RecurrentDropoutLSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix,
                                                forget_bias=config.forget_bias, dropout=config.dropout_recurrent)
            else:
                cell = mx.rnn.LSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix, forget_bias=config.forget_bias)
        elif config.cell_type == C.LNLSTM_TYPE:
            cell = LayerNormLSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix, forget_bias=config.forget_bias)
        elif config.cell_type == C.LNGLSTM_TYPE:
            cell = LayerNormPerGateLSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix,
                                            forget_bias=config.forget_bias)
        elif config.cell_type == C.GRU_TYPE:
            cell = mx.rnn.GRUCell(num_hidden=config.num_hidden, prefix=cell_prefix)
        elif config.cell_type == C.LNGRU_TYPE:
            cell = LayerNormGRUCell(num_hidden=config.num_hidden, prefix=cell_prefix)
        elif config.cell_type == C.LNGGRU_TYPE:
            cell = LayerNormPerGateGRUCell(num_hidden=config.num_hidden, prefix=cell_prefix)
        else:
            raise NotImplementedError()

        if config.dropout_inputs > 0 or config.dropout_states > 0:
            cell = VariationalDropoutCell(cell,
                                          dropout_inputs=config.dropout_inputs,
                                          dropout_states=config.dropout_states)

        if config.lhuc:
            cell = LHUCCell(cell, config.num_hidden, config.dtype)

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
        self._iN = LayerNormalization(prefix="%si2h" % self._prefix,
                                      scale=self.params.get('i2h_scale', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('i2h_shift', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_shift)))
        self._hN = LayerNormalization(prefix="%sh2h" % self._prefix,
                                      scale=self.params.get('h2h_scale', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('h2h_shift', shape=(num_hidden * 4,), init=mx.init.Constant(value=norm_shift)))
        self._cN = LayerNormalization(prefix="%sc" % self._prefix,
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
        self._norm_layers = list()  # type: List[LayerNormalization]
        for name in ['i', 'f', 'c', 'o', 's']:
            scale = self.params.get('%s_shift' % name,
                                    init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name,
                                    init=mx.init.Constant(value=norm_scale if name != "f" else forget_bias))
            self._norm_layers.append(
                LayerNormalization(prefix="%s%s" % (self._prefix, name), scale=scale, shift=shift))

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
        self.lhuc = LHUC(num_hidden, self.lhuc_params)

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
        self._iN = LayerNormalization(prefix="%si2h" % self._prefix,
                                      scale=self.params.get('i2h_scale', init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('i2h_shift', init=mx.init.Constant(value=norm_shift)))
        self._hN = LayerNormalization(prefix="%sh2h" % self._prefix,
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
        self._norm_layers = list()  # type: List[LayerNormalization]
        for name in ['r', 'z', 'o']:
            scale = self.params.get('%s_shift' % name, init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name, init=mx.init.Constant(value=norm_scale))
            self._norm_layers.append(LayerNormalization(prefix="%s%s" % (self._prefix, name), scale=scale, shift=shift))

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

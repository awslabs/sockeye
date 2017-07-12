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
import pytest

from sockeye import constants as C
from sockeye import rnn

cell_test_cases = [
    (rnn.LayerNormLSTMCell(100, prefix='rnn_', forget_bias=1.0),
     sorted(['rnn_c_scale', 'rnn_c_shift',
             'rnn_h2h_bias', 'rnn_h2h_scale', 'rnn_h2h_shift', 'rnn_h2h_weight',
             'rnn_i2h_bias', 'rnn_i2h_scale', 'rnn_i2h_shift', 'rnn_i2h_weight'])),
    (rnn.LayerNormPerGateLSTMCell(100, prefix='rnn_', forget_bias=1.0),
     sorted(['rnn_c_scale', 'rnn_c_shift',
             'rnn_f_scale', 'rnn_f_shift',
             'rnn_h2h_bias', 'rnn_h2h_weight',
             'rnn_i2h_bias', 'rnn_i2h_weight',
             'rnn_i_scale', 'rnn_i_shift',
             'rnn_o_scale', 'rnn_o_shift',
             'rnn_s_scale', 'rnn_s_shift'])),
    (rnn.LayerNormGRUCell(100, prefix='rnn_'),
     sorted(['rnn_h2h_bias', 'rnn_h2h_scale', 'rnn_h2h_shift', 'rnn_h2h_weight',
             'rnn_i2h_bias', 'rnn_i2h_scale', 'rnn_i2h_shift', 'rnn_i2h_weight'])),
    (rnn.LayerNormPerGateGRUCell(100, prefix='rnn_'),
     sorted(['rnn_h2h_bias', 'rnn_h2h_weight',
             'rnn_i2h_bias', 'rnn_i2h_weight',
             'rnn_o_scale', 'rnn_o_shift',
             'rnn_r_scale', 'rnn_r_shift',
             'rnn_z_scale', 'rnn_z_shift']))
]


@pytest.mark.parametrize("cell, expected_param_keys", cell_test_cases)
def test_ln_cell(cell, expected_param_keys):
    inputs = [mx.sym.Variable('rnn_t%d_data' % i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    print(sorted(cell.params._params.keys()))
    assert sorted(cell.params._params.keys()) == expected_param_keys
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10, 50), rnn_t1_data=(10, 50), rnn_t2_data=(10, 50))
    assert outs == [(10, 100), (10, 100), (10, 100)]


get_rnn_test_cases = [
    (C.LSTM_TYPE, 100, mx.rnn.LSTMCell),
    (C.LNLSTM_TYPE, 100, rnn.LayerNormLSTMCell),
    (C.LNGLSTM_TYPE, 100, rnn.LayerNormPerGateLSTMCell),
    (C.GRU_TYPE, 100, mx.rnn.GRUCell),
    (C.LNGRU_TYPE, 100, rnn.LayerNormGRUCell),
    (C.LNGGRU_TYPE, 100, rnn.LayerNormPerGateGRUCell)]


@pytest.mark.parametrize("cell_type, num_hidden, expected_cell", get_rnn_test_cases)
def test_get_stacked_rnn(cell_type, num_hidden, expected_cell):
    layers = 1
    dropout = 0.5
    cell = rnn.get_stacked_rnn(cell_type, num_hidden, layers, dropout, prefix="")
    assert isinstance(cell, mx.rnn.SequentialRNNCell)
    assert len(cell._cells) == 2
    assert isinstance(cell._cells[0], expected_cell)
    assert cell._cells[0]._num_hidden == num_hidden
    assert isinstance(cell._cells[-1], mx.rnn.DropoutCell)

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
import numpy as np
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
    (rnn.RNNConfig(cell_type=C.LSTM_TYPE, num_hidden=100, num_layers=2, dropout_inputs=0.5, dropout_states=0.5,
                   residual=False, forget_bias=0.0), mx.rnn.LSTMCell),
    (rnn.RNNConfig(cell_type=C.LSTM_TYPE, num_hidden=100, num_layers=2, dropout_inputs=0.0, dropout_states=0.0,
                   dropout_recurrent=0.5, residual=False, forget_bias=0.0), rnn.RecurrentDropoutLSTMCell),
    (rnn.RNNConfig(cell_type=C.LNLSTM_TYPE, num_hidden=12, num_layers=2, dropout_inputs=0.5, dropout_states=0.5,
                   residual=False, forget_bias=1.0), rnn.LayerNormLSTMCell),
    (rnn.RNNConfig(cell_type=C.LNGLSTM_TYPE, num_hidden=55, num_layers=2, dropout_inputs=0.5, dropout_states=0.5,
                   residual=False, forget_bias=0.0), rnn.LayerNormPerGateLSTMCell),
    (rnn.RNNConfig(cell_type=C.GRU_TYPE, num_hidden=200, num_layers=2, dropout_inputs=0.9, dropout_states=0.9,
                   residual=False, forget_bias=0.0), mx.rnn.GRUCell),
    (rnn.RNNConfig(cell_type=C.LNGRU_TYPE, num_hidden=100, num_layers=2, dropout_inputs=0.0, dropout_states=0.5,
                   residual=False, forget_bias=0.0), rnn.LayerNormGRUCell),
    (rnn.RNNConfig(cell_type=C.LNGGRU_TYPE, num_hidden=2, num_layers=2, dropout_inputs=0.0, dropout_states=0.0,
                   residual=True, forget_bias=0.0), rnn.LayerNormPerGateGRUCell),
    (rnn.RNNConfig(cell_type=C.LSTM_TYPE, num_hidden=2, num_layers=3, dropout_inputs=0.0, dropout_states=0.0,
                   residual=True, forget_bias=0.0), mx.rnn.LSTMCell)]


@pytest.mark.parametrize("config, expected_cell", get_rnn_test_cases)
def test_get_stacked_rnn(config, expected_cell):
    cell = rnn.get_stacked_rnn(config, prefix=config.cell_type)
    assert isinstance(cell, mx.rnn.SequentialRNNCell)
    cell = cell._cells[-1]  # last cell
    if config.residual:
        assert isinstance(cell, mx.rnn.ResidualCell)
        cell = cell.base_cell
    if config.dropout_inputs > 0 or config.dropout_states > 0:
        assert isinstance(cell, rnn.VariationalDropoutCell)
        cell = cell.base_cell
    assert isinstance(cell, expected_cell)
    assert cell._num_hidden, config.num_hidden


def test_cell_parallel_input():
    num_hidden = 128
    batch_size = 256
    parallel_size = 64

    input_shape = (batch_size, num_hidden)
    states_shape = (batch_size, num_hidden)
    parallel_shape = (batch_size, parallel_size)

    inp = mx.sym.Variable("input")
    parallel_input = mx.sym.Variable("parallel")
    params = mx.rnn.RNNParams("params_")
    states = mx.sym.Variable("states")

    default_cell = mx.rnn.RNNCell(num_hidden, params=params)
    default_cell_output, _ = default_cell(mx.sym.concat(inp, parallel_input), states)

    inner_rnn_cell = mx.rnn.RNNCell(num_hidden, params=params)
    parallel_cell = rnn.ParallelInputCell(inner_rnn_cell)
    parallel_cell_output, _ = parallel_cell(inp, parallel_input, states)

    input_nd = mx.nd.random_uniform(shape=input_shape)
    states_nd = mx.nd.random_uniform(shape=states_shape)
    parallel_nd = mx.nd.random_uniform(shape=parallel_shape)
    arg_shapes, _, _ = default_cell_output.infer_shape(input=input_shape, states=states_shape, parallel=parallel_shape)
    params_with_shapes = filter(lambda a: a[0].startswith("params_"),
                                [x for x in zip(default_cell_output.list_arguments(), arg_shapes)]
                                )
    params_nd = {}
    for name, shape in params_with_shapes:
        params_nd[name] = mx.nd.random_uniform(shape=shape)

    out_default_residual = default_cell_output.eval(input=input_nd,
                                                    states=states_nd,
                                                    parallel=parallel_nd,
                                                    **params_nd)[0]
    out_parallel = parallel_cell_output.eval(input=input_nd,
                                             states=states_nd,
                                             parallel=parallel_nd,
                                             **params_nd)[0]

    assert np.isclose(out_default_residual.asnumpy(), out_parallel.asnumpy()).all()


def test_residual_cell_parallel_input():
    num_hidden = 128
    batch_size = 256
    parallel_size = 64

    input_shape = (batch_size, num_hidden)
    states_shape = (batch_size, num_hidden)
    parallel_shape = (batch_size, parallel_size)

    inp = mx.sym.Variable("input")
    parallel_input = mx.sym.Variable("parallel")
    params = mx.rnn.RNNParams("params_")
    states = mx.sym.Variable("states")

    default_cell = mx.rnn.RNNCell(num_hidden, params=params)
    default_cell_output, _ = default_cell(mx.sym.concat(inp, parallel_input), states)
    default_residual_output = mx.sym.elemwise_add(default_cell_output, inp)

    inner_rnn_cell = mx.rnn.RNNCell(num_hidden, params=params)
    parallel_cell = rnn.ResidualCellParallelInput(inner_rnn_cell)
    parallel_cell_output, _ = parallel_cell(inp, parallel_input, states)

    input_nd = mx.nd.random_uniform(shape=input_shape)
    states_nd = mx.nd.random_uniform(shape=states_shape)
    parallel_nd = mx.nd.random_uniform(shape=parallel_shape)
    arg_shapes, _, _ = default_residual_output.infer_shape(input=input_shape,
                                                           states=states_shape,
                                                           parallel=parallel_shape)
    params_with_shapes = filter(lambda a: a[0].startswith("params_"),
                                [x for x in zip(default_residual_output.list_arguments(), arg_shapes)]
                                )
    params_nd = {}
    for name, shape in params_with_shapes:
        params_nd[name] = mx.nd.random_uniform(shape=shape)

    out_default_residual = default_residual_output.eval(input=input_nd,
                                                        states=states_nd,
                                                        parallel=parallel_nd,
                                                        **params_nd)[0]
    out_parallel = parallel_cell_output.eval(input=input_nd,
                                             states=states_nd,
                                             parallel=parallel_nd,
                                             **params_nd)[0]

    assert np.isclose(out_default_residual.asnumpy(), out_parallel.asnumpy()).all()


def test_sequential_rnn_cell_parallel_input():
    num_hidden = 128
    batch_size = 256
    parallel_size = 64
    n_layers = 3

    input_shape = (batch_size, num_hidden)
    states_shape = (batch_size, num_hidden)
    parallel_shape = (batch_size, parallel_size)

    input = mx.sym.Variable("input")
    parallel_input = mx.sym.Variable("parallel")
    params = mx.rnn.RNNParams("params_")  # To simplify, we will share the parameters across all layers
    states = mx.sym.Variable("states")    # ...and also the previous states

    last_output = input
    for _ in range(n_layers):
        cell = mx.rnn.RNNCell(num_hidden, params=params)
        last_output, _ = cell(mx.sym.concat(last_output, parallel_input), states)
    manual_stacking_output = last_output

    sequential_cell = rnn.SequentialRNNCellParallelInput()
    for _ in range(n_layers):
        cell = mx.rnn.RNNCell(num_hidden, params=params)
        cell = rnn.ParallelInputCell(cell)
        sequential_cell.add(cell)
    sequential_output, _ = sequential_cell(input, parallel_input, [states]*n_layers)

    input_nd = mx.nd.random_uniform(shape=input_shape)
    states_nd = mx.nd.random_uniform(shape=states_shape)
    parallel_nd = mx.nd.random_uniform(shape=parallel_shape)
    arg_shapes, _, _ = manual_stacking_output.infer_shape(input=input_shape, states=states_shape, parallel=parallel_shape)
    params_with_shapes = filter(lambda a: a[0].startswith("params_"),
                                [x for x in zip(manual_stacking_output.list_arguments(), arg_shapes)]
                                )
    params_nd = {}
    for name, shape in params_with_shapes:
        params_nd[name] = mx.nd.random_uniform(shape=shape)

    out_manual = manual_stacking_output.eval(input=input_nd,
                                             states=states_nd,
                                             parallel=parallel_nd,
                                             **params_nd)[0]
    out_sequential = sequential_output.eval(input=input_nd,
                                            states=states_nd,
                                            parallel=parallel_nd,
                                            **params_nd)[0]

    assert np.isclose(out_manual.asnumpy(), out_sequential.asnumpy()).all()

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

import sockeye.rnn


def test_lnlstm():
    cell = sockeye.rnn.LnLSTMCell(100, prefix='rnn_', forget_bias=1.0)
    inputs = [mx.sym.Variable('rnn_t%d_data' % i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.params._params.keys()) == sorted(['rnn_c_scale',
                                                         'rnn_c_shift',
                                                         'rnn_f_scale',
                                                         'rnn_f_shift',
                                                         'rnn_h2h_bias',
                                                         'rnn_h2h_weight',
                                                         'rnn_i2h_bias',
                                                         'rnn_i2h_weight',
                                                         'rnn_i_scale',
                                                         'rnn_i_shift',
                                                         'rnn_o_scale',
                                                         'rnn_o_shift',
                                                         'rnn_s_scale',
                                                         'rnn_s_shift'])
    assert outputs.list_outputs() == ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10, 50), rnn_t1_data=(10, 50), rnn_t2_data=(10, 50))
    assert outs == [(10, 100), (10, 100), (10, 100)]

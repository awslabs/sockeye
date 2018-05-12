# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sockeye.constants as C
import sockeye.transformer


def test_auto_regressive_bias_op():
    bias = mx.nd.Custom(op_type='auto_regressive_bias', length=2)

    assert bias.dtype == np.float32

    expected = np.array([[0.0, -1.0e8], [0.0, 0.0]]).reshape((1, 2, 2))
    np.testing.assert_array_equal(bias.asnumpy(), expected)


def test_auto_regressive_bias_op_float16():
    bias = mx.nd.Custom(op_type='auto_regressive_bias', length=2, dtype=C.DTYPE_FP16)

    assert bias.dtype == np.float16

    expected = np.array([[0.0, -49152.0], [0.0, 0.0]]).reshape((1, 2, 2))
    np.testing.assert_array_equal(bias.asnumpy(), expected)


def test_auto_regressive_bias_sym():
    bias = mx.sym.Custom(op_type='auto_regressive_bias', length=2)

    arg_types, out_types, aux_types = bias.infer_type()
    assert out_types[0] == np.float32

    out = bias.eval()[0]

    assert out.dtype == np.float32

    expected = np.array([[0.0, -1.0e8], [0.0, 0.0]]).reshape((1, 2, 2))
    np.testing.assert_array_equal(out.asnumpy(), expected)


def test_auto_regressive_bias_sym_float16():
    bias = mx.sym.Custom(op_type='auto_regressive_bias', length=2, dtype=C.DTYPE_FP16)

    arg_types, out_types, aux_types = bias.infer_type()
    assert out_types[0] == np.float16

    out = bias.eval()[0]

    assert out.dtype == np.float16

    expected = np.array([[0.0, -49152.0], [0.0, 0.0]]).reshape((1, 2, 2))
    np.testing.assert_array_equal(out.asnumpy(), expected)

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
from unittest.mock import patch

import mxnet as mx
import numpy as np
import pytest
import sockeye.coverage
from test.common import gaussian_vector, integer_vector, uniform_vector

activation_types = ["tanh", "sigmoid", "relu", "softrelu"]


def setup_module():
    #  Store a reference to the original MXNet sequence mask function.
    _mask_with_one.original_sequence_mask = mx.sym.SequenceMask


@pytest.mark.parametrize("act_type", activation_types)
def test_activation_coverage(act_type):
    # Before running our test we patch MXNet's sequence mask function with a custom implementation.  Our custom function
    # will call the built in masking operation, but ensure the masking value is the number one.  This masking value
    # allows for clear test assertions.
    _patch_sequence_mask(lambda: _test_activation_coverage(act_type))


def test_gru_coverage():
    # Before running our test we patch MXNet's sequence mask function with a custom implementation.  Our custom function
    # will call the built in masking operation, but ensure the masking value is the number one.  This masking value
    # allows for clear test assertions.
    _patch_sequence_mask(lambda: _test_gru_coverage())


def _test_activation_coverage(act_type):
    config_coverage = sockeye.coverage.CoverageConfig(type=act_type, num_hidden=2, layer_normalization=False)
    encoder_num_hidden, decoder_num_hidden, source_seq_len, batch_size = 5, 5, 10, 4
    # source: (batch_size, source_seq_len, encoder_num_hidden)
    source = mx.sym.Variable("source")
    # source_length: (batch_size,)
    source_length = mx.sym.Variable("source_length")
    # prev_hidden: (batch_size, decoder_num_hidden)
    prev_hidden = mx.sym.Variable("prev_hidden")
    # prev_coverage: (batch_size, source_seq_len, coverage_num_hidden)
    prev_coverage = mx.sym.Variable("prev_coverage")
    # attention_scores: (batch_size, source_seq_len)
    attention_scores = mx.sym.Variable("attention_scores")
    source_shape = (batch_size, source_seq_len, encoder_num_hidden)
    source_length_shape = (batch_size,)
    prev_hidden_shape = (batch_size, decoder_num_hidden)
    attention_scores_shape = (batch_size, source_seq_len)
    prev_coverage_shape = (batch_size, source_seq_len, config_coverage.num_hidden)
    source_data = gaussian_vector(shape=source_shape)
    source_length_data = integer_vector(shape=source_length_shape, max_value=source_seq_len)
    prev_hidden_data = gaussian_vector(shape=prev_hidden_shape)
    prev_coverage_data = gaussian_vector(shape=prev_coverage_shape)
    attention_scores_data = uniform_vector(shape=attention_scores_shape)
    attention_scores_data = attention_scores_data / np.sum(attention_scores_data)

    coverage = sockeye.coverage.get_coverage(config_coverage)
    coverage_func = coverage.on(source, source_length, source_seq_len)
    updated_coverage = coverage_func(prev_hidden, attention_scores, prev_coverage)
    executor = updated_coverage.simple_bind(ctx=mx.cpu(),
                                            source=source_shape,
                                            source_length=source_length_shape,
                                            prev_hidden=prev_hidden_shape,
                                            prev_coverage=prev_coverage_shape,
                                            attention_scores=attention_scores_shape)
    executor.arg_dict["source"][:] = source_data
    executor.arg_dict["source_length"][:] = source_length_data
    executor.arg_dict["prev_hidden"][:] = prev_hidden_data
    executor.arg_dict["prev_coverage"][:] = prev_coverage_data
    executor.arg_dict["attention_scores"][:] = attention_scores_data
    result = executor.forward()
    # this is needed to modulate the 0 input. The output changes according to the activation type used.
    activation = mx.sym.Activation(name="activation", act_type=act_type)
    modulated = activation.eval(ctx=mx.cpu(), activation_data=mx.nd.zeros((1, 1)))[0].asnumpy()
    new_coverage = result[0].asnumpy()
    assert new_coverage.shape == prev_coverage_shape
    assert (np.sum(np.sum(new_coverage == modulated, axis=2) != 0, axis=1) == source_length_data).all()


def _test_gru_coverage():
    config_coverage = sockeye.coverage.CoverageConfig(type="gru", num_hidden=2, layer_normalization=False)
    encoder_num_hidden, decoder_num_hidden, source_seq_len, batch_size = 5, 5, 10, 4
    # source: (batch_size, source_seq_len, encoder_num_hidden)
    source = mx.sym.Variable("source")
    # source_length: (batch_size,)
    source_length = mx.sym.Variable("source_length")
    # prev_hidden: (batch_size, decoder_num_hidden)
    prev_hidden = mx.sym.Variable("prev_hidden")
    # prev_coverage: (batch_size, source_seq_len, coverage_num_hidden)
    prev_coverage = mx.sym.Variable("prev_coverage")
    # attention_scores: (batch_size, source_seq_len)
    attention_scores = mx.sym.Variable("attention_scores")
    source_shape = (batch_size, source_seq_len, encoder_num_hidden)
    source_length_shape = (batch_size,)
    prev_hidden_shape = (batch_size, decoder_num_hidden)
    attention_scores_shape = (batch_size, source_seq_len)
    prev_coverage_shape = (batch_size, source_seq_len, config_coverage.num_hidden)
    source_data = gaussian_vector(shape=source_shape)
    source_length_data = integer_vector(shape=source_length_shape, max_value=source_seq_len)
    prev_hidden_data = gaussian_vector(shape=prev_hidden_shape)
    prev_coverage_data = gaussian_vector(shape=prev_coverage_shape)
    attention_scores_data = uniform_vector(shape=attention_scores_shape)
    attention_scores_data = attention_scores_data / np.sum(attention_scores_data)
    coverage = sockeye.coverage.get_coverage(config_coverage)
    coverage_func = coverage.on(source, source_length, source_seq_len)
    updated_coverage = coverage_func(prev_hidden, attention_scores, prev_coverage)
    executor = updated_coverage.simple_bind(ctx=mx.cpu(),
                                            source=source_shape,
                                            source_length=source_length_shape,
                                            prev_hidden=prev_hidden_shape,
                                            prev_coverage=prev_coverage_shape,
                                            attention_scores=attention_scores_shape)
    executor.arg_dict["source"][:] = source_data
    executor.arg_dict["source_length"][:] = source_length_data
    executor.arg_dict["prev_hidden"][:] = prev_hidden_data
    executor.arg_dict["prev_coverage"][:] = prev_coverage_data
    executor.arg_dict["attention_scores"][:] = attention_scores_data
    result = executor.forward()
    new_coverage = result[0].asnumpy()
    assert new_coverage.shape == prev_coverage_shape
    assert (np.sum(np.sum(new_coverage != 1, axis=2) != 0, axis=1) == source_length_data).all()


def _mask_with_one(data, axis, use_sequence_length, sequence_length):
    return _mask_with_one.original_sequence_mask(data=data, axis=axis, use_sequence_length=use_sequence_length,
                                                 sequence_length=sequence_length, value=1)


def _patch_sequence_mask(test):
    #  Wrap mx.sym to make it easily patchable.  All un-patched methods will fall-back to their default implementation.
    with patch.object(mx, 'sym', wraps=mx.sym) as mxnet_mock:
        #  Patch Sequence Mask to use ones for padding.
        mxnet_mock.SequenceMask = _mask_with_one

        test()

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

import sockeye.constants as C
import sockeye.coverage
import sockeye.rnn_attention
from test.common import gaussian_vector, integer_vector

attention_types = [C.ATT_BILINEAR, C.ATT_DOT, C.ATT_LOC, C.ATT_MLP]


def test_att_bilinear():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_BILINEAR,
                                                             num_hidden=None,
                                                             input_previous_word=True,
                                                             source_num_hidden=None,
                                                             query_num_hidden=6,
                                                             layer_normalization=False,
                                                             config_coverage=None)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=None)

    assert type(attention) == sockeye.rnn_attention.BilinearAttention
    assert not attention._input_previous_word
    assert attention.num_hidden == 6


def test_att_dot():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_DOT,
                                                             num_hidden=2,
                                                             input_previous_word=True,
                                                             source_num_hidden=4,
                                                             query_num_hidden=6,
                                                             layer_normalization=False,
                                                             config_coverage=None,
                                                             is_scaled=False)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=None)

    assert type(attention) == sockeye.rnn_attention.DotAttention
    assert attention._input_previous_word
    assert attention.project_source
    assert attention.project_query
    assert attention.num_hidden == 2
    assert attention.is_scaled is False
    assert not attention.scale


def test_att_dot_scaled():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_DOT,
                                                             num_hidden=16,
                                                             input_previous_word=True,
                                                             source_num_hidden=None,
                                                             query_num_hidden=None,
                                                             layer_normalization=False,
                                                             config_coverage=None,
                                                             is_scaled=True)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=None)

    assert type(attention) == sockeye.rnn_attention.DotAttention
    assert attention._input_previous_word
    assert attention.project_source
    assert attention.project_query
    assert attention.num_hidden == 16
    assert attention.is_scaled is True
    assert attention.scale == 0.25


def test_att_mh_dot():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_MH_DOT,
                                                             num_hidden=None,
                                                             input_previous_word=True,
                                                             source_num_hidden=8,
                                                             query_num_hidden=None,
                                                             layer_normalization=False,
                                                             config_coverage=None,
                                                             num_heads=2)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=None)

    assert type(attention) == sockeye.rnn_attention.MultiHeadDotAttention
    assert attention._input_previous_word
    assert attention.num_hidden == 8
    assert attention.heads == 2
    assert attention.num_hidden_per_head == 4


def test_att_fixed():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_FIXED,
                                                             num_hidden=None,
                                                             input_previous_word=True,
                                                             source_num_hidden=None,
                                                             query_num_hidden=None,
                                                             layer_normalization=False,
                                                             config_coverage=None)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=None)

    assert type(attention) == sockeye.rnn_attention.EncoderLastStateAttention
    assert attention._input_previous_word


def test_att_loc():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_LOC,
                                                             num_hidden=None,
                                                             input_previous_word=True,
                                                             source_num_hidden=None,
                                                             query_num_hidden=None,
                                                             layer_normalization=False,
                                                             config_coverage=None)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=10)

    assert type(attention) == sockeye.rnn_attention.LocationAttention
    assert attention._input_previous_word
    assert attention.max_source_seq_len == 10


def test_att_mlp():
    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_MLP,
                                                             num_hidden=16,
                                                             input_previous_word=True,
                                                             source_num_hidden=None,
                                                             query_num_hidden=None,
                                                             layer_normalization=True,
                                                             config_coverage=None)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=10)

    assert type(attention) == sockeye.rnn_attention.MlpAttention
    assert attention._input_previous_word
    assert attention.attention_num_hidden == 16
    assert attention.dynamic_source_num_hidden == 1
    assert attention._ln
    assert not attention.coverage


def test_att_cov():
    config_coverage = sockeye.coverage.CoverageConfig(type='tanh', num_hidden=5, layer_normalization=True)

    config_attention = sockeye.rnn_attention.AttentionConfig(type=C.ATT_COV,
                                                             num_hidden=16,
                                                             input_previous_word=True,
                                                             source_num_hidden=None,
                                                             query_num_hidden=None,
                                                             layer_normalization=True,
                                                             config_coverage=config_coverage)

    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=10)

    assert type(attention) == sockeye.rnn_attention.MlpCovAttention
    assert attention._input_previous_word
    assert attention.attention_num_hidden == 16
    assert attention.dynamic_source_num_hidden == 5
    assert attention._ln
    assert type(attention.coverage) == sockeye.coverage.ActivationCoverage


@pytest.mark.parametrize("attention_type", attention_types)
def test_attention(attention_type,
                   batch_size=1,
                   encoder_num_hidden=2,
                   decoder_num_hidden=2):
    # source: (batch_size, seq_len, encoder_num_hidden)
    source = mx.sym.Variable("source")
    # source_length: (batch_size,)
    source_length = mx.sym.Variable("source_length")
    source_seq_len = 3

    config_attention = sockeye.rnn_attention.AttentionConfig(type=attention_type,
                                                             num_hidden=2,
                                                             input_previous_word=False,
                                                             source_num_hidden=2,
                                                             query_num_hidden=2,
                                                             layer_normalization=False,
                                                             config_coverage=None)
    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=source_seq_len)

    attention_state = attention.get_initial_state(source_length, source_seq_len)
    attention_func = attention.on(source, source_length, source_seq_len)
    attention_input = attention.make_input(0, mx.sym.Variable("word_vec_prev"), mx.sym.Variable("decoder_state"))
    attention_state = attention_func(attention_input, attention_state)
    sym = mx.sym.Group([attention_state.context, attention_state.probs])

    executor = sym.simple_bind(ctx=mx.cpu(),
                               source=(batch_size, source_seq_len, encoder_num_hidden),
                               source_length=(batch_size,),
                               decoder_state=(batch_size, decoder_num_hidden))

    # TODO: test for other inputs (that are not equal at each source position)
    executor.arg_dict["source"][:] = np.asarray([[[1., 2.], [1., 2.], [3., 4.]]])
    executor.arg_dict["source_length"][:] = np.asarray([2.0])
    executor.arg_dict["decoder_state"][:] = np.asarray([[5, 6]])
    exec_output = executor.forward()
    context_result = exec_output[0].asnumpy()
    attention_prob_result = exec_output[1].asnumpy()

    # expecting uniform attention_weights of 0.5: 0.5 * seq1 + 0.5 * seq2
    assert np.isclose(context_result, np.asarray([[1., 2.]])).all()
    # equal attention to first two and no attention to third
    assert np.isclose(attention_prob_result, np.asarray([[0.5, 0.5, 0.]])).all()


coverage_cases = [("gru", 10), ("tanh", 4), ("count", 1), ("sigmoid", 1), ("relu", 30)]


@pytest.mark.parametrize("attention_coverage_type,attention_coverage_num_hidden", coverage_cases)
def test_coverage_attention(attention_coverage_type,
                            attention_coverage_num_hidden,
                            batch_size=3,
                            encoder_num_hidden=2,
                            decoder_num_hidden=2):
    # source: (batch_size, seq_len, encoder_num_hidden)
    source = mx.sym.Variable("source")
    # source_length: (batch_size, )
    source_length = mx.sym.Variable("source_length")
    source_seq_len = 10

    config_coverage = sockeye.coverage.CoverageConfig(type=attention_coverage_type,
                                                      num_hidden=attention_coverage_num_hidden,
                                                      layer_normalization=False)
    config_attention = sockeye.rnn_attention.AttentionConfig(type="coverage",
                                                             num_hidden=5,
                                                             input_previous_word=False,
                                                             source_num_hidden=encoder_num_hidden,
                                                             query_num_hidden=decoder_num_hidden,
                                                             layer_normalization=False,
                                                             config_coverage=config_coverage)
    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=source_seq_len)

    attention_state = attention.get_initial_state(source_length, source_seq_len)
    attention_func = attention.on(source, source_length, source_seq_len)
    attention_input = attention.make_input(0, mx.sym.Variable("word_vec_prev"), mx.sym.Variable("decoder_state"))
    attention_state = attention_func(attention_input, attention_state)
    sym = mx.sym.Group([attention_state.context, attention_state.probs, attention_state.dynamic_source])

    source_shape = (batch_size, source_seq_len, encoder_num_hidden)
    source_length_shape = (batch_size,)
    decoder_state_shape = (batch_size, decoder_num_hidden)

    executor = sym.simple_bind(ctx=mx.cpu(),
                               source=source_shape,
                               source_length=source_length_shape,
                               decoder_state=decoder_state_shape)

    source_length_vector = integer_vector(shape=source_length_shape, max_value=source_seq_len)
    executor.arg_dict["source"][:] = gaussian_vector(shape=source_shape)
    executor.arg_dict["source_length"][:] = source_length_vector
    executor.arg_dict["decoder_state"][:] = gaussian_vector(shape=decoder_state_shape)
    exec_output = executor.forward()
    context_result = exec_output[0].asnumpy()
    attention_prob_result = exec_output[1].asnumpy()
    dynamic_source_result = exec_output[2].asnumpy()

    expected_probs = (1. / source_length_vector).reshape((batch_size, 1))

    assert context_result.shape == (batch_size, encoder_num_hidden)
    assert attention_prob_result.shape == (batch_size, source_seq_len)
    assert dynamic_source_result.shape == (batch_size, source_seq_len, attention_coverage_num_hidden)
    assert (np.sum(np.isclose(attention_prob_result, expected_probs), axis=1) == source_length_vector).all()


def test_last_state_attention(batch_size=1,
                              encoder_num_hidden=2):
    """
    EncoderLastStateAttention is a bit different from other attention mechanisms as it doesn't take a query argument
    and doesn't return a probability distribution over the inputs (aka alignment).
    """
    # source: (batch_size, seq_len, encoder_num_hidden)
    source = mx.sym.Variable("source")
    # source_length: (batch_size,)
    source_length = mx.sym.Variable("source_length")
    source_seq_len = 3

    config_attention = sockeye.rnn_attention.AttentionConfig(type="fixed",
                                                             num_hidden=0,
                                                             input_previous_word=False,
                                                             source_num_hidden=2,
                                                             query_num_hidden=2,
                                                             layer_normalization=False,
                                                             config_coverage=None)
    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=source_seq_len)

    attention_state = attention.get_initial_state(source_length, source_seq_len)
    attention_func = attention.on(source, source_length, source_seq_len)
    attention_input = attention.make_input(0, mx.sym.Variable("word_vec_prev"), mx.sym.Variable("decoder_state"))
    attention_state = attention_func(attention_input, attention_state)
    sym = mx.sym.Group([attention_state.context, attention_state.probs])

    executor = sym.simple_bind(ctx=mx.cpu(),
                               source=(batch_size, source_seq_len, encoder_num_hidden),
                               source_length=(batch_size,))

    # TODO: test for other inputs (that are not equal at each source position)
    executor.arg_dict["source"][:] = np.asarray([[[1., 2.], [1., 2.], [3., 4.]]])
    executor.arg_dict["source_length"][:] = np.asarray([2.0])
    exec_output = executor.forward()
    context_result = exec_output[0].asnumpy()
    attention_prob_result = exec_output[1].asnumpy()

    # expecting attention on last state based on source_length
    assert np.isclose(context_result, np.asarray([[1., 2.]])).all()
    assert np.isclose(attention_prob_result, np.asarray([[0., 1.0, 0.]])).all()


def test_get_context_and_attention_probs():
    source = mx.sym.Variable('source')
    source_length = mx.sym.Variable('source_length')
    attention_scores = mx.sym.Variable('scores')
    context, att_probs = sockeye.rnn_attention.get_context_and_attention_probs(
        source,
        source_length,
        attention_scores,
        C.DTYPE_FP32)
    sym = mx.sym.Group([context, att_probs])
    assert len(sym.list_arguments()) == 3

    batch_size, seq_len, num_hidden = 32, 50, 100

    # data
    source_nd = mx.nd.random_normal(shape=(batch_size, seq_len, num_hidden))
    source_length_np = np.random.randint(1, seq_len+1, (batch_size,))
    source_length_nd = mx.nd.array(source_length_np)
    scores_nd = mx.nd.zeros((batch_size, seq_len, 1))

    in_shapes, out_shapes, _ = sym.infer_shape(source=source_nd.shape,
                                               source_length=source_length_nd.shape,
                                               scores=scores_nd.shape)

    assert in_shapes == [(batch_size, seq_len, num_hidden), (batch_size, seq_len, 1), (batch_size,)]
    assert out_shapes == [(batch_size, num_hidden), (batch_size, seq_len)]

    context, probs = sym.eval(source=source_nd,
                              source_length=source_length_nd,
                              scores=scores_nd)

    expected_probs = (1. / source_length_nd).reshape((batch_size, 1)).asnumpy()
    assert (np.sum(np.isclose(probs.asnumpy(), expected_probs), axis=1) == source_length_np).all()

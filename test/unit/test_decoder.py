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

import sockeye.rnn_attention
import sockeye.rnn
import sockeye.constants as C
import sockeye.coverage
import sockeye.decoder
import sockeye.transformer
from test.common import gaussian_vector, integer_vector

step_tests = [(C.GRU_TYPE, True), (C.LSTM_TYPE, False)]


def test_get_decoder():
    config = sockeye.transformer.TransformerConfig(
        model_size=20,
        attention_heads=10,
        feed_forward_num_hidden=30,
        act_type='test_act',
        num_layers=50,
        dropout_attention=0.5,
        dropout_act=0.6,
        dropout_prepost=0.1,
        positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
        preprocess_sequence=C.FIXED_POSITIONAL_EMBEDDING,
        postprocess_sequence='test_post_seq',
        max_seq_len_source=60,
        max_seq_len_target=70,
        conv_config=None)
    decoder = sockeye.decoder.get_decoder(config, 'test_')

    assert type(decoder) == sockeye.decoder.TransformerDecoder
    assert decoder.prefix == 'test_' + C.TRANSFORMER_DECODER_PREFIX


@pytest.mark.parametrize("cell_type, context_gating", step_tests)
def test_step(cell_type, context_gating,
              num_embed=2,
              encoder_num_hidden=5,
              decoder_num_hidden=5):

    vocab_size, batch_size, source_seq_len = 10, 10, 7,

    # (batch_size, source_seq_len, encoder_num_hidden)
    source = mx.sym.Variable("source")
    source_shape = (batch_size, source_seq_len, encoder_num_hidden)
    # (batch_size,)
    source_length = mx.sym.Variable("source_length")
    source_length_shape = (batch_size,)
    # (batch_size, num_embed)
    word_vec_prev = mx.sym.Variable("word_vec_prev")
    word_vec_prev_shape = (batch_size, num_embed)
    # (batch_size, decoder_num_hidden)
    hidden_prev = mx.sym.Variable("hidden_prev")
    hidden_prev_shape = (batch_size, decoder_num_hidden)
    # List(mx.sym.Symbol(batch_size, decoder_num_hidden)
    states_shape = (batch_size, decoder_num_hidden)

    config_coverage = sockeye.coverage.CoverageConfig(type="tanh",
                                                      num_hidden=2,
                                                      layer_normalization=False)
    config_attention = sockeye.rnn_attention.AttentionConfig(type="coverage",
                                                             num_hidden=2,
                                                             input_previous_word=False,
                                                             source_num_hidden=decoder_num_hidden,
                                                             query_num_hidden=decoder_num_hidden,
                                                             layer_normalization=False,
                                                             config_coverage=config_coverage)
    attention = sockeye.rnn_attention.get_attention(config_attention, max_seq_len=source_seq_len)
    attention_state = attention.get_initial_state(source_length, source_seq_len)
    attention_func = attention.on(source, source_length, source_seq_len)

    config_rnn = sockeye.rnn.RNNConfig(cell_type=cell_type,
                                       num_hidden=decoder_num_hidden,
                                       num_layers=1,
                                       dropout_inputs=0.,
                                       dropout_states=0.,
                                       residual=False,
                                       forget_bias=0.)

    config_decoder = sockeye.decoder.RecurrentDecoderConfig(max_seq_len_source=source_seq_len,
                                                            rnn_config=config_rnn,
                                                            attention_config=config_attention,
                                                            context_gating=context_gating)

    decoder = sockeye.decoder.RecurrentDecoder(config=config_decoder)

    if cell_type == C.GRU_TYPE:
        layer_states = [gaussian_vector(shape=states_shape, return_symbol=True) for _ in range(config_rnn.num_layers)]
    elif cell_type == C.LSTM_TYPE:
        layer_states = [gaussian_vector(shape=states_shape, return_symbol=True) for _ in range(config_rnn.num_layers*2)]
    else:
        raise ValueError

    state, attention_state = decoder._step(word_vec_prev=word_vec_prev,
                                           state=sockeye.decoder.RecurrentDecoderState(hidden_prev, layer_states),
                                           attention_func=attention_func,
                                           attention_state=attention_state)
    sym = mx.sym.Group([state.hidden, attention_state.probs, attention_state.dynamic_source])

    executor = sym.simple_bind(ctx=mx.cpu(),
                               source=source_shape,
                               source_length=source_length_shape,
                               word_vec_prev=word_vec_prev_shape,
                               hidden_prev=hidden_prev_shape)
    executor.arg_dict["source"][:] = gaussian_vector(source_shape)
    executor.arg_dict["source_length"][:] = integer_vector(source_length_shape, source_seq_len)
    executor.arg_dict["word_vec_prev"][:] = gaussian_vector(word_vec_prev_shape)
    executor.arg_dict["hidden_prev"][:] = gaussian_vector(hidden_prev_shape)
    executor.arg_dict["states"] = layer_states
    hidden_result, attention_probs_result, attention_dynamic_source_result = executor.forward()

    assert hidden_result.shape == hidden_prev_shape
    assert attention_probs_result.shape == (batch_size, source_seq_len)
    assert attention_dynamic_source_result.shape == (batch_size, source_seq_len, config_coverage.num_hidden)

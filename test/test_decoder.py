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
import sockeye.decoder
import sockeye.attention
import sockeye.constants as C
from test.test_utils import gaussian_vector, integer_vector


step_tests = [(C.GRU_TYPE, True), (C.LSTM_TYPE, False)]


@pytest.mark.parametrize("cell_type, context_gating", step_tests)
def test_step(cell_type, context_gating,
              num_embed=2,
              encoder_num_hidden=5,
              decoder_num_hidden=5):

    attention_num_hidden, vocab_size, num_layers, \
    batch_size, source_seq_len, coverage_num_hidden = 2, 10, 1, 10, 7, 2

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

    attention = sockeye.attention.get_attention(input_previous_word=False,
                                                attention_type="coverage",
                                                attention_num_hidden=attention_num_hidden,
                                                rnn_num_hidden=decoder_num_hidden,
                                                max_seq_len=source_seq_len,
                                                attention_coverage_type="tanh",
                                                attention_coverage_num_hidden=coverage_num_hidden)
    attention_state = attention.get_initial_state(source_length, source_seq_len)
    attention_func = attention.on(source, source_length, source_seq_len)

    decoder = sockeye.decoder.get_decoder(num_embed=num_embed,
                                          vocab_size=vocab_size,
                                          num_layers=num_layers,
                                          rnn_num_hidden=decoder_num_hidden,
                                          attention=attention,
                                          cell_type=cell_type,
                                          residual=False,
                                          forget_bias=0.,
                                          dropout=0.,
                                          weight_tying=False,
                                          lexicon=None,
                                          context_gating=context_gating)

    if cell_type == C.GRU_TYPE:
        layer_states = [gaussian_vector(shape=states_shape, return_symbol=True) for _ in range(num_layers)]
    elif cell_type == C.LSTM_TYPE:
        layer_states = [gaussian_vector(shape=states_shape, return_symbol=True) for _ in range(num_layers*2)]

    state, attention_state = decoder._step(word_vec_prev=word_vec_prev,
                                           state=sockeye.decoder.DecoderState(hidden_prev, layer_states),
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
    assert attention_dynamic_source_result.shape == (batch_size, source_seq_len, coverage_num_hidden)

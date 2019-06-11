# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from unittest import mock

import pytest

import sockeye.constants as C
from sockeye.training import TrainingModel


NUM_LAYERS = 3

# Abbreviated version of weights from different model types.
ALL_PARAMS = [
    # RNN
    'encoder_birnn_forward_l0_W',
    'encoder_birnn_reverse_l0_W',
    'encoder_rnn_l0_W',
    'encoder_rnn_l1_W',
    'decoder_rnn_att_W',
    'decoder_rnn_enc2decinit_W',
    'decoder_rnn_hidden_W',
    'decoder_rnn_l0_W',
    'decoder_rnn_l1_W',
    'decoder_rnn_l2_W',
    # Transformer
    'encoder_transformer_0_W',
    'encoder_transformer_1_W',
    'encoder_transformer_2_W',
    'encoder_transformer_final_W',
    'decoder_transformer_0_W',
    'decoder_transformer_1_W',
    'decoder_transformer_2_W',
    'decoder_transformer_final_W',
    # CNN
    'encoder_cnn_0_W',
    'encoder_cnn_1_W',
    'encoder_cnn_2_W',
    'encoder_cnn_i2h_W',
    'decoder_cnn_0_W',
    'decoder_cnn_1_W',
    'decoder_cnn_2_W',
    'decoder_cnn_i2h_W',
    # Embeddings
    'source_embed_factor0_weight',
    'source_embed_factor1_weight',
    'source_embed_weight',
    'source_pos_embed_weight',
    'target_embed_weight',
    'target_pos_embed_weight',
    'source_target_embed_weight',
    # Output
    'target_output_bias',
    'target_output_weight',
]

ALL_EXCEPT_DECODER_PARAMS = [
    # RNN
    'encoder_birnn_forward_l0_W',
    'encoder_birnn_reverse_l0_W',
    'encoder_rnn_l0_W',
    'encoder_rnn_l1_W',
    # Transformer
    'encoder_transformer_0_W',
    'encoder_transformer_1_W',
    'encoder_transformer_2_W',
    'encoder_transformer_final_W',
    # CNN
    'encoder_cnn_0_W',
    'encoder_cnn_1_W',
    'encoder_cnn_2_W',
    'encoder_cnn_i2h_W',
    # Embeddings
    'source_embed_factor0_weight',
    'source_embed_factor1_weight',
    'source_embed_weight',
    'source_pos_embed_weight',
    'target_embed_weight',
    'target_pos_embed_weight',
    'source_target_embed_weight',
    # Output
    'target_output_bias',
    'target_output_weight',
]

ALL_EXCEPT_OUTER_LAYERS_PARAMS = [
    # RNN
    'encoder_rnn_l0_W',
    'decoder_rnn_att_W',
    'decoder_rnn_enc2decinit_W',
    'decoder_rnn_hidden_W',
    'decoder_rnn_l1_W',
    # Transformer
    'encoder_transformer_1_W',
    'encoder_transformer_final_W',
    'decoder_transformer_1_W',
    'decoder_transformer_final_W',
    # CNN
    'encoder_cnn_1_W',
    'encoder_cnn_i2h_W',
    'decoder_cnn_1_W',
    'decoder_cnn_i2h_W',
    # Embeddings
    'source_embed_factor0_weight',
    'source_embed_factor1_weight',
    'source_embed_weight',
    'source_pos_embed_weight',
    'target_embed_weight',
    'target_pos_embed_weight',
    'source_target_embed_weight',
    # Output
    'target_output_bias',
    'target_output_weight',
]

ALL_EXCEPT_EMBED_PARAMS = [
    # RNN
    'encoder_birnn_forward_l0_W',
    'encoder_birnn_reverse_l0_W',
    'encoder_rnn_l0_W',
    'encoder_rnn_l1_W',
    'decoder_rnn_att_W',
    'decoder_rnn_enc2decinit_W',
    'decoder_rnn_hidden_W',
    'decoder_rnn_l0_W',
    'decoder_rnn_l1_W',
    'decoder_rnn_l2_W',
    # Transformer
    'encoder_transformer_0_W',
    'encoder_transformer_1_W',
    'encoder_transformer_2_W',
    'encoder_transformer_final_W',
    'decoder_transformer_0_W',
    'decoder_transformer_1_W',
    'decoder_transformer_2_W',
    'decoder_transformer_final_W',
    # CNN
    'encoder_cnn_0_W',
    'encoder_cnn_1_W',
    'encoder_cnn_2_W',
    'encoder_cnn_i2h_W',
    'decoder_cnn_0_W',
    'decoder_cnn_1_W',
    'decoder_cnn_2_W',
    'decoder_cnn_i2h_W',
    # Output
    'target_output_bias',
    'target_output_weight',
]

ALL_EXCEPT_OUTPUT_PROJ_PARAMS = [
    # RNN
    'encoder_birnn_forward_l0_W',
    'encoder_birnn_reverse_l0_W',
    'encoder_rnn_l0_W',
    'encoder_rnn_l1_W',
    'decoder_rnn_att_W',
    'decoder_rnn_enc2decinit_W',
    'decoder_rnn_hidden_W',
    'decoder_rnn_l0_W',
    'decoder_rnn_l1_W',
    'decoder_rnn_l2_W',
    # Transformer
    'encoder_transformer_0_W',
    'encoder_transformer_1_W',
    'encoder_transformer_2_W',
    'encoder_transformer_final_W',
    'decoder_transformer_0_W',
    'decoder_transformer_1_W',
    'decoder_transformer_2_W',
    'decoder_transformer_final_W',
    # CNN
    'encoder_cnn_0_W',
    'encoder_cnn_1_W',
    'encoder_cnn_2_W',
    'encoder_cnn_i2h_W',
    'decoder_cnn_0_W',
    'decoder_cnn_1_W',
    'decoder_cnn_2_W',
    'decoder_cnn_i2h_W',
    # Embeddings
    'source_embed_factor0_weight',
    'source_embed_factor1_weight',
    'source_embed_weight',
    'source_pos_embed_weight',
    'target_embed_weight',
    'target_pos_embed_weight',
    'source_target_embed_weight',
]

@pytest.mark.parametrize("param_names, strategy, expected_fixed_param_names", [
    (ALL_PARAMS, C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER, ALL_EXCEPT_DECODER_PARAMS),
    (ALL_PARAMS, C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTER_LAYERS, ALL_EXCEPT_OUTER_LAYERS_PARAMS),
    (ALL_PARAMS, C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_EMBEDDINGS, ALL_EXCEPT_EMBED_PARAMS),
    (ALL_PARAMS, C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTPUT_PROJ, ALL_EXCEPT_OUTPUT_PROJ_PARAMS),
])
def test_fixed_param_strategy(param_names, strategy, expected_fixed_param_names):
    model = mock.Mock()
    model.config.config_encoder.num_layers = NUM_LAYERS
    model.config.config_decoder.num_layers = NUM_LAYERS
    fixed_param_names = TrainingModel._generate_fixed_param_names(model, param_names, strategy)
    assert fixed_param_names == expected_fixed_param_names

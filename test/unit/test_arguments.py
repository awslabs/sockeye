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

import argparse

import pytest

import sockeye.arguments as arguments
import sockeye.constants as C


@pytest.mark.parametrize("test_params, expected_params", [
    # mandatory parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=False,
          monitor_pattern=None, monitor_stat_func='mx_default')),

    # all parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output '
     '--source-vocab test_src_vocab --target-vocab test_tgt_vocab '
     '--use-tensorboard --overwrite-output --quiet',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=True,
          source_vocab='test_src_vocab', target_vocab='test_tgt_vocab', use_tensorboard=True, quiet=True,
          monitor_pattern=None, monitor_stat_func='mx_default')),

    # short parameters
    ('-s test_src -t test_tgt '
     '-vs test_validation_src -vt test_validation_tgt '
     '-o test_output -q',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=True,
          monitor_pattern=None, monitor_stat_func='mx_default'))
])
def test_io_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_io_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('', dict(device_ids=[-1], use_cpu=False, disable_device_locking=False, lock_dir='/tmp')),
    ('--device-ids 1 2 3 --use-cpu --disable-device-locking --lock-dir test_dir',
     dict(device_ids=[1, 2, 3], use_cpu=True, disable_device_locking=True, lock_dir='test_dir'))
])
def test_device_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_device_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('', dict(params=None,
              num_words=(50000,50000),
              word_min_count=(1,1),
              num_layers=(1,1),
              num_embed=(512,512),
              attention_type='mlp',
              attention_num_hidden=None,
              attention_coverage_type='count',
              attention_coverage_num_hidden=1,
              lexical_bias=None,
              learn_lexical_bias=False,
              weight_tying=False,
              weight_tying_type="trg_softmax",
              max_seq_len=(100,100),
              attention_mhdot_heads=None,
              transformer_attention_heads=8,
              transformer_feed_forward_num_hidden=2048,
              transformer_model_size=512,
              transformer_no_positional_encodings=False,
              attention_use_prev_word=False,
              rnn_decoder_zero_init=False,
              rnn_encoder_reverse_input=False,
              rnn_context_gating=False,
              rnn_cell_type=C.LSTM_TYPE,
              rnn_num_hidden=1024,
              rnn_residual_connections=False,
              rnn_first_residual_layer=2,
              layer_normalization=False,
              encoder=C.RNN_NAME,
              conv_embed_max_filter_width=8,
              decoder=C.RNN_NAME,
              conv_embed_output_dim=None,
              conv_embed_num_filters=(200, 200, 250, 250, 300, 300, 300, 300),
              conv_embed_num_highway_layers=4,
              conv_embed_pool_stride=5,
              conv_embed_add_positional_encodings=False))])
def test_model_parameters(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_model_parameters)


@pytest.mark.parametrize("test_params, expected_params", [
    ('', dict(batch_size=64,
              fill_up='replicate',
              no_bucketing=False,
              bucket_width=10,
              loss=C.CROSS_ENTROPY,
              smoothed_cross_entropy_alpha=0.3,
              normalize_loss=False,
              metrics=[C.PERPLEXITY],
              optimized_metric=C.PERPLEXITY,
              max_updates=-1,
              checkpoint_frequency=1000,
              max_num_checkpoint_not_improved=8,
              embed_dropout=(.0, .0),
              transformer_dropout_attention=0.0,
              transformer_dropout_relu=0.0,
              transformer_dropout_residual=0.0,
              conv_embed_dropout=0.0,
              optimizer='adam', min_num_epochs=0,
              initial_learning_rate=0.0003,
              weight_decay=0.0,
              momentum=None,
              clip_gradient=1.0,
              learning_rate_scheduler_type='plateau-reduce',
              learning_rate_reduce_factor=0.5,
              learning_rate_reduce_num_not_improved=3,
              learning_rate_half_life=10,
              learning_rate_warmup=0,
              learning_rate_schedule=None,
              use_fused_rnn=False,
              weight_init='xavier',
              weight_init_scale=0.04,
              rnn_dropout=(.0, .0),
              rnn_decoder_hidden_dropout=.0,
              rnn_forget_bias=0.0,
              rnn_h2h_init=C.RNN_INIT_ORTHOGONAL,
              monitor_bleu=0,
              seed=13,
              keep_last_params=-1)),
])
def test_training_arg(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_training_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('--models m1 m2 m3', dict(input=None,
                               output=None,
                               models=['m1', 'm2', 'm3'],
                               checkpoints=None,
                               beam_size=5,
                               ensemble_mode='linear',
                               max_input_len=None,
                               softmax_temperature=None,
                               output_type='translation',
                               sure_align_threshold=0.9,
                               length_penalty_alpha=1.0,
                               length_penalty_beta=0.0)),
])
def test_inference_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_inference_args)


def _test_args(test_params, expected_params, args_func):
    test_parser = argparse.ArgumentParser()
    args_func(test_parser)
    parsed_params = test_parser.parse_args(test_params.split())
    assert dict(vars(parsed_params)) == expected_params

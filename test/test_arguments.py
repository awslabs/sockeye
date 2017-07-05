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

import pytest

import sockeye.constants as C
import sockeye.arguments as arguments
import argparse


@pytest.mark.parametrize("test_params, expected_params", [
    # mandatory parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=False)),

    # all parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output '
     '--source-vocab test_src_vocab --target-vocab test_tgt_vocab '
     '--use-tensorboard --overwrite-output --quiet',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=True,
          source_vocab='test_src_vocab', target_vocab='test_tgt_vocab', use_tensorboard=True, quiet=True)),

    # short parameters
    ('-s test_src -t test_tgt '
     '-vs test_validation_src -vt test_validation_tgt '
     '-o test_output -q',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=True))
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
    ('', dict(params=None, num_words=50000, word_min_count=1, rnn_num_layers=1, rnn_cell_type=C.LSTM_TYPE, rnn_num_hidden=1024,
              rnn_residual_connections=False, num_embed=512, num_embed_source=None, num_embed_target=None,
              attention_type='mlp', attention_num_hidden=None, attention_coverage_type='count',
              attention_coverage_num_hidden=1,
              lexical_bias=None, learn_lexical_bias=False, weight_tying=False, max_seq_len=100,
              attention_use_prev_word=False, context_gating=False)),
    ('--params test_params --num-words 10 --word-min-count 10 --rnn-num-layers 10 --rnn-cell-type gru '
     '--rnn-num-hidden 512 --rnn-residual-connections --num-embed 1024 --num-embed-source 10 --num-embed-target 10 '
     '--attention-type dot --attention-num-hidden 10 --attention-coverage-type tanh '
     '--attention-coverage-num-hidden 10 --lexical-bias test_bias --learn-lexical-bias --weight-tying '
     '--max-seq-len 10 --attention-use-prev-word --context-gating',
     dict(params='test_params', num_words=10, word_min_count=10, rnn_num_layers=10, rnn_cell_type=C.GRU_TYPE,
          rnn_num_hidden=512,
          rnn_residual_connections=True, num_embed=1024, num_embed_source=10, num_embed_target=10,
          attention_type='dot', attention_num_hidden=10, attention_coverage_type='tanh',
          attention_coverage_num_hidden=10,
          lexical_bias='test_bias', learn_lexical_bias=True, weight_tying=True, max_seq_len=10,
          attention_use_prev_word=True, context_gating=True))
])
def test_model_parameters(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_model_parameters)


@pytest.mark.parametrize("test_params, expected_params", [
    ('', dict(batch_size=64, fill_up='replicate', no_bucketing=False, bucket_width=10, loss=C.CROSS_ENTROPY,
              smoothed_cross_entropy_alpha=0.3, normalize_loss=False, metrics=[C.PERPLEXITY],
              optimized_metric=C.PERPLEXITY,
              max_updates=-1, checkpoint_frequency=1000, max_num_checkpoint_not_improved=8, dropout=0.0,
              optimizer='adam', min_num_epochs=0,
              initial_learning_rate=0.0003, weight_decay=0.0, momentum=None, clip_gradient=1.0,
              learning_rate_scheduler_type='plateau-reduce', learning_rate_reduce_factor=0.5,
              learning_rate_reduce_num_not_improved=3, learning_rate_half_life=10, use_fused_rnn=False,
              rnn_forget_bias=0.0, rnn_h2h_init=C.RNN_INIT_ORTHOGONAL, monitor_bleu=0, seed=13)),
    ('--batch-size 128 --fill-up test_fill_up --no-bucketing --bucket-width 20 --loss smoothed-cross-entropy '
     '--smoothed-cross-entropy-alpha 1.0 --normalize-loss --metrics perplexity accuracy '
     '--optimized-metric bleu --max-updates 10 --checkpoint-frequency 10 --min-num-epochs 10 '
     '--max-num-checkpoint-not-improved 16 --dropout 1.0 --optimizer sgd --initial-learning-rate 1.0 '
     '--weight-decay 1.0 --momentum 1.0 --clip-gradient 2.0 --learning-rate-scheduler-type fixed-rate-inv-t '
     '--learning-rate-reduce-factor 1.0 --learning-rate-reduce-num-not-improved 10 --learning-rate-half-life 20 '
     '--use-fused-rnn --rnn-forget-bias 1.0 --rnn-h2h-init orthogonal_stacked --monitor-bleu 10 --seed 10',
    dict(batch_size=128, fill_up='test_fill_up', no_bucketing=True, bucket_width=20, loss=C.SMOOTHED_CROSS_ENTROPY,
         smoothed_cross_entropy_alpha=1.0, normalize_loss=True, metrics=[C.PERPLEXITY, C.ACCURACY],
         optimized_metric=C.BLEU, min_num_epochs=10,
         max_updates=10, checkpoint_frequency=10, max_num_checkpoint_not_improved=16, dropout=1.0, optimizer='sgd',
         initial_learning_rate=1.0, weight_decay=1.0, momentum=1.0, clip_gradient=2.0,
         learning_rate_scheduler_type='fixed-rate-inv-t', learning_rate_reduce_factor=1.0,
         learning_rate_reduce_num_not_improved=10, learning_rate_half_life=20.0, use_fused_rnn=True,
         rnn_forget_bias=1.0, rnn_h2h_init=C.RNN_INIT_ORTHOGONAL_STACKED, monitor_bleu=10, seed=10)),
])
def test_training_arg(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_training_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('--models m1 m2 m3', dict(input=None, output=None, models=['m1', 'm2', 'm3'],
                               checkpoints=None, beam_size=5, ensemble_mode='linear',
                               max_input_len=None, softmax_temperature=None, output_type='translation',
                               sure_align_threshold=0.9)),
    ('--input test_input --output test_output --models m1 m2 m3 --checkpoints 1 2 3 --beam-size 10 '
     '--ensemble-mode log_linear --max-input-len 10 --softmax-temperature 1.0 '
     '--output-type translation_with_alignments --sure-align-threshold 1.0',
     dict(input='test_input', output='test_output', models=['m1', 'm2', 'm3'],
          checkpoints=[1, 2, 3], beam_size=10, ensemble_mode='log_linear',
          max_input_len=10, softmax_temperature=1.0,
          output_type='translation_with_alignments', sure_align_threshold=1.0)),
    ('-i test_input -o test_output -m m1 m2 m3 -c 1 2 3 -b 10 -n 10',
     dict(input='test_input', output='test_output', models=['m1', 'm2', 'm3'],
          checkpoints=[1, 2, 3], beam_size=10, ensemble_mode='linear',
          max_input_len=10, softmax_temperature=None, output_type='translation', sure_align_threshold=0.9))
])
def test_inference_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_inference_args)


def _test_args(test_params, expected_params, args_func):
    test_parser = argparse.ArgumentParser()
    args_func(test_parser)
    parsed_params = test_parser.parse_args(test_params.split())
    assert dict(vars(parsed_params)) == expected_params

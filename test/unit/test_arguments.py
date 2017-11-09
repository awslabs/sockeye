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
import os

import sockeye.arguments as arguments
import sockeye.constants as C

from itertools import zip_longest

@pytest.mark.parametrize("test_params, expected_params", [
    # mandatory parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output',
     dict(source='test_src', target='test_tgt', limit=None,
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=False,
          monitor_pattern=None, monitor_stat_func='mx_default')),

    # all parameters
    ('--source test_src --target test_tgt --limit 10 '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output '
     '--source-vocab test_src_vocab --target-vocab test_tgt_vocab '
     '--use-tensorboard --overwrite-output --quiet',
     dict(source='test_src', target='test_tgt', limit=10,
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=True,
          source_vocab='test_src_vocab', target_vocab='test_tgt_vocab', use_tensorboard=True, quiet=True,
          monitor_pattern=None, monitor_stat_func='mx_default')),

    # short parameters
    ('-s test_src -t test_tgt '
     '-vs test_validation_src -vt test_validation_tgt '
     '-o test_output -q',
     dict(source='test_src', target='test_tgt', limit=None,
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
              num_words=(50000, 50000),
              word_min_count=(1, 1),
              num_layers=(1, 1),
              num_embed=(512, 512),
              rnn_attention_type='mlp',
              rnn_attention_num_hidden=None,
              rnn_attention_coverage_type='count',
              rnn_attention_coverage_num_hidden=1,
              weight_tying=False,
              weight_tying_type="trg_softmax",
              max_seq_len=(100, 100),
              rnn_attention_mhdot_heads=None,
              transformer_attention_heads=8,
              transformer_feed_forward_num_hidden=2048,
              transformer_model_size=512,
              transformer_positional_embedding_type="fixed",
              transformer_preprocess=('', ''),
              transformer_postprocess=('drn', 'drn'),
              rnn_attention_use_prev_word=False,
              rnn_decoder_state_init="last",
              rnn_encoder_reverse_input=False,
              rnn_context_gating=False,
              rnn_cell_type=C.LSTM_TYPE,
              rnn_num_hidden=1024,
              rnn_residual_connections=False,
              rnn_first_residual_layer=2,
              cnn_activation_type='glu',
              cnn_kernel_width=(3, 5),
              cnn_num_hidden=512,
              cnn_positional_embedding_type="learned",
              layer_normalization=False,
              weight_normalization=False,
              encoder=C.RNN_NAME,
              conv_embed_max_filter_width=8,
              decoder=C.RNN_NAME,
              conv_embed_output_dim=None,
              conv_embed_num_filters=(200, 200, 250, 250, 300, 300, 300, 300),
              conv_embed_num_highway_layers=4,
              conv_embed_pool_stride=5,
              conv_embed_add_positional_encodings=False,
              rnn_attention_in_upper_layers=False))])
def test_model_parameters(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_model_parameters)


@pytest.mark.parametrize("test_params, expected_params", [
    ('', dict(batch_size=64,
              batch_type="sentence",
              fill_up='replicate',
              no_bucketing=False,
              bucket_width=10,
              loss=C.CROSS_ENTROPY,
              label_smoothing=0.0,
              loss_normalization_type='valid',
              metrics=[C.PERPLEXITY],
              optimized_metric=C.PERPLEXITY,
              max_updates=None,
              checkpoint_frequency=1000,
              max_num_checkpoint_not_improved=8,
              embed_dropout=(.0, .0),
              transformer_dropout_attention=0.0,
              transformer_dropout_relu=0.0,
              transformer_dropout_prepost=0.0,
              conv_embed_dropout=0.0,
              optimizer='adam',
              optimizer_params=None,
              kvstore='device',
              min_num_epochs=None,
              max_num_epochs=None,
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
              learning_rate_decay_param_reset=False,
              learning_rate_decay_optimizer_states_reset='off',
              weight_init='xavier',
              weight_init_scale=2.34,
              weight_init_xavier_factor_type='in',
              embed_weight_init='default',
              rnn_dropout_inputs=(.0, .0),
              rnn_dropout_states=(.0, .0),
              rnn_dropout_recurrent=(.0, .0),
              rnn_decoder_hidden_dropout=.0,
              cnn_hidden_dropout=0.0,
              rnn_forget_bias=0.0,
              rnn_h2h_init=C.RNN_INIT_ORTHOGONAL,
              monitor_bleu=0,
              seed=13,
              keep_last_params=-1)),
])
def test_training_arg(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_training_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('-m model', dict(input=None,
                      output=None,
                      checkpoints=None,
                      models=['model'],
                      beam_size=5,
                      batch_size=1,
                      chunk_size=1,
                      ensemble_mode='linear',
                      bucket_width=(10, 2),
                      max_input_len=None,
                      restrict_lexicon=None,
                      softmax_temperature=None,
                      output_type='translation',
                      sure_align_threshold=0.9,
                      max_output_length_num_stds=2,
                      length_penalty_alpha=1.0,
                      length_penalty_beta=0.0)),
])
def test_inference_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_inference_args)


# Make sure that the parameter names and default values used in the tutorials do not change without the tutorials
# being updated accordingly.
@pytest.mark.parametrize("test_params, expected_params, expected_params_present", [
    # seqcopy tutorial
    ('-s train.source '
     '-t train.target '
     '-vs dev.source '
     '-vt dev.target '
     '--num-embed 32 '
     '--rnn-num-hidden 64 '
     '--rnn-attention-type dot '
     '--use-cpu '
     '--metrics perplexity accuracy '
     '--max-num-checkpoint-not-improved 3 '
     '-o seqcopy_model',
     dict(source="train.source",
          target="train.target",
          validation_source="dev.source",
          validation_target="dev.target",
          num_embed=(32, 32),
          rnn_num_hidden=64,
          use_cpu=True,
          metrics=['perplexity', 'accuracy'],
          max_num_checkpoint_not_improved=3,
          output="seqcopy_model",
          # The tutorial text mentions that we train a RNN model:
          encoder="rnn",
          decoder="rnn"),
     # Additionally we mention the checkpoint_frequency
     ['checkpoint_frequency']),
    # WMT tutorial
    ('-s corpus.tc.BPE.de '
     '-t corpus.tc.BPE.en '
     '-vs newstest2016.tc.BPE.de '
     '-vt newstest2016.tc.BPE.en '
     '--num-embed 256 '
     '--rnn-num-hidden 512 '
     '--rnn-attention-type dot '
     '--max-seq-len 60 '
     '--monitor-bleu 500 '
     '--use-tensorboard '
     '--use-cpu '
     '-o wmt_mode',
     dict(
         source="corpus.tc.BPE.de",
         target="corpus.tc.BPE.en",
         validation_source="newstest2016.tc.BPE.de",
         validation_target="newstest2016.tc.BPE.en",
         num_embed=(256, 256),
         rnn_num_hidden=512,
         rnn_attention_type='dot',
         max_seq_len=(60, 60),
         monitor_bleu=500,
         use_tensorboard=True,
         use_cpu=True,
         # Arguments mentioned in the text, should be renamed in the tutorial if they change:
         rnn_cell_type="lstm",
         encoder="rnn",
         decoder="rnn",
         optimizer="adam"),
     ["num_layers",
      "rnn_residual_connections",
      "batch_size",
      "learning_rate_schedule",
      "optimized_metric",
      "monitor_bleu",
      "seed"])
])
def test_tutorial_train_args(test_params, expected_params, expected_params_present):
    _test_args_subset(test_params, expected_params, expected_params_present, arguments.add_train_cli_args)


@pytest.mark.parametrize("test_params, expected_params, expected_params_present", [
    # seqcopy tutorial
    ('-m seqcopy_model '
     '--use-cpu',
     dict(models=["seqcopy_model"],
          use_cpu=True),
     []),
    # WMT tutorial
    ('-m wmt_model wmt_model_seed2 '
     '--use-cpu '
     '--output-type align_plot',
     dict(models=["wmt_model", "wmt_model_seed2"],
          use_cpu=True,
          output_type="align_plot"),
     # Other parameters mentioned in the WMT tutorial
     ["beam_size",
      "softmax_temperature",
      "length_penalty_alpha"]),
])
def test_tutorial_translate_args(test_params, expected_params, expected_params_present):
    _test_args_subset(test_params, expected_params, expected_params_present, arguments.add_translate_cli_args)


@pytest.mark.parametrize("test_params, expected_params, expected_params_present", [
    # WMT tutorial
    ('-o wmt_model_avg/param.best wmt_model',
     dict(inputs=["wmt_model"],
          output="wmt_model_avg/param.best"),
     []),
])
def test_tutorial_averaging_args(test_params, expected_params, expected_params_present):
    _test_args_subset(test_params, expected_params, expected_params_present, arguments.add_average_args)


def _create_argument_values_that_must_be_files(params):
    """
    Loop over test_params and create temporary files for training/validation sources/targets.
    """

    def grouper(iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    params = params.split()
    regular_files_params = {'-vs', '-vt', '-t', '-s', '--source', '--target', '--validation-source', '--validation-target'}
    to_unlink = set()
    for arg, val in grouper(params, 2):
        if arg in regular_files_params and not os.path.isfile(val):
            to_unlink.add((val, open(val, 'w')))
    return to_unlink


def _delete_argument_values_that_must_be_files(to_unlink):
    """
    Close and delete previously created files.
    """
    for name, f in to_unlink:
        f.close()
        os.unlink(name)


def _test_args(test_params, expected_params, args_func):
    test_parser = argparse.ArgumentParser()
    args_func(test_parser)
    created = _create_argument_values_that_must_be_files(test_params)
    parsed_params = test_parser.parse_args(test_params.split())
    _delete_argument_values_that_must_be_files(created)
    assert dict(vars(parsed_params)) == expected_params


def _test_args_subset(test_params, expected_params, expected_params_present, args_func):
    """
    Only checks the subset of the parameters given in `expected_params`.

    :param test_params: A string of test parameters.
    :param expected_params: A dict of parameters to test for the exact value.
    :param expected_params_present: A dict of parameters to test for presence.
    :param args_func: The function correctly setting up the parameters for ArgumentParser.
    """
    test_parser = argparse.ArgumentParser()
    args_func(test_parser)
    created = _create_argument_values_that_must_be_files(test_params)
    parsed_params = dict(vars(test_parser.parse_args(test_params.split())))
    _delete_argument_values_that_must_be_files(created)
    parsed_params_subset = {k: v for k, v in parsed_params.items() if k in expected_params}
    assert parsed_params_subset == expected_params
    for expected_param_present in expected_params_present:
        assert expected_param_present in parsed_params, "Expected param %s to be present." % expected_param_present

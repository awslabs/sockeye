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

"""
Defines commandline arguments for the main CLIs with reasonable defaults.
"""
import argparse
from typing import Callable

import sockeye.constants as C


def int_greater_or_equal(threshold: int) -> Callable:
    """
    Returns a method that can be used in argument parsing to check that the argument is greater or equal to `threshold`.

    :param threshold: The threshold that we assume the cli argument value is greater or equal to.
    :return: A method that can be used as a type in argparse.
    """

    def check_greater_equal(value_to_check):
        value_to_check = int(value_to_check)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError("must be greater or equal to %d." % threshold)
        return value_to_check

    return check_greater_equal


def add_io_args(params):
    data_params = params.add_argument_group("Data & I/O")

    data_params.add_argument('--source', '-s',
                             required=True,
                             help='Source side of parallel training data.')
    data_params.add_argument('--target', '-t',
                             required=True,
                             help='Target side of parallel training data.')

    data_params.add_argument('--validation-source', '-vs',
                             required=True,
                             help='Source side of validation data.')
    data_params.add_argument('--validation-target', '-vt',
                             required=True,
                             help='Target side of validation data.')

    data_params.add_argument('--output', '-o',
                             required=True,
                             help='Folder where model & training results are written to.')

    data_params.add_argument('--source-vocab',
                             required=False,
                             default=None,
                             help='Existing source vocabulary (JSON)')
    data_params.add_argument('--target-vocab',
                             required=False,
                             default=None,
                             help='Existing target vocabulary (JSON)')

    data_params.add_argument('--use-tensorboard',
                             action='store_true',
                             help='Track metrics through tensorboard. Requires installed tensorboard.')

    data_params.add_argument('--quiet', '-q',
                             default=False,
                             action="store_true",
                             help='Suppress console logging.')
    return params


def add_device_args(params):
    device_params = params.add_argument_group("Device parameters")

    device_params.add_argument('--device-ids', default=[-1],
                               help='List of GPU ids to use. Default: %(default)s. '
                                    'Use -1 to automatically acquire a GPU through a file locking mechanism. '
                                    '(Note that this assumes GPU processes are using automatic sockeye GPU ids).',
                               nargs='+', type=int)
    device_params.add_argument('--use-cpu',
                               action='store_true',
                               help='Use CPU device instead of GPU.')
    return params


def add_model_parameters(params):
    model_params = params.add_argument_group("ModelConfig")

    model_params.add_argument('--params', '-p',
                              type=str,
                              default=None,
                              help='Initialize model parameters from file. Overrides random initializations.')

    model_params.add_argument('--num-words',
                              type=int_greater_or_equal(0),
                              default=50000,
                              help='Maximum vocabulary size. Default: %(default)s.')
    model_params.add_argument('--word-min-count',
                              type=int_greater_or_equal(1),
                              default=1,
                              help='Minimum frequency of words to be included in vocabularies. Default: %(default)s.')

    model_params.add_argument('--rnn-num-layers',
                              type=int_greater_or_equal(1),
                              default=1,
                              help='Number of layers for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-cell-type',
                              choices=[C.LSTM_TYPE, C.GRU_TYPE],
                              default=C.LSTM_TYPE,
                              help='RNN cell type for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-num-hidden',
                              type=int_greater_or_equal(1),
                              default=1024,
                              help='Number of RNN hidden units for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-residual-connections',
                              action="store_true",
                              default=False,
                              help="Add residual connections to stacked RNNs if --rnn-num-layers > 3. "
                                   "(see Wu ETAL'16). Default: %(default)s.")

    model_params.add_argument('--num-embed',
                              type=int_greater_or_equal(1),
                              default=512,
                              help='Embedding size for source and target tokens. Default: %(default)s.')
    model_params.add_argument('--num-embed-source',
                              type=int_greater_or_equal(1),
                              default=None,
                              help='Embedding size for source tokens. Overrides --num-embed. Default: %(default)s')
    model_params.add_argument('--num-embed-target',
                              type=int_greater_or_equal(1),
                              default=None,
                              help='Embedding size for target tokens. Overrides --num-embed. Default: %(default)s')

    model_params.add_argument('--attention-type',
                              choices=["bilinear", "dot", "fixed", "location", "mlp", "coverage"],
                              default="mlp",
                              help='Attention model. Choices: {%(choices)s}. '
                                   'Default: %(default)s.')
    model_params.add_argument('--attention-num-hidden',
                              default=None,
                              type=int,
                              help='Number of hidden units for attention layers. Default: equal to --rnn-num-hidden.')

    model_params.add_argument('--attention-coverage-type',
                              choices=["tanh", "sigmoid", "relu", "softrelu", "gru", "count"],
                              default="count",
                              help="Type of model for updating coverage vectors. 'count' refers to an update method"
                                   "that accumulates attention scores. 'tanh', 'sigmoid', 'relu', 'softrelu' "
                                   "use non-linear layers with the respective activation type, and 'gru' uses a"
                                   "GRU to update the coverage vectors. Default: %(default)s.")
    model_params.add_argument('--attention-coverage-num-hidden',
                              type=int,
                              default=1,
                              help="Number of hidden units for coverage vectors. Default: %(default)s")

    model_params.add_argument('--lexical-bias',
                              default=None,
                              type=str,
                              help="Specify probabilistic lexicon for lexical biasing (Arthur ETAL'16). "
                                   "Set smoothing value epsilon by appending :<eps>")
    model_params.add_argument('--learn-lexical-bias',
                              action='store_true',
                              help='Adjust lexicon probabilities during training. Default: %(default)s')

    model_params.add_argument('--weight-tying',
                              action='store_true',
                              help='Share target embedding and output layer parameter matrix. Default: %(default)s.')

    model_params.add_argument('--max-seq-len',
                              type=int_greater_or_equal(1),
                              default=100,
                              help='Maximum sequence length in tokens. Default: %(default)s')

    model_params.add_argument('--attention-use-prev-word', action="store_true",
                              help="Feed the previous target embedding into the attention mechanism.")

    model_params.add_argument('--context-gating', action="store_true",
                              help="Enables a context gate which adaptively weighs the decoder input against the"
                                   "source context vector before each update of the decoder hidden state.")

    return params


def add_training_args(params):
    train_params = params.add_argument_group("Training parameters")

    train_params.add_argument('--batch-size', '-b',
                              type=int_greater_or_equal(1),
                              default=64,
                              help='Mini-batch size. Default: %(default)s.')
    train_params.add_argument('--fill-up',
                              type=str,
                              default='replicate',
                              help=argparse.SUPPRESS)
    train_params.add_argument('--no-bucketing',
                              action='store_true',
                              help='Disable bucketing: always unroll to the max_len.')
    train_params.add_argument('--bucket-width',
                              type=int_greater_or_equal(1),
                              default=10,
                              help='Width of buckets in tokens. Default: %(default)s.')

    train_params.add_argument('--loss',
                              default=C.CROSS_ENTROPY,
                              choices=[C.CROSS_ENTROPY, C.SMOOTHED_CROSS_ENTROPY],
                              help='Loss to optimize. Default: %(default)s.')
    train_params.add_argument('--smoothed-cross-entropy-alpha',
                              default=0.3,
                              type=float,
                              help='Smoothing value for smoothed-cross-entropy loss. Default: %(default)s.')
    train_params.add_argument('--normalize-loss',
                              default=False,
                              action="store_true",
                              help='Normalize the loss by dividing by the number of non-PAD tokens.')

    train_params.add_argument('--metrics',
                              nargs='+',
                              default=[C.PERPLEXITY],
                              choices=[C.PERPLEXITY, C.ACCURACY],
                              help='Names of metrics to track on training and validation data. Default: %(default)s.')
    train_params.add_argument('--optimized-metric',
                              default='perplexity',
                              choices=[C.PERPLEXITY, C.ACCURACY, C.BLEU],
                              help='Metric to optimize with early stopping {%(choices)s}. '
                                   'Default: %(default)s.')

    train_params.add_argument('--max-updates',
                              type=int,
                              default=-1,
                              help='Maximum number of updates/batches to process. -1 for infinite. '
                                   'Default: %(default)s.')
    train_params.add_argument('--checkpoint-frequency',
                              type=int_greater_or_equal(1),
                              default=1000,
                              help='Checkpoint and evaluate every x updates/batches. Default: %(default)s.')
    train_params.add_argument('--max-num-checkpoint-not-improved',
                              type=int,
                              default=8,
                              help='Maximum number of checkpoints the model is allowed to not improve in '
                                   '<optimized-metric> on validation data before training is stopped. '
                                   'Default: %(default)s')

    train_params.add_argument('--dropout',
                              type=float,
                              default=0.,
                              help='Dropout probability for source embedding and source and target RNNs. '
                                   'Default: %(default)s.')

    train_params.add_argument('--optimizer',
                              default='adam',
                              choices=['adam', 'sgd', 'rmsprop'],
                              help='SGD update rule. Default: %(default)s.')
    train_params.add_argument('--initial-learning-rate',
                              type=float,
                              default=0.0003,
                              help='Initial learning rate. Default: %(default)s.')
    train_params.add_argument('--weight-decay',
                              type=float,
                              default=0.0,
                              help='Weight decay constant. Default: %(default)s.')
    train_params.add_argument('--momentum',
                              type=float,
                              default=None,
                              help='Momentum constant. Default: %(default)s.')
    train_params.add_argument('--clip-gradient',
                              type=float,
                              default=1.0,
                              help='Clip absolute gradients values greater than this value. '
                                   'Set to negative to disable. Default: %(default)s.')

    train_params.add_argument('--learning-rate-scheduler-type',
                              default='plateau-reduce',
                              choices=["fixed-rate-inv-sqrt-t", "fixed-rate-inv-t", "plateau-reduce"],
                              help='Learning rate scheduler type. Default: %(default)s.')
    train_params.add_argument('--learning-rate-reduce-factor',
                              type=float,
                              default=0.5,
                              help="Factor to multiply learning rate with "
                                   "(for 'plateau-reduce' learning rate scheduler). Default: %(default)s.")
    train_params.add_argument('--learning-rate-reduce-num-not-improved',
                              type=int,
                              default=3,
                              help="For 'plateau-reduce' learning rate scheduler. Adjust learning rate "
                                   "if <optimized-metric> did not improve for x checkpoints. Default: %(default)s.")
    train_params.add_argument('--learning-rate-half-life',
                              type=float,
                              default=10,
                              help="Half-life of learning rate in checkpoints. For 'fixed-rate-*' "
                                   "learning rate schedulers. Default: 10.")

    train_params.add_argument('--use-fused-rnn',
                              default=False,
                              action="store_true",
                              help='Use FusedRNNCell in encoder (requires GPU device). Speeds up training.')

    train_params.add_argument('--rnn-forget-bias',
                              default=0.0,
                              type=float,
                              help='Initial value of RNN forget biases.')
    train_params.add_argument('--rnn-h2h-init', type=str, default=C.RNN_INIT_ORTHOGONAL,
                              choices=[C.RNN_INIT_ORTHOGONAL, C.RNN_INIT_ORTHOGONAL_STACKED],
                              help="Initialization method for RNN parameters. Default: %(default)s.")

    train_params.add_argument('--monitor-bleu',
                              default=0,
                              type=int,
                              help='x>0: sample and decode x sentences from validation data and monitor BLEU score. '
                                   'x==-1: use full validation data. Default: %(default)s.')

    train_params.add_argument('--seed',
                              type=int,
                              default=13,
                              help='Random seed. Default: %(default)s.')
    return params


def add_inference_args(params):
    decode_params = params.add_argument_group("Inference parameters")
    decode_params.add_argument('--models', '-m',
                               required=True,
                               nargs='+',
                               help='Model folder(s). Use multiple for ensemble decoding. '
                                    'Model determines config, best parameters and vocab files.')
    decode_params.add_argument('--checkpoints', '-c',
                               default=None,
                               type=int,
                               nargs='+',
                               help='If not given, chooses best checkpoints for model(s). '
                                    'If specified, must have the same length as --models and be integer')

    decode_params.add_argument('--beam-size', '-b',
                               type=int,
                               default=5,
                               help='Size of the beam. Default: %(default)s.')
    decode_params.add_argument('--ensemble-mode',
                               type=str,
                               default='linear',
                               choices=['linear', 'log_linear'],
                               help='Ensemble mode: [linear, log-linear]. Default: %(default)s.')
    decode_params.add_argument('--max-input-len', '-n',
                               type=int,
                               default=None,
                               help='Maximum sequence length. Default: value from model(s).')
    decode_params.add_argument('--softmax-temperature',
                               type=float,
                               default=None,
                               help='Controls peakiness of model predictions. Values < 1.0 produce '
                                    'peaked predictions, values > 1.0 produce smoothed distributions.')

    decode_params.add_argument('--output-type',
                               default='translation',
                               choices=["translation", "translation_with_alignments", "align_plot", "align_text"],
                               help='Output type. Choices: [translation, translation_with_alignments, '
                                    'align_plot, align_text]. Default: %(default)s.')
    decode_params.add_argument('--align-plot-prefix',
                               default="align",
                               help='Filename prefix for generated alignment visualization. Default: %(default)s')
    decode_params.add_argument('--sure-align-threshold',
                               default=0.9,
                               type=float,
                               help='Threshold to consider a soft alignment a sure alignment. Default: %(default)s')
    return params

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
from typing import Callable, Optional

from sockeye.lr_scheduler import LearningRateSchedulerFixedStep
from . import constants as C


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


def learning_schedule() -> Callable:
    """
    Returns a method that can be used in argument parsing to check that the argument is a valid learning rate schedule
    string.

    :return: A method that can be used as a type in argparse.
    """

    def parse(schedule_str):
        try:
            schedule = LearningRateSchedulerFixedStep.parse_schedule_str(schedule_str)
        except ValueError:
            raise argparse.ArgumentTypeError("Learning rate schedule string should have form rate1:num_updates1[,rate2:num_updates2,...]")
        return schedule

    return parse


def multiple_values(num_values: int = 0,
                    greater_or_equal: Optional[float] = None,
                    data_type: Callable = int) -> Callable:
    """
    Returns a method to be used in argument parsing to parse a string of the form "<val>:<val>[:<val>...]" into
    a tuple of values of type data_type.

    :param num_values: Optional number of ints required.
    :param greater_or_equal: Optional constraint that all values should be greater or equal to this value.
    :param data_type: Type of values. Default: int.
    :return: Method for parsing.
    """

    def parse(value_to_check):
        if ':' in value_to_check:
            expected_num_separators = num_values - 1 if num_values else 0
            if expected_num_separators > 0 and (value_to_check.count(':') != expected_num_separators):
                raise argparse.ArgumentTypeError("Expected either a single value or %d values separated by %s" %
                                                 (num_values, C.ARG_SEPARATOR))
            values = tuple(map(data_type, value_to_check.split(C.ARG_SEPARATOR, num_values - 1)))
        else:
            values = tuple([data_type(value_to_check)] * num_values)
        if greater_or_equal is not None:
            if any((value < greater_or_equal for value in values)):
                raise argparse.ArgumentTypeError("Must provide value greater or equal to %d" % greater_or_equal)
        return values

    return parse


def add_average_args(params):
    average_params = params.add_argument_group("Averaging")
    average_params.add_argument(
        "inputs",
        metavar="INPUT",
        type=str,
        nargs="+",
        help="either a single model directory (automatic checkpoint selection) "
             "or multiple .params files (manual checkpoint selection)")
    average_params.add_argument(
        "--metric",
        help="Name of the metric to choose n-best checkpoints from. Default: %(default)s.",
        default=C.PERPLEXITY,
        choices=C.METRICS)
    average_params.add_argument(
        "-n",
        type=int,
        default=4,
        help="number of checkpoints to find. Default: %(default)s.")
    average_params.add_argument(
        "--output", "-o", required=True, type=str, help="File to write averaged parameters to.")
    average_params.add_argument(
        "--strategy",
        choices=["best", "last", "lifespan"],
        default="best",
        help="selection method. Default: %(default)s.")


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
    data_params.add_argument('--overwrite-output',
                             action='store_true',
                             help='Overwrite output folder if it exists.')

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

    data_params.add_argument('--monitor-pattern',
                             default=None,
                             type=str,
                             help="Pattern to match outputs/weights/gradients to monitor. '.*' monitors everything. "
                                  "Default: %(default)s.")

    data_params.add_argument('--monitor-stat-func',
                             default=C.STAT_FUNC_DEFAULT,
                             choices=list(C.MONITOR_STAT_FUNCS.keys()),
                             help="Statistics function to run on monitored outputs/weights/gradients. "
                                  "Default: %(default)s.")

    data_params.add_argument('--quiet', '-q',
                             default=False,
                             action="store_true",
                             help='Suppress console logging.')


def add_device_args(params):
    device_params = params.add_argument_group("Device parameters")

    device_params.add_argument('--device-ids', default=[-1],
                               help='List or number of GPUs ids to use. Default: %(default)s. '
                                    'Use negative numbers to automatically acquire a certain number of GPUs, e.g. -5 '
                                    'will find 5 free GPUs. '
                                    'Use positive numbers to acquire a specific GPU id on this host. '
                                    '(Note that automatic acquisition of GPUs assumes that all GPU processes on '
                                    'this host are using automatic sockeye GPU acquisition).',
                               nargs='+', type=int)
    device_params.add_argument('--use-cpu',
                               action='store_true',
                               help='Use CPU device instead of GPU.')
    device_params.add_argument('--disable-device-locking',
                               action='store_true',
                               help='Just use the specified device ids without locking.')
    device_params.add_argument('--lock-dir',
                               default="/tmp",
                               help='When acquiring a GPU we do file based locking so that only one Sockeye process '
                                    'can run on the a GPU. This is the folder in which we store the file '
                                    'locks. For locking to work correctly it is assumed all processes use the same '
                                    'lock directory. The only requirement for the directory are file '
                                    'write permissions.')


def add_model_parameters(params):
    model_params = params.add_argument_group("ModelConfig")

    model_params.add_argument('--params', '-p',
                              type=str,
                              default=None,
                              help='Initialize model parameters from file. Overrides random initializations.')

    model_params.add_argument('--num-words',
                              type=multiple_values(num_values=2, greater_or_equal=0),
                              default=(50000, 50000),
                              help='Maximum vocabulary size. Use "x:x" to specify separate values for src&tgt. '
                                   'Default: %(default)s.')
    model_params.add_argument('--word-min-count',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(1, 1),
                              help='Minimum frequency of words to be included in vocabularies. Default: %(default)s.')

    model_params.add_argument('--encoder',
                              choices=C.ENCODERS,
                              default=C.RNN_NAME,
                              help="Type of encoder. Default: %(default)s.")
    model_params.add_argument('--decoder',
                              choices=C.DECODERS,
                              default=C.RNN_NAME,
                              help="Type of encoder. Default: %(default)s.")

    model_params.add_argument('--num-layers',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(1, 1),
                              help='Number of layers for encoder & decoder. '
                                   'Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')

    model_params.add_argument('--conv-embed-output-dim',
                              type=int_greater_or_equal(1),
                              default=None,
                              help="Project segment embeddings to this size for ConvolutionalEmbeddingEncoder. Omit to"
                                   " avoid projection, leaving segment embeddings total size of all filters. Default:"
                                   " %(default)s.")
    model_params.add_argument('--conv-embed-max-filter-width',
                              type=int_greater_or_equal(1),
                              default=8,
                              help="Maximum filter width for ConvolutionalEmbeddingEncoder. Default: %(default)s.")
    model_params.add_argument('--conv-embed-num-filters',
                              type=multiple_values(greater_or_equal=1),
                              default=(200, 200, 250, 250, 300, 300, 300, 300),
                              help="List of number of filters of each width 1..max for ConvolutionalEmbeddingEncoder. "
                                   "Default: %(default)s.")
    model_params.add_argument('--conv-embed-pool-stride',
                              type=int_greater_or_equal(1),
                              default=5,
                              help="Pooling stride for ConvolutionalEmbeddingEncoder. Default: %(default)s.")
    model_params.add_argument('--conv-embed-num-highway-layers',
                              type=int_greater_or_equal(0),
                              default=4,
                              help="Number of highway layers for ConvolutionalEmbeddingEncoder. Default: %(default)s.")
    model_params.add_argument('--conv-embed-add-positional-encodings',
                              action='store_true',
                              default=False,
                              help="Add positional encodings to final segment embeddings for"
                                   " ConvolutionalEmbeddingEncoder. Default: %(default)s.")

    # rnn arguments
    model_params.add_argument('--rnn-cell-type',
                              choices=C.CELL_TYPES,
                              default=C.LSTM_TYPE,
                              help='RNN cell type for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-num-hidden',
                              type=int_greater_or_equal(1),
                              default=1024,
                              help='Number of RNN hidden units for encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--rnn-encoder-reverse-input',
                              action='store_true',
                              help='Reverse input sequence for RNN encoder. Default: %(default)s.')
    model_params.add_argument('--rnn-decoder-zero-init',
                              action='store_true',
                              help='Initialize decoder RNN states with zeros instead from last & highest encoder '
                                   'RNN state. Default: %(default)s.')
    model_params.add_argument('--rnn-residual-connections',
                              action="store_true",
                              default=False,
                              help="Add residual connections to stacked RNNs. (see Wu ETAL'16). Default: %(default)s.")
    model_params.add_argument('--rnn-first-residual-layer',
                              type=int_greater_or_equal(2),
                              default=2,
                              help='First RNN layer to have a residual connection. Default: %(default)s.')
    model_params.add_argument('--rnn-context-gating', action="store_true",
                              help="Enables a context gate which adaptively weighs the RNN decoder input against the "
                                   "source context vector before each update of the decoder hidden state.")

    # transformer arguments
    model_params.add_argument('--transformer-model-size',
                              type=int_greater_or_equal(1),
                              default=512,
                              help='Size of all layers and embeddings when using transformer. Default: %(default)s.')
    model_params.add_argument('--transformer-attention-heads',
                              type=int_greater_or_equal(1),
                              default=8,
                              help='Number of heads for all self-attention when using transformer layers. '
                                   'Default: %(default)s.')
    model_params.add_argument('--transformer-feed-forward-num-hidden',
                              type=int_greater_or_equal(1),
                              default=2048,
                              help='Number of hidden units in feed forward layers when using transformer. '
                                   'Default: %(default)s.')
    model_params.add_argument('--transformer-no-positional-encodings',
                              action='store_true',
                              help='Do not use positional encodings.')

    # embedding arguments
    model_params.add_argument('--num-embed',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(512, 512),
                              help='Embedding size for source and target tokens. '
                                   'Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')

    # attention arguments
    model_params.add_argument('--attention-type',
                              choices=C.ATT_TYPES,
                              default=C.ATT_MLP,
                              help='Attention model for RNN decoders. Choices: {%(choices)s}. '
                                   'Default: %(default)s.')
    model_params.add_argument('--attention-num-hidden',
                              default=None,
                              type=int,
                              help='Number of hidden units for attention layers. Default: equal to --rnn-num-hidden.')
    model_params.add_argument('--attention-use-prev-word', action="store_true",
                              help="Feed the previous target embedding into the attention mechanism.")

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
                              help="Number of hidden units for coverage vectors. Default: %(default)s.")
    model_params.add_argument('--attention-mhdot-heads',
                              type=int, default=None,
                              help='Number of heads for Multi-head dot attention. Default: %(default)s.')

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
                              help='Turn on weight tying. The type of weight sharing is determined through '
                                   '--weight-tying-type. Default: %(default)s.')
    model_params.add_argument('--weight-tying-type',
                              default=C.WEIGHT_TYING_TRG_SOFTMAX,
                              choices=[C.WEIGHT_TYING_SRC_TRG_SOFTMAX,
                                       C.WEIGHT_TYING_SRC_TRG,
                                       C.WEIGHT_TYING_TRG_SOFTMAX],
                              help='The type of weight tying. source embeddings=src, target embeddings=trg, '
                                   'target softmax weight matrix=softmax. Default: %(default)s.')

    model_params.add_argument('--max-seq-len',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(100, 100),
                              help='Maximum sequence length in tokens. '
                                   'Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')

    model_params.add_argument('--layer-normalization', action="store_true",
                              help="Adds layer normalization before non-linear activations. "
                                   "This includes MLP attention, RNN decoder state initialization, "
                                   "RNN decoder hidden state, transformer layers."
                                   "It does not normalize RNN cell activations "
                                   "(this can be done using the '%s' or '%s' rnn-cell-type." % (C.LNLSTM_TYPE,
                                                                                                C.LNGLSTM_TYPE))


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
                              help='If turned on we normalize the loss by dividing by the number of non-PAD tokens.'
                                   'If turned off the loss is only normalized by the number of sentences in a batch.')

    train_params.add_argument('--metrics',
                              nargs='+',
                              default=[C.PERPLEXITY],
                              choices=[C.PERPLEXITY, C.ACCURACY],
                              help='Names of metrics to track on training and validation data. Default: %(default)s.')
    train_params.add_argument('--optimized-metric',
                              default=C.PERPLEXITY,
                              choices=C.METRICS,
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
    train_params.add_argument('--min-num-epochs',
                              type=int,
                              default=0,
                              help='Minimum number of epochs (passes through the training data) '
                                   'before fitting is stopped. Default: %(default)s.')

    train_params.add_argument('--embed-dropout',
                              type=multiple_values(2, data_type=float),
                              default=(.0, .0),
                              help='Dropout probability for source & target embeddings. Use <val>:<val> to specify '
                                   'separate values. Default: %(default)s.')
    train_params.add_argument('--rnn-dropout',
                              type=multiple_values(2, data_type=float),
                              default=(.0, .0),
                              help='RNN variational dropout probability for encoder & decoder RNNs.'
                                   'Use <val>:<val> to specify separate values. Default: %(default)s.')

    train_params.add_argument('--rnn-decoder-hidden-dropout',
                              type=float,
                              default=.0,
                              help='Dropout probability for hidden state that combines the context with the '
                                   'RNN hidden state in the decoder. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-attention',
                              type=float,
                              default=0.,
                              help='Dropout probability for multi-head attention. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-relu',
                              type=float,
                              default=0.,
                              help='Dropout probability before relu in feed-forward block. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-residual',
                              type=float,
                              default=0.,
                              help='Dropout probability for residual connections. Default: %(default)s.')
    train_params.add_argument('--conv-embed-dropout',
                              type=float,
                              default=.0,
                              help="Dropout probability for ConvolutionalEmbeddingEncoder. Default: %(default)s.")

    train_params.add_argument('--optimizer',
                              default='adam',
                              choices=['adam', 'sgd', 'rmsprop'],
                              help='SGD update rule. Default: %(default)s.')
    train_params.add_argument('--weight-init',
                              type=str,
                              default=C.INIT_XAVIER,
                              choices=C.INIT_TYPES,
                              help='Type of weight initialization. Default: %(default)s.')
    train_params.add_argument('--weight-init-scale',
                              type=float,
                              default=0.04,
                              help='Weight initialization scale (currently only applies to uniform initialization). '
                                   'Default: %(default)s.')
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
                              default=C.LR_SCHEDULER_PLATEAU_REDUCE,
                              choices=C.LR_SCHEDULERS,
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
    train_params.add_argument('--learning-rate-schedule',
                              type=learning_schedule(),
                              default=None,
                              help="For 'fixed-step' scheduler. Fully specified learning schedule in the form"
                              " rate1:num_updates1[,rate2:num_updates2,...]. Overrides all other args related to"
                              " learning rate and stopping conditions. Default: %(default)s.")
    train_params.add_argument('--learning-rate-half-life',
                              type=float,
                              default=10,
                              help="Half-life of learning rate in checkpoints. For 'fixed-rate-*' "
                                   "learning rate schedulers. Default: %(default)s.")
    train_params.add_argument('--learning-rate-warmup',
                              type=int,
                              default=0,
                              help="Number of warmup steps. If set to x, linearly increases learning rate from 10%% "
                                   "to 100%% of the initial learning rate. Default: %(default)s.")

    train_params.add_argument('--use-fused-rnn',
                              default=False,
                              action="store_true",
                              help='Use FusedRNNCell in encoder (requires GPU device). Speeds up training.')

    train_params.add_argument('--rnn-forget-bias',
                              default=0.0,
                              type=float,
                              help='Initial value of RNN forget biases.')
    train_params.add_argument('--rnn-h2h-init', type=str, default=C.RNN_INIT_ORTHOGONAL,
                              choices=[C.RNN_INIT_ORTHOGONAL, C.RNN_INIT_ORTHOGONAL_STACKED, C.RNN_INIT_DEFAULT],
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

    train_params.add_argument('--keep-last-params',
                              type=int,
                              default=-1,
                              help='Keep only the last n params files, use -1 to keep all files. Default: %(default)s')


def add_inference_args(params):
    decode_params = params.add_argument_group("Inference parameters")

    decode_params.add_argument(C.INFERENCE_ARG_INPUT_LONG, C.INFERENCE_ARG_INPUT_SHORT,
                               default=None,
                               help='Input file to translate. One sentence per line. '
                                    'If not given, will read from stdin.')

    decode_params.add_argument(C.INFERENCE_ARG_OUTPUT_LONG, C.INFERENCE_ARG_OUTPUT_SHORT,
                               default=None,
                               help='Output file to write translations to. '
                                    'If not given, will write to stdout.')

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
                               type=int_greater_or_equal(1),
                               default=5,
                               help='Size of the beam. Default: %(default)s.')
    decode_params.add_argument('--ensemble-mode',
                               type=str,
                               default='linear',
                               choices=['linear', 'log_linear'],
                               help='Ensemble mode. Default: %(default)s.')
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
                               choices=C.OUTPUT_HANDLERS,
                               help='Output type. Default: %(default)s.')
    decode_params.add_argument('--sure-align-threshold',
                               default=0.9,
                               type=float,
                               help='Threshold to consider a soft alignment a sure alignment. Default: %(default)s')
    decode_params.add_argument('--length-penalty-alpha',
                               default=1.0,
                               type=float,
                               help='Alpha factor for the length penalty used in beam search: '
                                    '(beta + len(Y))**alpha/(beta + 1)**alpha. A value of 0.0 will therefore turn off '
                                    'length normalization. Default: %(default)s')
    decode_params.add_argument('--length-penalty-beta',
                               default=0.0,
                               type=float,
                               help='Beta factor for the length penalty used in beam search: '
                                    '(beta + len(Y))**alpha/(beta + 1)**alpha. Default: %(default)s')

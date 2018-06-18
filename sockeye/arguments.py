# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import sys
import types
import yaml
from typing import Any, Callable, Dict, List, Tuple, Optional

from sockeye.lr_scheduler import LearningRateSchedulerFixedStep
from . import constants as C
from . import data_io


class ConfigArgumentParser(argparse.ArgumentParser):
    """
    Extension of argparse.ArgumentParser supporting config files.

    The option --config is added automatically and expects a YAML serialized
    dictionary, similar to the return value of parse_args(). Command line
    parameters have precendence over config file values. Usage should be
    transparent, just substitute argparse.ArgumentParser with this class.

    Extended from
    https://stackoverflow.com/questions/28579661/getting-required-option-from-namespace-in-python
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.argument_definitions = {}  # type: Dict[Tuple, Dict]
        self.argument_actions = []  # type: List[Any]
        self._overwrite_add_argument(self)
        self.add_argument("--config", help="Config file in YAML format.", type=str)
        # Note: not FileType so that we can get the path here

    def _register_argument(self, _action, *args, **kwargs):
        self.argument_definitions[args] = kwargs
        self.argument_actions.append(_action)

    def _overwrite_add_argument(self, original_object):
        def _new_add_argument(this_self, *args, **kwargs):
            action = this_self.original_add_argument(*args, **kwargs)
            this_self.config_container._register_argument(action, *args, **kwargs)

        original_object.original_add_argument = original_object.add_argument
        original_object.config_container = self
        original_object.add_argument = types.MethodType(_new_add_argument, original_object)

        return original_object

    def add_argument_group(self, *args, **kwargs):
        group = super().add_argument_group(*args, **kwargs)
        return self._overwrite_add_argument(group)

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        # Mini argument parser to find the config file
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("--config", type=regular_file())
        config_args, _ = config_parser.parse_known_args(args=args)
        initial_args = argparse.Namespace()
        if config_args.config:
            initial_args = load_args(config_args.config)
            # Remove the 'required' flag from options loaded from config file
            for action in self.argument_actions:
                if action.dest in initial_args:
                    action.required = False
        return super().parse_args(args=args, namespace=initial_args)


def save_args(args: argparse.Namespace, fname: str):
    with open(fname, 'w') as out:
        yaml.safe_dump(args.__dict__, out, default_flow_style=False)


def load_args(fname: str) -> argparse.Namespace:
    with open(fname, 'r') as inp:
        return argparse.Namespace(**yaml.safe_load(inp))


def regular_file() -> Callable:
    """
    Returns a method that can be used in argument parsing to check the argument is a regular file or a symbolic link,
    but not, e.g., a process substitution.

    :return: A method that can be used as a type in argparse.
    """

    def check_regular_file(value_to_check):
        value_to_check = str(value_to_check)
        if not os.path.isfile(value_to_check):
            raise argparse.ArgumentTypeError("must exist and be a regular file.")
        return value_to_check

    return check_regular_file


def regular_folder() -> Callable:
    """
    Returns a method that can be used in argument parsing to check the argument is a directory.

    :return: A method that can be used as a type in argparse.
    """

    def check_regular_directory(value_to_check):
        value_to_check = str(value_to_check)
        if not os.path.isdir(value_to_check):
            raise argparse.ArgumentTypeError("must be a directory.")
        return value_to_check

    return check_regular_directory


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
            raise argparse.ArgumentTypeError(
                "Learning rate schedule string should have form rate1:num_updates1[,rate2:num_updates2,...]")
        return schedule

    return parse


def simple_dict() -> Callable:
    """
    A simple dictionary format that does not require spaces or quoting.

    Supported types: bool, int, float

    :return: A method that can be used as a type in argparse.
    """

    def parse(dict_str: str):

        def _parse(value: str):
            if value == "True":
                return True
            if value == "False":
                return False
            if "." in value:
                return float(value)
            return int(value)

        _dict = dict()
        try:
            for entry in dict_str.split(","):
                key, value = entry.split(":")
                _dict[key] = _parse(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Specify argument dictionary as key1:value1,key2:value2,..."
                                             " Supported types: bool, int, float.")
        return _dict

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


def file_or_stdin() -> Callable:
    """
    Returns a file descriptor from stdin or opening a file from a given path.
    """

    def parse(path):
        if path is None or path == "-":
            return sys.stdin
        else:
            return data_io.smart_open(path)

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


def add_extract_args(params):
    extract_params = params.add_argument_group("Extracting")
    extract_params.add_argument("input",
                                metavar="INPUT",
                                type=str,
                                help="Either a model directory (using params.best) or a specific params.x file.")
    extract_params.add_argument('--names', '-n',
                                nargs='*',
                                default=[],
                                help='Names of parameters to be extracted.')
    extract_params.add_argument('--list-all', '-l',
                                action='store_true',
                                help='List names of all available parameters.')
    extract_params.add_argument('--output', '-o',
                                type=str,
                                help="File to write extracted parameters to (in .npz format).")


def add_lexicon_args(params):
    lexicon_params = params.add_argument_group("Model & Top-k")
    lexicon_params.add_argument("--model", "-m", required=True,
                                help="Model directory containing source and target vocabularies.")
    lexicon_params.add_argument("-k", type=int, default=200,
                                help="Number of target translations to keep per source. Default: %(default)s.")


def add_lexicon_create_args(params):
    lexicon_params = params.add_argument_group("I/O")
    lexicon_params.add_argument("--input", "-i", required=True,
                                help="Probabilistic lexicon (fast_align format) to build top-k lexicon from.")
    lexicon_params.add_argument("--output", "-o", required=True, help="File name to write top-k lexicon to.")


def add_lexicon_inspect_args(params):
    lexicon_params = params.add_argument_group("Lexicon to inspect")
    lexicon_params.add_argument("--lexicon", "-l", required=True, help="File name of top-k lexicon to inspect.")


def add_logging_args(params):
    logging_params = params.add_argument_group("Logging")
    logging_params.add_argument('--quiet', '-q',
                                default=False,
                                action="store_true",
                                help='Suppress console logging.')


def add_training_data_args(params, required=False):
    params.add_argument(C.TRAINING_ARG_SOURCE, '-s',
                        required=required,
                        type=regular_file(),
                        help='Source side of parallel training data.')
    params.add_argument('--source-factors', '-sf',
                        required=False,
                        nargs='+',
                        type=regular_file(),
                        default=[],
                        help='File(s) containing additional token-parallel source side factors. Default: %(default)s.')
    params.add_argument(C.TRAINING_ARG_TARGET, '-t',
                        required=required,
                        type=regular_file(),
                        help='Target side of parallel training data.')


def add_validation_data_params(params):
    params.add_argument('--validation-source', '-vs',
                        required=True,
                        type=regular_file(),
                        help='Source side of validation data.')
    params.add_argument('--validation-source-factors', '-vsf',
                        required=False,
                        nargs='+',
                        type=regular_file(),
                        default=[],
                        help='File(s) containing additional token-parallel validation source side factors. '
                             'Default: %(default)s.')
    params.add_argument('--validation-target', '-vt',
                        required=True,
                        type=regular_file(),
                        help='Target side of validation data.')


def add_prepared_data_args(params):
    params.add_argument(C.TRAINING_ARG_PREPARED_DATA, '-d',
                        type=regular_folder(),
                        help='Prepared training data directory created through python -m sockeye.prepare_data.')


def add_monitoring_args(params):
    params.add_argument('--monitor-pattern',
                        default=None,
                        type=str,
                        help="Pattern to match outputs/weights/gradients to monitor. '.*' monitors everything. "
                             "Default: %(default)s.")

    params.add_argument('--monitor-stat-func',
                        default=C.STAT_FUNC_DEFAULT,
                        choices=list(C.MONITOR_STAT_FUNCS.keys()),
                        help="Statistics function to run on monitored outputs/weights/gradients. "
                             "Default: %(default)s.")


def add_training_output_args(params):
    params.add_argument('--output', '-o',
                        required=True,
                        help='Folder where model & training results are written to.')
    params.add_argument('--overwrite-output',
                        action='store_true',
                        help='Delete all contents of the model directory if it already exists.')


def add_training_io_args(params):
    params = params.add_argument_group("Data & I/O")

    # Unfortunately we must set --source/--target to not required as we either accept these parameters
    # or --prepared-data which can not easily be encoded in argparse.
    add_training_data_args(params, required=False)
    add_prepared_data_args(params)
    add_validation_data_params(params)
    add_bucketing_args(params)
    add_vocab_args(params)
    add_training_output_args(params)
    add_monitoring_args(params)


def add_bucketing_args(params):
    params.add_argument('--no-bucketing',
                        action='store_true',
                        help='Disable bucketing: always unroll the graph to --max-seq-len. Default: %(default)s.')

    params.add_argument('--bucket-width',
                        type=int_greater_or_equal(1),
                        default=10,
                        help='Width of buckets in tokens. Default: %(default)s.')

    params.add_argument('--max-seq-len',
                        type=multiple_values(num_values=2, greater_or_equal=1),
                        default=(99, 99),
                        help='Maximum sequence length in tokens.'
                             'Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')


def add_prepare_data_cli_args(params):
    params = params.add_argument_group("Data preparation.")
    add_training_data_args(params, required=True)
    add_vocab_args(params)
    add_bucketing_args(params)

    params.add_argument('--num-samples-per-shard',
                        type=int_greater_or_equal(1),
                        default=1000000,
                        help='The approximate number of samples per shard. Default: %(default)s.')

    params.add_argument('--min-num-shards',
                        default=1,
                        type=int_greater_or_equal(1),
                        help='The minimum number of shards to use, even if they would not '
                             'reach the desired number of samples per shard. Default: %(default)s.')

    params.add_argument('--seed',
                        type=int,
                        default=13,
                        help='Random seed used that makes shard assignments deterministic. Default: %(default)s.')

    params.add_argument('--output', '-o',
                        required=True,
                        help='Folder where the prepared and possibly sharded data is written to.')


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


def add_vocab_args(params):
    params.add_argument('--source-vocab',
                        required=False,
                        default=None,
                        help='Existing source vocabulary (JSON).')
    params.add_argument('--target-vocab',
                        required=False,
                        default=None,
                        help='Existing target vocabulary (JSON).')
    params.add_argument(C.VOCAB_ARG_SHARED_VOCAB,
                        action='store_true',
                        default=False,
                        help='Share source and target vocabulary. '
                             'Will be automatically turned on when using weight tying. Default: %(default)s.')
    params.add_argument('--num-words',
                        type=multiple_values(num_values=2, greater_or_equal=0),
                        default=(50000, 50000),
                        help='Maximum vocabulary size. Use "x:x" to specify separate values for src&tgt. '
                             'Default: %(default)s.')
    params.add_argument('--word-min-count',
                        type=multiple_values(num_values=2, greater_or_equal=1),
                        default=(1, 1),
                        help='Minimum frequency of words to be included in vocabularies. Default: %(default)s.')


def add_model_parameters(params):
    model_params = params.add_argument_group("ModelConfig")

    model_params.add_argument('--params', '-p',
                              type=str,
                              default=None,
                              help='Initialize model parameters from file. Overrides random initializations.')
    model_params.add_argument('--allow-missing-params',
                              action="store_true",
                              default=False,
                              help="Allow missing parameters when initializing model parameters from file. "
                                   "Default: %(default)s.")

    model_params.add_argument('--encoder',
                              choices=C.ENCODERS,
                              default=C.TRANSFORMER_TYPE,
                              help="Type of encoder. Default: %(default)s.")
    model_params.add_argument('--decoder',
                              choices=C.DECODERS,
                              default=C.TRANSFORMER_TYPE,
                              help="Type of encoder. Default: %(default)s.")

    model_params.add_argument('--num-layers',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(6, 6),
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

    # convolutional encoder/decoder arguments arguments
    model_params.add_argument('--cnn-kernel-width',
                              type=multiple_values(num_values=2, greater_or_equal=1, data_type=int),
                              default=(3, 3),
                              help='Kernel width of the convolutional encoder and decoder. Default: %(default)s.')
    model_params.add_argument('--cnn-num-hidden',
                              type=int_greater_or_equal(1),
                              default=512,
                              help='Number of hidden units for the convolutional encoder and decoder. '
                                   'Default: %(default)s.')
    model_params.add_argument('--cnn-activation-type',
                              choices=C.CNN_ACTIVATION_TYPES,
                              default=C.GLU,
                              help="Type activation to use for each convolutional layer. Default: %(default)s.")
    model_params.add_argument('--cnn-positional-embedding-type',
                              choices=C.POSITIONAL_EMBEDDING_TYPES,
                              default=C.LEARNED_POSITIONAL_EMBEDDING,
                              help='The type of positional embedding. Default: %(default)s.')
    model_params.add_argument('--cnn-project-qkv',
                              action='store_true',
                              default=False,
                              help="Optionally apply query, key and value projections to the source and target hidden "
                                   "vectors before applying the attention mechanism.")

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
    model_params.add_argument('--rnn-decoder-state-init',
                              default=C.RNN_DEC_INIT_LAST,
                              choices=C.RNN_DEC_INIT_CHOICES,
                              help='How to initialize RNN decoder states. Default: %(default)s.')
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
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(512, 512),
                              help='Number of hidden units in transformer layers. '
                                   'Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-attention-heads',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(8, 8),
                              help='Number of heads for all self-attention when using transformer layers. '
                                   'Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-feed-forward-num-hidden',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(2048, 2048),
                              help='Number of hidden units in transformers feed forward layers. '
                                   'Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-activation-type',
                              choices=C.TRANSFORMER_ACTIVATION_TYPES,
                              default=C.RELU,
                              help="Type activation to use for each feed forward layer. Default: %(default)s.")
    model_params.add_argument('--transformer-positional-embedding-type',
                              choices=C.POSITIONAL_EMBEDDING_TYPES,
                              default=C.FIXED_POSITIONAL_EMBEDDING,
                              help='The type of positional embedding. Default: %(default)s.')
    model_params.add_argument('--transformer-preprocess',
                              type=multiple_values(num_values=2, greater_or_equal=None, data_type=str),
                              default=('n', 'n'),
                              help='Transformer preprocess sequence for encoder and decoder. Supports three types of '
                                   'operations: d=dropout, r=residual connection, n=layer normalization. You can '
                                   'combine in any order, for example: "ndr". '
                                   'Leave empty to not use any of these operations. '
                                   'You can specify separate sequences for encoder and decoder by separating with ":" '
                                   'For example: n:drn '
                                   'Default: %(default)s.')
    model_params.add_argument('--transformer-postprocess',
                              type=multiple_values(num_values=2, greater_or_equal=None, data_type=str),
                              default=('dr', 'dr'),
                              help='Transformer postprocess sequence for encoder and decoder. Supports three types of '
                                   'operations: d=dropout, r=residual connection, n=layer normalization. You can '
                                   'combine in any order, for example: "ndr". '
                                   'Leave empty to not use any of these operations. '
                                   'You can specify separate sequences for encoder and decoder by separating with ":" '
                                   'For example: n:drn '
                                   'Default: %(default)s.')

    # LHUC
    # TODO: The convolutional model does not support lhuc yet
    model_params.add_argument('--lhuc',
                              nargs="+",
                              default=None,
                              choices=C.LHUC_CHOICES,
                              metavar="COMPONENT",
                              help="Use LHUC (Vilar 2018). Include an amplitude parameter to hidden units for"
                              " domain adaptation. Needs a pre-trained model. Valid values: {values}. Currently not"
                              " supported for convolutional models. Default: %(default)s.".format(
                                  values=", ".join(C.LHUC_CHOICES)))

    # embedding arguments
    model_params.add_argument('--num-embed',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(512, 512),
                              help='Embedding size for source and target tokens. '
                                   'Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')
    model_params.add_argument('--source-factors-num-embed',
                              type=int,
                              nargs='+',
                              default=[],
                              help='Embedding size for additional source factors. '
                                   'You must provide as many dimensions as '
                                   '(validation) source factor files. Default: %(default)s.')

    # attention arguments
    model_params.add_argument('--rnn-attention-type',
                              choices=C.ATT_TYPES,
                              default=C.ATT_MLP,
                              help='Attention model for RNN decoders. Choices: {%(choices)s}. '
                                   'Default: %(default)s.')
    model_params.add_argument('--rnn-attention-num-hidden',
                              default=None,
                              type=int,
                              help='Number of hidden units for attention layers. Default: equal to --rnn-num-hidden.')
    model_params.add_argument('--rnn-attention-use-prev-word', action="store_true",
                              help="Feed the previous target embedding into the attention mechanism.")

    model_params.add_argument('--rnn-scale-dot-attention',
                              action='store_true',
                              help='Optional scale before dot product. Only applicable to \'dot\' attention type. '
                                   '[Vaswani et al, 2017]')

    model_params.add_argument('--rnn-attention-coverage-type',
                              choices=["tanh", "sigmoid", "relu", "softrelu", "gru", "count"],
                              default="count",
                              help="Type of model for updating coverage vectors. 'count' refers to an update method "
                                   "that accumulates attention scores. 'tanh', 'sigmoid', 'relu', 'softrelu' "
                                   "use non-linear layers with the respective activation type, and 'gru' uses a "
                                   "GRU to update the coverage vectors. Default: %(default)s.")
    model_params.add_argument('--rnn-attention-coverage-num-hidden',
                              type=int,
                              default=1,
                              help="Number of hidden units for coverage vectors. Default: %(default)s.")
    model_params.add_argument('--rnn-attention-in-upper-layers',
                              action="store_true",
                              help="Pass the attention to the upper layers of the RNN decoder, similar "
                                   "to GNMT paper. Only applicable if more than one layer is used.")
    model_params.add_argument('--rnn-attention-mhdot-heads',
                              type=int, default=None,
                              help='Number of heads for Multi-head dot attention. Default: %(default)s.')

    model_params.add_argument('--weight-tying',
                              action='store_true',
                              help='Turn on weight tying (see arxiv.org/abs/1608.05859). '
                                   'The type of weight sharing is determined through '
                                   '--weight-tying-type. Default: %(default)s.')
    model_params.add_argument('--weight-tying-type',
                              default=C.WEIGHT_TYING_TRG_SOFTMAX,
                              choices=[C.WEIGHT_TYING_SRC_TRG_SOFTMAX,
                                       C.WEIGHT_TYING_SRC_TRG,
                                       C.WEIGHT_TYING_TRG_SOFTMAX],
                              help='The type of weight tying. source embeddings=src, target embeddings=trg, '
                                   'target softmax weight matrix=softmax. Default: %(default)s.')

    model_params.add_argument('--layer-normalization', action="store_true",
                              help="Adds layer normalization before non-linear activations. "
                                   "This includes MLP attention, RNN decoder state initialization, "
                                   "RNN decoder hidden state, and cnn layers."
                                   "It does not normalize RNN cell activations "
                                   "(this can be done using the '%s' or '%s' rnn-cell-type." % (C.LNLSTM_TYPE,
                                                                                                C.LNGLSTM_TYPE))

    model_params.add_argument('--weight-normalization', action="store_true",
                              help="Adds weight normalization to decoder output layers "
                                   "(and all convolutional weight matrices for CNN decoders). Default: %(default)s.")


def add_training_args(params):
    train_params = params.add_argument_group("Training parameters")

    train_params.add_argument('--batch-size', '-b',
                              type=int_greater_or_equal(1),
                              default=4096,
                              help='Mini-batch size. Note that depending on the batch-type this either refers to '
                                   'words or sentences.'
                                   'Sentence: each batch contains X sentences, number of words varies. '
                                   'Word: each batch contains (approximately) X words, number of sentences varies. '
                                   'Default: %(default)s.')
    train_params.add_argument("--batch-type",
                              type=str,
                              default=C.BATCH_TYPE_WORD,
                              choices=[C.BATCH_TYPE_SENTENCE, C.BATCH_TYPE_WORD],
                              help="Sentence: each batch contains X sentences, number of words varies."
                                   "Word: each batch contains (approximately) X target words, "
                                   "number of sentences varies. Default: %(default)s.")

    train_params.add_argument('--fill-up',
                              type=str,
                              default='replicate',
                              help=argparse.SUPPRESS)

    train_params.add_argument('--loss',
                              default=C.CROSS_ENTROPY,
                              choices=[C.CROSS_ENTROPY],
                              help='Loss to optimize. Default: %(default)s.')
    train_params.add_argument('--label-smoothing',
                              default=0.1,
                              type=float,
                              help='Smoothing constant for label smoothing. Default: %(default)s.')
    train_params.add_argument('--loss-normalization-type',
                              default=C.LOSS_NORM_VALID,
                              choices=[C.LOSS_NORM_VALID, C.LOSS_NORM_BATCH],
                              help='How to normalize the loss. By default loss is normalized by the number '
                                   'of valid (non-PAD) tokens (%s).' % C.LOSS_NORM_VALID)

    train_params.add_argument('--metrics',
                              nargs='+',
                              default=[C.PERPLEXITY],
                              choices=[C.PERPLEXITY, C.ACCURACY],
                              help='Names of metrics to track on training and validation data. Default: %(default)s.')
    train_params.add_argument('--optimized-metric',
                              default=C.PERPLEXITY,
                              choices=C.METRICS,
                              help='Metric to optimize with early stopping {%(choices)s}. Default: %(default)s.')

    train_params.add_argument('--min-updates',
                              type=int,
                              default=None,
                              help='Minimum number of updates before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-updates',
                              type=int,
                              default=None,
                              help='Maximum number of updates. Default: %(default)s.')
    train_params.add_argument('--min-samples',
                              type=int,
                              default=None,
                              help='Minimum number of samples before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-samples',
                              type=int,
                              default=None,
                              help='Maximum number of samples. Default: %(default)s.')
    train_params.add_argument(C.TRAIN_ARGS_CHECKPOINT_FREQUENCY,
                              type=int_greater_or_equal(1),
                              default=4000,
                              help='Checkpoint and evaluate every x updates/batches. Default: %(default)s.')
    train_params.add_argument('--max-num-checkpoint-not-improved',
                              type=int,
                              default=32,
                              help='Maximum number of checkpoints the model is allowed to not improve in '
                                   '<optimized-metric> on validation data before training is stopped. '
                                   'Default: %(default)s.')
    train_params.add_argument('--min-num-epochs',
                              type=int,
                              default=None,
                              help='Minimum number of epochs (passes through the training data) '
                                   'before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-num-epochs',
                              type=int,
                              default=None,
                              help='Maximum number of epochs (passes through the training data) Default: %(default)s.')

    train_params.add_argument('--embed-dropout',
                              type=multiple_values(2, data_type=float),
                              default=(.0, .0),
                              help='Dropout probability for source & target embeddings. Use "x:x" to specify '
                                   'separate values. Default: %(default)s.')
    train_params.add_argument('--rnn-dropout-inputs',
                              type=multiple_values(2, data_type=float),
                              default=(.0, .0),
                              help='RNN variational dropout probability for encoder & decoder RNN inputs. (Gal, 2015)'
                                   'Use "x:x" to specify separate values. Default: %(default)s.')
    train_params.add_argument('--rnn-dropout-states',
                              type=multiple_values(2, data_type=float),
                              default=(.0, .0),
                              help='RNN variational dropout probability for encoder & decoder RNN states. (Gal, 2015)'
                                   'Use "x:x" to specify separate values. Default: %(default)s.')
    train_params.add_argument('--rnn-dropout-recurrent',
                              type=multiple_values(2, data_type=float),
                              default=(.0, .0),
                              help='Recurrent dropout without memory loss (Semeniuta, 2016) for encoder & decoder '
                                   'LSTMs. Use "x:x" to specify separate values. Default: %(default)s.')
    train_params.add_argument('--rnn-enc-last-hidden-concat-to-embedding',
                              action="store_true",
                              help='Concatenate the last hidden layer of the encoder to the input of the decoder, '
                                   'instead of the previous state of the decoder. Default: %(default)s.')

    train_params.add_argument('--rnn-decoder-hidden-dropout',
                              type=float,
                              default=.2,
                              help='Dropout probability for hidden state that combines the context with the '
                                   'RNN hidden state in the decoder. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-attention',
                              type=float,
                              default=0.1,
                              help='Dropout probability for multi-head attention. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-act',
                              type=float,
                              default=0.1,
                              help='Dropout probability before activation in feed-forward block. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-prepost',
                              type=float,
                              default=0.1,
                              help='Dropout probability for pre/postprocessing blocks. Default: %(default)s.')
    train_params.add_argument('--conv-embed-dropout',
                              type=float,
                              default=.0,
                              help="Dropout probability for ConvolutionalEmbeddingEncoder. Default: %(default)s.")
    train_params.add_argument('--cnn-hidden-dropout',
                              type=float,
                              default=.2,
                              help="Dropout probability for dropout between convolutional layers. Default: %(default)s.")

    train_params.add_argument('--optimizer',
                              default=C.OPTIMIZER_ADAM,
                              choices=C.OPTIMIZERS,
                              help='SGD update rule. Default: %(default)s.')
    train_params.add_argument('--optimizer-params',
                              type=simple_dict(),
                              default=None,
                              help='Additional optimizer params as dictionary. Format: key1:value1,key2:value2,...')

    train_params.add_argument("--kvstore",
                              type=str,
                              default=C.KVSTORE_DEVICE,
                              choices=C.KVSTORE_TYPES,
                              help="The MXNet kvstore to use. 'device' is recommended for single process training. "
                                   "Use any of 'dist_sync', 'dist_device_sync' and 'dist_async' for distributed "
                                   "training. Default: %(default)s.")
    train_params.add_argument("--gradient-compression-type",
                              type=str,
                              default=C.GRADIENT_COMPRESSION_NONE,
                              choices=C.GRADIENT_COMPRESSION_TYPES,
                              help='Type of gradient compression to use. Default: %(default)s.')
    train_params.add_argument("--gradient-compression-threshold",
                              type=float,
                              default=0.5,
                              help="Threshold for gradient compression if --gctype is '2bit'. Default: %(default)s.")

    train_params.add_argument('--weight-init',
                              type=str,
                              default=C.INIT_XAVIER,
                              choices=C.INIT_TYPES,
                              help='Type of base weight initialization. Default: %(default)s.')
    train_params.add_argument('--weight-init-scale',
                              type=float,
                              default=3.0,
                              help='Weight initialization scale. Applies to uniform (scale) and xavier (magnitude). '
                                   'Default: %(default)s.')
    train_params.add_argument('--weight-init-xavier-factor-type',
                              type=str,
                              default=C.INIT_XAVIER_FACTOR_TYPE_AVG,
                              choices=C.INIT_XAVIER_FACTOR_TYPES,
                              help='Xavier factor type. Default: %(default)s.')
    train_params.add_argument('--weight-init-xavier-rand-type',
                              type=str,
                              default=C.RAND_TYPE_UNIFORM,
                              choices=[C.RAND_TYPE_UNIFORM, C.RAND_TYPE_GAUSSIAN],
                              help='Xavier random number generator type. Default: %(default)s.')
    train_params.add_argument('--embed-weight-init',
                              type=str,
                              default=C.EMBED_INIT_DEFAULT,
                              choices=C.EMBED_INIT_TYPES,
                              help='Type of embedding matrix weight initialization. If normal, initializes embedding '
                                   'weights using a normal distribution with std=1/srqt(vocab_size). '
                                   'Default: %(default)s.')
    train_params.add_argument('--initial-learning-rate',
                              type=float,
                              default=0.0002,
                              help='Initial learning rate. Default: %(default)s.')
    train_params.add_argument('--weight-decay',
                              type=float,
                              default=0.0,
                              help='Weight decay constant. Default: %(default)s.')
    train_params.add_argument('--momentum',
                              type=float,
                              default=None,
                              help='Momentum constant. Default: %(default)s.')
    train_params.add_argument('--gradient-clipping-threshold',
                              type=float,
                              default=1.0,
                              help='Clip absolute gradients values greater than this value. '
                                   'Set to negative to disable. Default: %(default)s.')
    train_params.add_argument('--gradient-clipping-type',
                              choices=C.GRADIENT_CLIPPING_TYPES,
                              default=C.GRADIENT_CLIPPING_TYPE_NONE,
                              help='The type of gradient clipping. Default: %(default)s.')

    train_params.add_argument('--learning-rate-scheduler-type',
                              default=C.LR_SCHEDULER_PLATEAU_REDUCE,
                              choices=C.LR_SCHEDULERS,
                              help='Learning rate scheduler type. Default: %(default)s.')
    train_params.add_argument('--learning-rate-reduce-factor',
                              type=float,
                              default=0.7,
                              help="Factor to multiply learning rate with "
                                   "(for 'plateau-reduce' learning rate scheduler). Default: %(default)s.")
    train_params.add_argument('--learning-rate-reduce-num-not-improved',
                              type=int,
                              default=8,
                              help="For 'plateau-reduce' learning rate scheduler. Adjust learning rate "
                                   "if <optimized-metric> did not improve for x checkpoints. Default: %(default)s.")
    train_params.add_argument('--learning-rate-schedule',
                              type=learning_schedule(),
                              default=None,
                              help="For 'fixed-step' scheduler. Fully specified learning schedule in the form"
                                   " \"rate1:num_updates1[,rate2:num_updates2,...]\". Overrides all other args related"
                                   " to learning rate and stopping conditions. Default: %(default)s.")
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
    train_params.add_argument('--learning-rate-decay-param-reset',
                              action='store_true',
                              help='Resets model parameters to current best when learning rate is reduced due to the '
                                   'value of --learning-rate-reduce-num-not-improved. Default: %(default)s.')
    train_params.add_argument('--learning-rate-decay-optimizer-states-reset',
                              choices=C.LR_DECAY_OPT_STATES_RESET_CHOICES,
                              default=C.LR_DECAY_OPT_STATES_RESET_OFF,
                              help="Action to take on optimizer states (e.g. Adam states) when learning rate is "
                                   "reduced due to the value of --learning-rate-reduce-num-not-improved. "
                                   "Default: %(default)s.")

    train_params.add_argument('--rnn-forget-bias',
                              default=0.0,
                              type=float,
                              help='Initial value of RNN forget biases.')
    train_params.add_argument('--rnn-h2h-init', type=str, default=C.RNN_INIT_ORTHOGONAL,
                              choices=[C.RNN_INIT_ORTHOGONAL, C.RNN_INIT_ORTHOGONAL_STACKED, C.RNN_INIT_DEFAULT],
                              help="Initialization method for RNN parameters. Default: %(default)s.")

    train_params.add_argument('--fixed-param-names',
                              default=[],
                              nargs='*',
                              help="Names of parameters to fix at training time. Default: %(default)s.")

    train_params.add_argument(C.TRAIN_ARGS_MONITOR_BLEU,
                              default=500,
                              type=int,
                              help='x>0: decode x sampled sentences from validation data and '
                                   'compute evaluation metrics. x==-1: use full validation data. Default: %(default)s.')
    train_params.add_argument('--decode-and-evaluate-use-cpu',
                              action='store_true',
                              help='Use CPU for decoding validation data. Overrides --decode-and-evaluate-device-id. '
                                   'Default: %(default)s.')
    train_params.add_argument('--decode-and-evaluate-device-id',
                              default=None,
                              type=int,
                              help='Separate device for decoding validation data. '
                                   'Use a negative number to automatically acquire a GPU. '
                                   'Use a positive number to acquire a specific GPU. Default: %(default)s.')

    train_params.add_argument('--seed',
                              type=int,
                              default=13,
                              help='Random seed. Default: %(default)s.')

    train_params.add_argument('--keep-last-params',
                              type=int,
                              default=-1,
                              help='Keep only the last n params files, use -1 to keep all files. Default: %(default)s')

    train_params.add_argument('--dry-run',
                              action='store_true',
                              help="Do not perform any actual training, but print statistics about the model"
                              " and mode of operation.")


def add_train_cli_args(params):
    add_training_io_args(params)
    add_model_parameters(params)
    add_training_args(params)
    add_device_args(params)
    add_logging_args(params)


def add_translate_cli_args(params):
    add_inference_args(params)
    add_device_args(params)
    add_logging_args(params)


def add_max_output_cli_args(params):
    params.add_argument('--max-output-length',
                        type=int,
                        default=None,
                        help='Maximum number of words to generate during translation. If None, it will be computed automatically. Default: %(default)s.')


def add_inference_args(params):
    decode_params = params.add_argument_group("Inference parameters")

    decode_params.add_argument(C.INFERENCE_ARG_INPUT_LONG, C.INFERENCE_ARG_INPUT_SHORT,
                               default=None,
                               help='Input file to translate. One sentence per line. '
                                    'If not given, will read from stdin.')

    decode_params.add_argument(C.INFERENCE_ARG_INPUT_FACTORS_LONG, C.INFERENCE_ARG_INPUT_FACTORS_SHORT,
                               required=False,
                               nargs='+',
                               type=regular_file(),
                               default=None,
                               help='List of input files containing additional source factors,'
                                    'each token-parallel to the source. Default: %(default)s.')

    decode_params.add_argument('--json-input',
                               action='store_true',
                               default=False,
                               help="If given, the CLI expects string-serialized json objects as input."
                                    "Requires at least the input text field, for example: "
                                    "{'text': 'some input string'} "
                                    "Optionally, a list of factors can be provided: "
                                    "{'text': 'some input string', 'factors': ['C C C', 'X X X']}.")

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
    decode_params.add_argument('--beam-prune', '-p',
                               type=float,
                               default=0,
                               help='Pruning threshold for beam search. All hypotheses with scores not within '
                               'this amount of the best finished hypothesis are discarded (0 = off). Default: %(default)s.')
    decode_params.add_argument('--beam-search-stop',
                               choices=[C.BEAM_SEARCH_STOP_ALL, C.BEAM_SEARCH_STOP_FIRST],
                               default=C.BEAM_SEARCH_STOP_ALL,
                               help='Stopping criteria. Quit when (all) hypotheses are finished or when a finished hypothesis is in (first) position. Default: %(default)s.')
    decode_params.add_argument('--batch-size',
                               type=int_greater_or_equal(1),
                               default=1,
                               help='Batch size during decoding. Determines how many sentences are translated '
                                    'simultaneously. Default: %(default)s.')
    decode_params.add_argument('--chunk-size',
                               type=int_greater_or_equal(1),
                               default=None,
                               help='Size of the chunks to be read from input at once. The chunks are sorted and then '
                                    'split into batches. Therefore the larger the chunk size the better the grouping '
                                    'of segments of similar length and therefore the higher the increase in throughput.'
                                    ' Default: %d without batching '
                                    'and %d * batch_size with batching.' % (C.CHUNK_SIZE_NO_BATCHING,
                                                                            C.CHUNK_SIZE_PER_BATCH_SEGMENT))
    decode_params.add_argument('--ensemble-mode',
                               type=str,
                               default='linear',
                               choices=['linear', 'log_linear'],
                               help='Ensemble mode. Default: %(default)s.')
    decode_params.add_argument('--bucket-width',
                               type=int_greater_or_equal(0),
                               default=10,
                               help='Bucket width for encoder steps. 0 means no bucketing. Default: %(default)s.')
    decode_params.add_argument('--max-input-len', '-n',
                               type=int,
                               default=None,
                               help='Maximum input sequence length. Default: value from model(s).')
    decode_params.add_argument('--softmax-temperature',
                               type=float,
                               default=None,
                               help='Controls peakiness of model predictions. Values < 1.0 produce '
                                    'peaked predictions, values > 1.0 produce smoothed distributions.')
    decode_params.add_argument('--max-output-length-num-stds',
                               type=int,
                               default=C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                               help='Number of target-to-source length ratio standard deviations from training to add '
                                    'to calculate maximum output length for beam search for each sentence. '
                                    'Default: %(default)s.')
    decode_params.add_argument('--restrict-lexicon',
                               type=str,
                               default=None,
                               help="Specify top-k lexicon to restrict output vocabulary based on source. See lexicon "
                                    "module. Default: %(default)s.")
    decode_params.add_argument('--restrict-lexicon-topk',
                               type=int,
                               default=None,
                               help="Specify the number of translations to load for each source word from the lexicon "
                                    "given with --restrict-lexicon. Default: Load all entries from the lexicon.")
    decode_params.add_argument('--strip-unknown-words',
                               action='store_true',
                               default=False,
                               help='Remove any <unk> symbols from outputs. Default: %(default)s.')

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
    decode_params.add_argument('--override-dtype',
                               default=None,
                               type=str,
                               help='EXPERIMENTAL: may be changed or removed in future. Overrides training dtype of '
                                    'encoders and decoders during inference. Default: %(default)s')

def add_evaluate_args(params):
    eval_params = params.add_argument_group("Evaluate parameters")
    eval_params.add_argument('--references', '-r',
                             required=True,
                             type=str,
                             help="File with references.")
    eval_params.add_argument('--hypotheses', '-i',
                             type=file_or_stdin(),
                             default=[sys.stdin],
                             nargs='+',
                             help="File(s) with hypotheses. If none will read from stdin. Default: %(default)s.")
    eval_params.add_argument('--metrics',
                             nargs='+',
                             default=[C.BLEU, C.CHRF],
                             help='List of metrics to compute. Default: %(default)s.')
    eval_params.add_argument('--sentence', '-s',
                             action="store_true",
                             help="Show sentence-level metrics. Default: %(default)s.")
    eval_params.add_argument('--offset',
                             type=float,
                             default=0.01,
                             help="Numerical value of the offset of zero n-gram counts. Default: %(default)s.")
    eval_params.add_argument('--not-strict', '-n',
                             action="store_true",
                             help="Do not fail if number of hypotheses does not match number of references. "
                                  "Default: %(default)s.")


def add_build_vocab_args(params):
    params.add_argument('-i', '--inputs', required=True, nargs='+', help='List of text files to build vocabulary from.')
    params.add_argument('-o', '--output', required=True, type=str, help="Output filename to write vocabulary to.")
    add_vocab_args(params)


def add_init_embedding_args(params):
    params.add_argument('--weight-files', '-w', required=True, nargs='+',
                        help='List of input weight files in .npy, .npz or Sockeye parameter format.')
    params.add_argument('--vocabularies-in', '-i', required=True, nargs='+',
                        help='List of input vocabularies as token-index dictionaries in .json format.')
    params.add_argument('--vocabularies-out', '-o', required=True, nargs='+',
                        help='List of output vocabularies as token-index dictionaries in .json format.')
    params.add_argument('--names', '-n', nargs='+',
                        help='List of Sockeye parameter names for (embedding) weights. Default: %(default)s.',
                        default=[n + "weight" for n in [C.SOURCE_EMBEDDING_PREFIX, C.TARGET_EMBEDDING_PREFIX]])
    params.add_argument('--file', '-f', required=True,
                        help='File to write initialized parameters to.')
    params.add_argument('--encoding', '-c', type=str, default=C.VOCAB_ENCODING,
                        help='Open input vocabularies with specified encoding. Default: %(default)s.')

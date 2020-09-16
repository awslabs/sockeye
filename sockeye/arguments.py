# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Any, Callable, Dict, List, Tuple, Optional

import yaml

from . import constants as C
from . import data_io
from . import utils


class ConfigArgumentParser(argparse.ArgumentParser):
    """
    Extension of argparse.ArgumentParser supporting config files.

    The option --config is added automatically and expects a YAML serialized
    dictionary, similar to the return value of parse_args(). Command line
    parameters have precedence over config file values. Usage should be
    transparent, just substitute argparse.ArgumentParser with this class.

    Extended from
    https://stackoverflow.com/questions/28579661/getting-required-option-from-namespace-in-python
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.argument_definitions = {}  # type: Dict[Tuple, Dict]
        self.argument_actions = []  # type: List[Any]
        self._overwrite_add_argument(self)
        self.add_argument("--config", help="Path to CLI arguments in yaml format "
                                           "(as saved in Sockeye model directories as 'args.yaml'). "
                                           "Commandline arguments have precedence over values in this file.", type=str)
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


class StoreDeprecatedAction(argparse.Action):

    def __init__(self, option_strings, dest, deprecated_dest, nargs=None, **kwargs):
        super(StoreDeprecatedAction, self).__init__(option_strings, dest, **kwargs)
        self.deprecated_dest = deprecated_dest

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        setattr(namespace, self.deprecated_dest, value)


def save_args(args: argparse.Namespace, fname: str):
    with open(fname, 'w') as out:
        yaml.safe_dump(args.__dict__, out, default_flow_style=False)


def load_args(fname: str) -> argparse.Namespace:
    with open(fname, 'r') as inp:
        return argparse.Namespace(**yaml.safe_load(inp))


class Removed(argparse.Action):
    """
    When this argument is specified, raise an error with the argument's help
    message.  This is used to notify users when arguments are removed.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        raise RuntimeError(self.help)


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
    Returns a method that can be used in argument parsing to check that the int argument is greater or equal to `threshold`.

    :param threshold: The threshold that we assume the cli argument value is greater or equal to.
    :return: A method that can be used as a type in argparse.
    """

    def check_greater_equal(value: str):
        value_to_check = int(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError("must be greater or equal to %d." % threshold)
        return value_to_check

    return check_greater_equal


def float_greater_or_equal(threshold: float) -> Callable:
    """
    Returns a method that can be used in argument parsing to check that the float argument is greater or equal to `threshold`.

    :param threshold: The threshold that we assume the cli argument value is greater or equal to.
    :return: A method that can be used as a type in argparse.
    """

    def check_greater_equal(value: str):
        value_to_check = float(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError("must be greater or equal to %f." % threshold)
        return value_to_check

    return check_greater_equal


def bool_str() -> Callable:
    """
    Returns a method that can be used in argument parsing to check that the argument is a valid representation of
    a boolean value.

    :return: A method that can be used as a type in argparse.
    """
    def parse(value: str):
        lower_value = value.lower()
        if lower_value in ["true", "yes", "1"]:
            return True
        elif lower_value in ["false", "no", "0"]:
            return False
        else:
            raise argparse.ArgumentTypeError(
                "Invalid value for bool argument. Use true/false, yes/no or 1/0.")

    return parse


def simple_dict() -> Callable:
    """
    A simple dictionary format that does not require spaces or quoting.

    Supported types: bool, int, float

    :return: A method that can be used as a type in argparse.
    """

    def parse(dict_str: str):

        def _parse(value: str):
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
            if "." in value or "e" in value:
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
        choices=C.AVERAGE_CHOICES,
        default=C.AVERAGE_BEST,
        help="selection method. Default: %(default)s.")


def add_extract_args(params):
    extract_params = params.add_argument_group("Extracting")
    extract_params.add_argument("input",
                                metavar="INPUT",
                                type=str,
                                help="Either a model directory (using its %s) or a specific params.x file." % C.PARAMS_BEST_NAME)
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


def add_rerank_args(params):
    rerank_params = params.add_argument_group("Reranking")
    rerank_params.add_argument("--reference", "-r",
                               type=str,
                               required=True,
                               help="File where target reference translations are stored.")
    rerank_params.add_argument("--hypotheses", "-hy",
                               type=str,
                               required=True,
                               help="File with nbest translations, one nbest list per line,"
                                    "in JSON format as returned by sockeye.translate with --nbest-size x.")
    rerank_params.add_argument("--metric", "-m",
                               type=str,
                               required=False,
                               default=C.RERANK_BLEU,
                               choices=C.RERANK_METRICS,
                               help="Sentence-level metric used to compare each nbest translation to the reference."
                                    "Default: %(default)s.")
    rerank_params.add_argument("--output", "-o", default=None, help="File to write output to. Default: STDOUT.")
    rerank_params.add_argument("--output-best",
                               action="store_true",
                               help="Output only the best hypothesis from each nbest list.")
    rerank_params.add_argument("--output-reference-instead-of-blank",
                               action="store_true",
                               help="When outputting only the best hypothesis (--output-best) and the best hypothesis "
                                    "is a blank line, output the reference instead.")
    rerank_params.add_argument("--return-score",
                               action="store_true",
                               help="Returns the reranking scores as scores in output JSON objects.")


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
    logging_params.add_argument('--quiet-secondary-workers', '-qsw',
                                default=False,
                                action="store_true",
                                help='Suppress console logging for secondary workers when training with Horovod/MPI.')
    logging_params.add_argument('--no-logfile',
                                default=False,
                                action="store_true",
                                help='Suppress file logging')
    logging_params.add_argument('--loglevel',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='Log level. Default: %(default)s.')


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
    params.add_argument('--source-factors-use-source-vocab',
                        required=False,
                        nargs='+',
                        type=bool_str(),
                        default=[],
                        help='List of bools signaling wether to use the source vocabulary for the source factors. '
                        'If empty (default) each factor has its own vocabulary.')
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
                        default=8,
                        help='Width of buckets in tokens. Default: %(default)s.')

    params.add_argument('--bucket-scaling',
                        action='store_true',
                        help='Scale source/target buckets based on length ratio to reduce padding. Default: '
                             '%(default)s.')
    params.add_argument('--no-bucket-scaling',
                        action=Removed,
                        nargs=0,
                        help='Removed: The argument "--no-bucket-scaling" has been removed because this is now the '
                             'default behavior. To activate bucket scaling, use the argument "--bucket-scaling".')

    params.add_argument(C.TRAINING_ARG_MAX_SEQ_LEN,
                        type=multiple_values(num_values=2, greater_or_equal=1),
                        default=(95, 95),
                        help='Maximum sequence length in tokens, not counting BOS/EOS tokens (internal max sequence '
                             'length is X+1). Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')


def add_prepare_data_cli_args(params):
    add_training_data_args(params, required=True)
    add_vocab_args(params)
    add_bucketing_args(params)

    params.add_argument('--num-samples-per-shard',
                        type=int_greater_or_equal(1),
                        default=10000000,
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
    params.add_argument('--max-processes',
                        type=int_greater_or_equal(1),
                        default=1,
                        help='Process the shards in parallel using max-processes processes.')

    add_logging_args(params)


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
    device_params.add_argument('--omp-num-threads',
                               type=int,
                               help='Set the OMP_NUM_THREADS environment variable (CPU threads). Recommended: set to '
                                    'number of GPUs for training, number of physical CPU cores for inference. Default: '
                                    '%(default)s.')
    device_params.add_argument('--env',
                               help='List of environment variables to be set before importing MXNet. Separated by ",", '
                                    'e.g. --env=OMP_NUM_THREADS=4,MXNET_GPU_WORKER_NTHREADS=3 etc.')
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
    params.add_argument('--source-factor-vocabs',
                        required=False,
                        nargs='+',
                        type=regular_file(),
                        default=[],
                        help='Existing source factor vocabulary (-ies) (JSON).')
    params.add_argument(C.VOCAB_ARG_SHARED_VOCAB,
                        action='store_true',
                        default=False,
                        help='Share source and target vocabulary. '
                             'Will be automatically turned on when using weight tying. Default: %(default)s.')
    params.add_argument('--num-words',
                        type=multiple_values(num_values=2, greater_or_equal=0),
                        default=(0, 0),
                        help='Maximum vocabulary size. Use "x:x" to specify separate values for src&tgt. '
                             'A value of 0 indicates that the vocabulary unrestricted and determined from the data by '
                             'creating an entry for all words that occur at least --word-min-count times.'
                             'Default: %(default)s.')
    params.add_argument('--word-min-count',
                        type=multiple_values(num_values=2, greater_or_equal=1),
                        default=(1, 1),
                        help='Minimum frequency of words to be included in vocabularies. Default: %(default)s.')
    params.add_argument('--pad-vocab-to-multiple-of',
                        type=int,
                        default=None,
                        help='Pad vocabulary to a multiple of this integer. Default: %(default)s.')


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
    model_params.add_argument('--ignore-extra-params',
                              action="store_true",
                              default=False,
                              help="Allow extra parameters when initializing model parameters from file. "
                                   "Default: %(default)s.")

    model_params.add_argument('--encoder',
                              choices=C.ENCODERS,
                              default=C.TRANSFORMER_TYPE,
                              help="Type of encoder. Default: %(default)s.")
    model_params.add_argument('--decoder',
                              choices=C.DECODERS,
                              default=C.TRANSFORMER_TYPE,
                              help="Type of decoder. Default: %(default)s. "
                                   "'ssru_transformer' uses Simpler Simple Recurrent Units (Kim et al, 2019) "
                                   "as replacement for self-attention layers.")

    model_params.add_argument('--num-layers',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(6, 6),
                              help='Number of layers for encoder & decoder. '
                                   'Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')

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
                              type=multiple_values(num_values=2, greater_or_equal=None, data_type=str),
                              default=(C.RELU, C.RELU),
                              help='Type of activation to use for each feed forward layer. Use "x:x" to specify '
                                   'different values for encoder & decoder. Supported: {}. Default: '
                                   '%(default)s.'.format(' '.join(C.TRANSFORMER_ACTIVATION_TYPES)))
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

    model_params.add_argument('--lhuc',
                              nargs="+",
                              default=None,
                              choices=C.LHUC_CHOICES,
                              metavar="COMPONENT",
                              help="Use LHUC (Vilar 2018). Include an amplitude parameter to hidden units for"
                              " domain adaptation. Needs a pre-trained model. Valid values: {values}."
                              " Default: %(default)s.".format(
                                  values=", ".join(C.LHUC_CHOICES)))

    # embedding arguments
    model_params.add_argument('--num-embed',
                              type=multiple_values(num_values=2, greater_or_equal=1),
                              default=(None, None),
                              help='Embedding size for source and target tokens. '
                                   'Use "x:x" to specify separate values for src&tgt. Default: %d.' % C.DEFAULT_NUM_EMBED)
    model_params.add_argument('--source-factors-num-embed',
                              type=int,
                              nargs='+',
                              default=[],
                              help='Embedding size for additional source factors. '
                                   'You must provide as many dimensions as '
                                   '(validation) source factor files. Default: %(default)s.')
    model_params.add_argument('--source-factors-combine', '-sfc',
                              choices=C.SOURCE_FACTORS_COMBINE_CHOICES,
                              default=[C.SOURCE_FACTORS_COMBINE_CONCAT],
                              nargs='+',
                              help='How to combine source factors. Can be either one value which will be applied to all '
                              'source factors, or a list of values. Default: %(default)s.')
    model_params.add_argument('--source-factors-share-embedding',
                              type=bool_str(),
                              nargs='+',
                              default=[False],
                              help='Share the embeddings with the source language. Can be either one value which will be '
                              'applied to all source factors, or a list of values. Default: do not share.')

    model_params.add_argument('--weight-tying-type',
                              default=C.WEIGHT_TYING_SRC_TRG_SOFTMAX,
                              choices=C.WEIGHT_TYING_TYPES,
                              help='The type of weight tying. source embeddings=src, target embeddings=trg, '
                                   'target softmax weight matrix=softmax. Default: %(default)s.')

    model_params.add_argument('--dtype', default=C.DTYPE_FP32, choices=[C.DTYPE_FP32, C.DTYPE_FP16],
                              help="Data type.")

    model_params.add_argument('--amp', action='store_true', help='Use MXNet\'s automatic mixed precision (AMP).')
    model_params.add_argument('--amp-scale-interval', type=int, default=2000,
                              help='Attempt to increase loss scale after this many updates without overflow. '
                                   'Default: %(default)s.')


def add_batch_args(params, default_batch_size=4096):
    params.add_argument('--batch-size', '-b',
                        type=int_greater_or_equal(1),
                        default=default_batch_size,
                        help='Mini-batch size per process. Depending on --batch-type, this either refers to words or '
                             'sentences. The effective batch size (update size) is num_processes * batch_size * '
                             'update_interval. Default: %(default)s.')
    params.add_argument('--batch-type',
                        type=str,
                        default=C.BATCH_TYPE_WORD,
                        choices=C.BATCH_TYPES,
                        help='sentence: each batch contains exactly X sentences. '
                             'word: each batch contains approximately X target words. '
                             'max-word: each batch contains at most X target words. '
                             'Default: %(default)s.')
    params.add_argument('--batch-sentences-multiple-of',
                        type=int,
                        default=8,
                        help='For word and max-word batching, guarantee that each batch contains a multiple of X '
                             'sentences. For word batching, round up or down to nearest multiple. For max-word '
                             'batching, always round down. Default: %(default)s.')
    params.add_argument('--round-batch-sizes-to-multiple-of',
                        action=Removed,
                        help='Removed: The argument "--round-batch-sizes-to-multiple-of" has been renamed to '
                             '"--batch-sentences-multiple-of".')
    params.add_argument('--update-interval',
                        type=int,
                        default=1,
                        help='Accumulate gradients over X batches for each model update. Set a value higher than 1 to '
                             'simulate large batches (ex: batch_size 2560 with update_interval 4 gives effective batch '
                             'size 10240). Default: %(default)s.')

def add_hybridization_arg(params):
    params.add_argument('--no-hybridization',
                        action='store_true',
                        help='Turn off hybridization. Hybridization builds a static computation graph and computations will therefore be faster. '
                             'The downside is that one can not set breakpoints to inspect intermediate results. Default: %(default)s.')


def add_training_args(params):
    train_params = params.add_argument_group("Training parameters")

    add_batch_args(train_params)

    train_params.add_argument('--loss',
                              default=C.CROSS_ENTROPY_WITOUT_SOFTMAX_OUTPUT,
                              choices=[C.CROSS_ENTROPY, C.CROSS_ENTROPY_WITOUT_SOFTMAX_OUTPUT],
                              help='Loss to optimize. Default: %(default)s.')
    train_params.add_argument('--label-smoothing',
                              default=0.1,
                              type=float,
                              help='Smoothing constant for label smoothing. Default: %(default)s.')

    train_params.add_argument('--length-task',
                              type=str,
                              default=None,
                              choices=[C.LENGTH_TASK_RATIO, C.LENGTH_TASK_LENGTH],
                              help='If specified, adds an auxiliary task during training to predict source/target length ratios '
                                    '(mean squared error loss), or absolute lengths (Poisson) loss. Default %(default)s.')
    train_params.add_argument('--length-task-weight',
                              type=float_greater_or_equal(0.0),
                              default=1.0,
                              help='The weight of the auxiliary --length-task loss. Default %(default)s.')
    train_params.add_argument('--length-task-layers',
                              type=int_greater_or_equal(1),
                              default=1,
                              help='Number of fully-connected layers for predicting the length ratio. Default %(default)s.')

    train_params.add_argument('--optimized-metric',
                              default=C.PERPLEXITY,
                              choices=C.METRICS,
                              help='Metric to optimize with early stopping {%(choices)s}. Default: %(default)s.')

    train_params.add_argument(C.TRAIN_ARGS_CHECKPOINT_INTERVAL,
                              type=int_greater_or_equal(1),
                              default=4000,
                              help='Checkpoint and evaluate every x updates (update-interval * batches). '
                                   'Default: %(default)s.')

    train_params.add_argument('--min-samples',
                              type=int,
                              default=None,
                              help='Minimum number of samples before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-samples',
                              type=int,
                              default=None,
                              help='Maximum number of samples. Default: %(default)s.')
    train_params.add_argument('--min-updates',
                              type=int,
                              default=None,
                              help='Minimum number of updates before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-updates',
                              type=int,
                              default=None,
                              help='Maximum number of updates. Default: %(default)s.')
    train_params.add_argument('--max-seconds',
                              type=int,
                              default=None,
                              help='Training will stop on the next checkpoint after reaching the maximum seconds. '
                                   'Default: %(default)s.')

    train_params.add_argument('--max-checkpoints',
                              type=int,
                              default=None,
                              help='Maximum number of checkpoints to continue training the model '
                                   'before training is stopped. '
                                   'Default: %(default)s.')
    train_params.add_argument('--max-num-checkpoint-not-improved',
                              type=int,
                              default=None,
                              help='Maximum number of checkpoints the model is allowed to not improve in '
                                   '<optimized-metric> on validation data before training is stopped. '
                                   'Default: %(default)s.')
    train_params.add_argument('--checkpoint-improvement-threshold',
                              type=float,
                              default=0.,
                              help='Improvement in <optimized-metric> over specified number of checkpoints must exceed'
                                   'this value to be considered actual improvement. Default: %(default)s.')

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
                              help='Dropout probability for source & target embeddings. Use "x:x" to specify separate '
                                   'values. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-attention',
                              type=multiple_values(2, data_type=float),
                              default=(0.1, 0.1),
                              help='Dropout probability for multi-head attention. Use "x:x" to specify separate '
                                   'values for encoder & decoder. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-act',
                              type=multiple_values(2, data_type=float),
                              default=(0.1, 0.1),
                              help='Dropout probability before activation in feed-forward block. Use "x:x" to specify '
                                   'separate values for encoder & decoder. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-prepost',
                              type=multiple_values(2, data_type=float),
                              default=(0.1, 0.1),
                              help='Dropout probability for pre/postprocessing blocks. Use "x:x" to specify separate '
                                   'values for encoder & decoder. Default: %(default)s.')

    train_params.add_argument('--optimizer',
                              default=C.OPTIMIZER_ADAM,
                              choices=C.OPTIMIZERS,
                              help='SGD update rule. Default: %(default)s.')
    train_params.add_argument('--optimizer-params',
                              type=simple_dict(),
                              default=None,
                              help='Additional optimizer params as dictionary. Format: key1:value1,key2:value2,...')

    train_params.add_argument('--horovod',
                              action='store_true',
                              help='Use Horovod/MPI for distributed training (Sergeev and Del Balso 2018, '
                                   'arxiv.org/abs/1802.05799). When using this option, run Sockeye with `horovodrun '
                                   '-np X python3 -m sockeye.train` where X is the number of processes. Increasing '
                                   'the number of processes multiplies the effective batch size (ex: batch_size 2560 '
                                   'with `-np 4` gives effective batch size 10240).')

    train_params.add_argument("--kvstore",
                              type=str,
                              default=C.KVSTORE_DEVICE,
                              choices=C.KVSTORE_TYPES,
                              help="The MXNet kvstore to use. 'device' is recommended for single process training. "
                                   "Use any of 'dist_sync', 'dist_device_sync' and 'dist_async' for distributed "
                                   "training. Default: %(default)s.")

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
    train_params.add_argument('--learning-rate-t-scale',
                              type=float,
                              default=1.0,
                              help="Step number is multiplied by this value when determining learning rate for the "
                                   "current step. Default: %(default)s.")
    train_params.add_argument('--learning-rate-reduce-factor',
                              type=float,
                              default=0.9,
                              help="Factor to multiply learning rate with "
                                   "(for 'plateau-reduce' learning rate scheduler). Default: %(default)s.")
    train_params.add_argument('--learning-rate-reduce-num-not-improved',
                              type=int,
                              default=8,
                              help="For 'plateau-reduce' learning rate scheduler. Adjust learning rate "
                                   "if <optimized-metric> did not improve for x checkpoints. Default: %(default)s.")
    train_params.add_argument('--learning-rate-warmup',
                              type=int,
                              default=0,
                              help="Number of warmup steps. If set to x, linearly increases learning rate from 10%% "
                                   "to 100%% of the initial learning rate. Default: %(default)s.")

    train_params.add_argument('--fixed-param-strategy',
                               default=None,
                               choices=C.FIXED_PARAM_STRATEGY_CHOICES,
                               help="Fix various parameters during training using a named strategy. The strategy "
                                    "name indicates which parameters will be fixed (Wuebker et al., 2018). "
                                    "Default: %(default)s.")
    train_params.add_argument('--fixed-param-names',
                              default=[],
                              nargs='*',
                              help="Manually specify names of parameters to fix during training. Default: %(default)s.")

    train_params.add_argument(C.TRAIN_ARGS_MONITOR_BLEU,
                              default=500,
                              type=int,
                              help='x>0: decode x sampled sentences from validation data and '
                                   'compute evaluation metrics. x==-1: use full validation data. Default: %(default)s.')

    train_params.add_argument('--decode-and-evaluate-device-id',
                              default=None,
                              type=int,
                              help='Separate device for decoding validation data. '
                                   'Use a negative number to automatically acquire a GPU. '
                                   'Use a positive number to acquire a specific GPU. Default: %(default)s.')

    train_params.add_argument(C.TRAIN_ARGS_STOP_ON_DECODER_FAILURE,
                              action="store_true",
                              help='Stop training as soon as any checkpoint decoder fails (e.g. because there is not '
                                   'enough GPU memory). Default: %(default)s.')

    train_params.add_argument('--seed',
                              type=int,
                              default=1,
                              help='Random seed. Default: %(default)s.')

    train_params.add_argument('--keep-last-params',
                              type=int,
                              default=-1,
                              help='Keep only the last n params files, use -1 to keep all files. Default: %(default)s')

    train_params.add_argument('--keep-initializations',
                              action="store_true",
                              help='In addition to keeping the last n params files, also keep params from checkpoint 0.')

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
    add_hybridization_arg(params)


def add_translate_cli_args(params):
    add_inference_args(params)
    add_device_args(params)
    add_logging_args(params)
    add_hybridization_arg(params)


def add_score_cli_args(params):
    add_training_data_args(params, required=False)
    add_vocab_args(params)
    add_device_args(params)
    add_batch_args(params, default_batch_size=500)
    add_hybridization_arg(params)

    params = params.add_argument_group("Scoring parameters")

    params.add_argument("--model", "-m", required=True,
                        help="Model directory containing trained model.")

    params.add_argument(C.TRAINING_ARG_MAX_SEQ_LEN,
                        type=multiple_values(num_values=2, greater_or_equal=1),
                        default=None,
                        help='Maximum sequence length in tokens.'
                             'Use "x:x" to specify separate values for src&tgt. Default: Read from model.')

    # common params with translate CLI
    add_length_penalty_args(params)
    add_brevity_penalty_args(params)

    params.add_argument("--output", "-o", default=None,
                        help="File to write output to. Default: STDOUT.")

    params.add_argument('--output-type',
                        default=C.OUTPUT_HANDLER_SCORE,
                        choices=C.OUTPUT_HANDLERS_SCORING,
                        help='Output type. Default: %(default)s.')

    params.add_argument('--score-type',
                        choices=C.SCORING_TYPE_CHOICES,
                        default=C.SCORING_TYPE_DEFAULT,
                        help='Score type to output. Default: %(default)s')

    params.add_argument('--dtype', default=None, choices=[None, C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_INT8],
                        help="Data type. Default: %(default)s infers from saved model.")

    add_logging_args(params)


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
    decode_params.add_argument('--nbest-size',
                               type=int_greater_or_equal(1),
                               default=1,
                               help='Size of the nbest list of translations. Default: %(default)s.')
    decode_params.add_argument('--beam-size', '-b',
                               type=int_greater_or_equal(1),
                               default=5,
                               help='Size of the beam. Default: %(default)s.')

    decode_params.add_argument('--beam-search-stop',
                               choices=[C.BEAM_SEARCH_STOP_ALL, C.BEAM_SEARCH_STOP_FIRST],
                               default=C.BEAM_SEARCH_STOP_ALL,
                               help='Stopping criteria. Quit when (all) hypotheses are finished '
                                    'or when a finished hypothesis is in (first) position. Default: %(default)s.')
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
    decode_params.add_argument('--mc-dropout',
                               default=False,
                               action='store_true',
                               help='Turn on dropout during inference (Monte Carlo dropout). This will make translations non-deterministic and might slow down translation speed.')
    decode_params.add_argument('--sample',
                               type=int_greater_or_equal(0),
                               default=None,
                               nargs='?',
                               const=0,
                               help='Sample from softmax instead of taking best. Optional argument will restrict '
                                    'sampling to top N vocabulary items at each step. Default: %(default)s.')
    decode_params.add_argument('--seed',
                               type=int,
                               default=None,
                               help='Random seed used if sampling. Default: %(default)s.')
    decode_params.add_argument('--ensemble-mode',
                               type=str,
                               default='linear',
                               choices=['linear', 'log_linear'],
                               help='Ensemble mode. Default: %(default)s.')
    decode_params.add_argument('--bucket-width',
                               type=int_greater_or_equal(0),
                               default=10,
                               help='Bucket width for encoder steps. 0 means no bucketing. Default: %(default)s.')
    decode_params.add_argument('--max-input-length',
                               type=int_greater_or_equal(1),
                               default=None,
                               help='Maximum input sequence length. Default: value from model(s).')
    decode_params.add_argument('--max-output-length-num-stds',
                               type=int,
                               default=C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                               help='Number of target-to-source length ratio standard deviations from training to add '
                                    'to calculate maximum output length for beam search for each sentence. '
                                    'Default: %(default)s.')
    decode_params.add_argument('--max-output-length',
                               type=int_greater_or_equal(1),
                               default=None,
                               help='Maximum number of words to generate during translation. '
                                    'If None, it will be computed automatically. Default: %(default)s.')
    decode_params.add_argument('--restrict-lexicon',
                               nargs='+',
                               type=multiple_values(num_values=2, data_type=str),
                               default=None,
                               help="Specify top-k lexicon to restrict output vocabulary to the k most likely context-"
                                    "free translations of the source words in each sentence (Devlin, 2017). See the "
                                    "lexicon module for creating top-k lexicons. To use multiple lexicons, provide "
                                    "'--restrict-lexicon key1:path1 key2:path2 ...' and use JSON input to specify the "
                                    "lexicon for each sentence: "
                                    "{\"text\": \"some input string\", \"restrict_lexicon\": \"key\"}. "
                                    "Default: %(default)s.")
    decode_params.add_argument('--restrict-lexicon-topk',
                               type=int,
                               default=None,
                               help="Specify the number of translations to load for each source word from the lexicon "
                                    "given with --restrict-lexicon. Default: Load all entries from the lexicon.")
    decode_params.add_argument('--avoid-list',
                               type=str,
                               default=None,
                               help="Specify a file containing phrases (pre-processed, one per line) to block "
                                    "from the output. Default: %(default)s.")
    decode_params.add_argument('--strip-unknown-words',
                               action='store_true',
                               default=False,
                               help='Remove any <unk> symbols from outputs. Default: %(default)s.')

    decode_params.add_argument('--output-type',
                               default='translation',
                               choices=C.OUTPUT_HANDLERS,
                               help='Output type. Default: %(default)s.')

    # common params with score CLI
    add_length_penalty_args(decode_params)
    add_brevity_penalty_args(decode_params)

    decode_params.add_argument('--dtype', default=None, choices=[None, C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_INT8],
                               help="Data type. Default: %(default)s infers from saved model.")


def add_length_penalty_args(params):
    params.add_argument('--length-penalty-alpha',
                        default=1.0,
                        type=float,
                        help='Alpha factor for the length penalty used in beam search: '
                             '(beta + len(Y))**alpha/(beta + 1)**alpha. A value of 0.0 will therefore turn off '
                             'length normalization. Default: %(default)s.')
    params.add_argument('--length-penalty-beta',
                        default=0.0,
                        type=float,
                        help='Beta factor for the length penalty used in scoring: '
                        '(beta + len(Y))**alpha/(beta + 1)**alpha. Default: %(default)s')


def add_brevity_penalty_args(params):
    params.add_argument('--brevity-penalty-type',
                        default='none',
                        type=str,
                        choices=[C.BREVITY_PENALTY_NONE, C.BREVITY_PENALTY_LEARNED, C.BREVITY_PENALTY_CONSTANT],
                        help='If specified, adds brevity penalty to the hypotheses\' scores, calculated with learned '
                             'or constant length ratios. The latter, by default, uses the length ratio (|ref|/|hyp|) '
                             'estimated from the training data and averaged over models. Default: %(default)s.')
    params.add_argument('--brevity-penalty-weight',
                        default=1.0,
                        type=float_greater_or_equal(0.0),
                        help='Scaler for the brevity penalty in beam search: weight * log(BP) + score. Default: %(default)s')
    params.add_argument('--brevity-penalty-constant-length-ratio',
                        default=0.0,
                        type=float_greater_or_equal(0.0),
                        help='Has effect if --brevity-penalty-type is set to \'constant\'. If positive, overrides the length '
                             'ratio, used for brevity penalty calculation, for all inputs. If zero, uses the average of length '
                             'ratios from the training data over all models. Default: %(default)s.')


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
                             help="File(s) with hypotheses. If none will read from stdin. Default: stdin.")
    eval_params.add_argument('--metrics',
                             nargs='+',
                             choices=C.EVALUATE_METRICS,
                             default=[C.BLEU, C.CHRF],
                             help='List of metrics to compute. Default: %(default)s.')
    eval_params.add_argument('--sentence', '-s',
                             action="store_true",
                             help="Show sentence-level metrics. Default: %(default)s.")
    eval_params.add_argument('--offset',
                             type=float,
                             default=0.01,
                             help="Numerical value of the offset of zero n-gram counts for BLEU. Default: %(default)s.")
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

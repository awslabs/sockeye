# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
A set of utility methods.
"""
import argparse
import binascii
import gzip
import itertools
import logging
import math
import multiprocessing
import os
import pprint
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from itertools import starmap
from typing import Any, List, Iterator, Iterable, Tuple, Dict, Optional, Union, TypeVar

import numpy as np
import torch as pt
import torch.distributed

# Optional imports. Import errors are not an issue because these modules are
# only used when certain settings are activated. We check that these modules
# can be imported before activating the settings.
try:
    import deepspeed
except ImportError:
    pass

from . import __version__, constants as C
from .log import log_sockeye_version, log_torch_version

logger = logging.getLogger(__name__)


class SockeyeError(Exception):
    pass


def check_version(version: str):
    """
    Checks given version against code version and determines compatibility.
    Throws if versions are incompatible.

    :param version: Given version.
    """
    code_version = parse_version(__version__)
    given_version = parse_version(version)
    if given_version[0] == '3' and given_version[1] == '0':
        logger.info(f"Code version: {__version__}")
        logger.warning(f"Given release version ({version}) does not match code version ({__version__}). "
                       f"Models with version {version} should be compatible though.")
        return
    check_condition(code_version[0] == given_version[0],
                    "Given release version (%s) does not match release code version (%s)" % (version, __version__))
    check_condition(code_version[1] == given_version[1],
                    "Given major version (%s) does not match major code version (%s)" % (version, __version__))


def load_version(fname: str) -> str:
    """
    Loads version from file.

    :param fname: Name of file to load version from.
    :return: Version string.
    """
    if not os.path.exists(fname):
        logger.warning("No version file found. Defaulting to 1.0.3")
        return "1.0.3"
    with open(fname) as inp:
        return inp.read().strip()


def parse_version(version_string: str) -> Tuple[str, str, str]:
    """
    Parse version string into release, major, minor version.

    :param version_string: Version string.
    :return: Tuple of strings.
    """
    release, major, minor = version_string.split(".", 2)
    return release, major, minor


def log_basic_info(args) -> None:
    """
    Log basic information like version number, arguments, etc.

    :param args: Arguments as returned by argparse.
    """
    log_sockeye_version(logger)
    log_torch_version(logger)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)


def seed_rngs(seed: int) -> None:  # type: ignore
    """
    Seed the random number generators (Python, Numpy and MXNet).

    :param seed: The random seed.
    """
    logger.info(f"Random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        logger.info(f"PyTorch seed: {seed}")
    except ImportError:
        pass


def check_condition(condition: bool, error_message: str):
    """
    Check the condition and if it is not met, exit with the given error message
    and error_code, similar to assertions.

    :param condition: Condition to check.
    :param error_message: Error message to show to the user.
    """
    if not condition:
        raise SockeyeError(error_message)


class OnlineMeanAndVariance:
    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.
        self._M2 = 0.

    def update(self, value: Union[float, int]) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._M2 += delta * delta2

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        if self._count < 2:
            return float('nan')
        else:
            return self._M2 / self._count

    @property
    def std(self) -> float:
        variance = self.variance
        return math.sqrt(variance) if not math.isnan(variance) else 0.0


def chunks(some_list: List, n: int) -> Iterable[List]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(some_list), n):
        yield some_list[i:i + n]


def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token


def is_gzip_file(filename: str) -> bool:
    # check for magic gzip number
    with open(filename, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'


def smart_open(filename: str, mode: str = "rt", ftype: str = "auto", errors: str = 'replace'):
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.
    If ftype is "auto" and read mode requested, uses gzip iff is_gzip_file(filename).

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open.
    :param errors: Encoding error handling during reading. Defaults to 'replace'.
    :return: File descriptor.
    """
    if ftype in ('gzip', 'gz') \
            or (ftype == 'auto' and filename.endswith(".gz")) \
            or (ftype == 'auto' and 'r' in mode and is_gzip_file(filename)):
            if mode == "rb" or mode == "wb":
                return gzip.open(filename, mode=mode)
            else:
                return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    else:
        if mode == "rb" or mode == "wb":
            return open(filename, mode=mode)
        else:
            return open(filename, mode=mode, encoding='utf-8', errors=errors)


def combine_means(means: List[Optional[float]], num_sents: List[int]) -> float:
    """
    Takes a list of means and number of sentences of the same length and computes the combined mean.

    :param means: A list of mean values.
    :param num_sents: A list with the number of sentences used to compute each mean value.
    :return: The combined mean of the list of means.
    """
    if not means or not num_sents:
        raise ValueError("Invalid input list.")
    check_condition(len(means) == len(num_sents), "List lengths do not match")
    return sum(num_sent * mean for num_sent, mean in zip(num_sents, means) if mean is not None) / sum(num_sents)


def combine_stds(stds: List[Optional[float]], means: List[Optional[float]], num_sents: List[int]) -> float:
    """
    Takes a list of standard deviations, means and number of sentences of the same length and computes
    the combined standard deviation.

    :param stds: A list of standard deviations.
    :param means: A list of mean values.
    :param num_sents: A list with number of sentences used to compute each mean value.
    :return: The combined standard deviation.
    """
    if not stds or not means or not num_sents:
        raise ValueError("Invalid input list.")
    check_condition(all(len(stds) == len(l) for l in [means, num_sents]), "List lengths do not match") # type: ignore
    total_mean = combine_means(means, num_sents)
    return math.sqrt(sum(num_sent * (std**2 + (mean-total_mean)**2) for num_sent, std, mean in zip(num_sents, stds, means)
                         if std is not None and mean is not None) / sum(num_sents))


def average_tensors(tensors: List[pt.Tensor]) -> pt.Tensor:
    """
    Compute the element-wise average of a list of tensors of the same shape.

    :param tensors: A list of input tensors with the same shape.
    :return: The average of the tensors on the same device as tensors[0].
    """
    if not tensors:
        raise ValueError("tensors is empty.")
    if len(tensors) == 1:
        return tensors[0]
    check_condition(all(tensors[0].shape == t.shape for t in tensors), "tensor shapes do not match")
    return sum(tensors) / len(tensors)  # type: ignore


def gen_prefix_masking(prefix: pt.Tensor, vocab_size: int, dtype: pt.dtype) -> Tuple[pt.Tensor, int]:
    """
    Generate prefix masks from prefix ids, which are inf everywhere except zero for prefix ids.

    :param prefix: Target prefix token or factors in ids. Shape (batch size, max length of prefix).
    :param vocab_size: vocabulary size
    :param dtype: dtype of the retuning output
    :return prefix_masks (batch size, max length of prefix, vocab_size), with type as dtype

    """
    prefix_masks_sizes = list(prefix.size())  # type: List[int]
    max_length = prefix_masks_sizes[1]
    prefix_masks_sizes.append(vocab_size)

    # prefix_masks are inf everywhere except zero for indices of prefix ids.
    prefix_masks = pt.full(prefix_masks_sizes, fill_value=np.inf, device=prefix.device, dtype=dtype)
    prefix_masks.scatter_(-1, prefix.to(pt.int64).unsqueeze(-1), 0.)
    # Note: The use of prefix_masks.scatter_() function is equivalent (but much faster) to
    # prefix_masks[prefix_one_hot != 0] = 0., where
    # prefix_one_hot = pt.nn.functional.one_hot(prefix.to(pt.int64), num_classes=vocab_size).to(prefix.device)

    # In the same batch during inference, it is possible that some translations have target prefix
    # while others do not have. It is also possible that translation may have a target prefix with
    # different length to others. Thus prefix ids may include a full zero vector if a translation
    # in the batch does not have prefix, or include a vector padding with zeros on the right if some
    # translations are with shorter prefix. An example of prefix ids reflecting length differences \
    # is as follows:
    #
    # [1, 2, 3]
    # [1, 2, 0]
    # [0, 0, 0]
    #
    # Here, the first sentence has a prefix of length 3, the second one has a prefix of length 1 \
    # and the last one does not have prefix.
    #
    # At any timestep, some target prefix ids could be 0 (i.e. 0 in the target_prefix means 'no constraint'). \
    # If a prefix id is 0 for a translation at a timestep, all hots in the vocab are assigned to 0 (instead \
    # of only one hot is assigned to 0 and other hots are inf). This makes sure there is no constraint on \
    # selecting any specific target token for the translation in that case.

    prefix_masks.masked_fill_(prefix.unsqueeze(-1) == 0, 0)
    return prefix_masks, max_length


def shift_prefix_factors(prefix_factors: pt.Tensor) -> pt.Tensor:
    """
    Shift prefix factors one step to the right

    :param prefix_factors: tensor ids. Shape (batch size, length, num of factors).
    :return new prefix_factors_shift (batch size, length + 1, num of factors)
    """
    prefix_factors_sizes = prefix_factors.size()
    prefix_factors_shift = pt.zeros(prefix_factors_sizes[0], prefix_factors_sizes[1] + 1, prefix_factors_sizes[2],
                                    dtype=prefix_factors.dtype, device=prefix_factors.device)
    prefix_factors_shift[:, 1:] = prefix_factors
    return prefix_factors_shift


def adjust_first_step_masking(target_prefix: pt.Tensor, first_step_mask: pt.Tensor) -> pt.Tensor:
    """
    Adjust first_step_masking based on the target prefix
    (Target prefix for each input in the same batch may have a different length. \
    Thus first_step_mask needs to be adjusted accordingly.)

    :param target_prefix: Shape (batch size, max target prefix length).
    :param first_step_mask: Shape (batch_size * beam_size, 1)
    :return (adjusted) first_steps_masking (batch_size * beam_size, max target prefix length + 1).

    An illustrative example of how first_step_masking is adjusted

    Inputs:

    target_prefix (batch_size = 2, max target prefix length = 2)

    tensor([1 2]
           [1 0])
    Note: Two target prefix tokens in the first sentence, \
    one target prefix token in the second sentence.

    first_step_mask (batch_size = 2 * beam_size = 5, 1)

    tensor([[0],
    [inf],
    [inf],
    [inf],
    [inf],
    [0],
    [inf],
    [inf],
    [inf],
    [inf])

    Output:
    Adjusted first_step_mask (batch_size * beam_size, max target prefix length + 1):

    tensor([[0 0 0],
            [inf inf inf],
            [inf inf inf],
            [inf inf inf],
            [inf inf, inf],
            [0 0 0],
            [inf inf 0],
            [inf inf 0],
            [inf inf 0],
            [inf inf 0]])

    The concrete steps of what this function does are as follows:

    Step 1: Create a zero masking matrix with shape (batch size, max target prefix length + 1)
    Fill 1 into this masking matrix based on the target prefix

    target prefix  initialize masking    masking      roll one step to the right
                   from target prefix    is not 0      and assign 1 at index 0
        [1 2]    ->    [1 2 0]        -> [1 1 0]  ->           [1 1 1]
        [1 0]          [1 0 0]           [1 0 0]               [1 1 0]

    Step 2: Adjust first_step_mask based on masking

    masking     expand masking with     expand first_step_mask with max target
                     beam size         prefix length, fill 0 where masking is 0
    [1 1 1]  ->      [1 1 1]        ->             [0 0 0]
    [1 1 0]          [1 1 1]                       [inf inf inf]
                     [1 1 1]                       [inf inf inf]
                     [1 1 1]                       [inf inf inf]
                     [1 1 1]                       [inf inf inf]
                     [1 1 0]                       [0 0 0]
                     [1 1 0]                       [inf inf 0]
                     [1 1 0]                       [inf inf 0]
                     [1 1 0]                       [inf inf 0]
                     [1 1 0]                       [inf inf 0]
    """
    batch_beam, _  = first_step_mask.size()
    batch, max_prefix_len = target_prefix.size()
    beam_size = batch_beam // batch
    # Step 1
    masking = pt.zeros((batch, max_prefix_len + 1), device=target_prefix.device)
    masking[:, :max_prefix_len] = target_prefix
    masking = pt.clamp(masking, 0., 1.) # force all non zero ids to 1
    masking = pt.roll(masking, 1, -1)
    masking[:, 0] = 1.

    # Step 2
    masking = masking.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_beam, -1)
    first_step_mask = first_step_mask.expand(-1, masking.size(-1)).clone()
    first_step_mask.masked_fill_(masking == 0., 0.)
    return first_step_mask


def parse_metrics_line(line_number: int, line: str) -> Dict[str, Any]:
    """
    Parse a line of metrics into a mappings of key and values.

    :param line_number: Line's number for checking if checkpoints are aligned to it.
    :param line: A line from the Sockeye metrics file.
    :return: Dictionary of metric names (e.g. perplexity-train) mapping to a list of values.
    """
    fields = line.split('\t')
    checkpoint = int(fields[0])
    check_condition(line_number == checkpoint,
                    "Line (%d) and loaded checkpoint (%d) do not align." % (line_number, checkpoint))
    metric = dict()  # type: Dict[str, Any]
    for field in fields[1:]:
        key, value = field.split("=", 1)
        if value == 'True' or value == 'False':
            metric[key] = (value == 'True')
        elif value == 'None':
            metric[key] = None
        else:
            metric[key] = float(value)
    return metric


def read_metrics_file(path: str) -> List[Dict[str, Any]]:
    """
    Reads lines metrics file and returns list of mappings of key and values.

    :param path: File to read metric values from.
    :return: Dictionary of metric names (e.g. perplexity-train) mapping to a list of values.
    """
    with open(path) as fin:
        metrics = [parse_metrics_line(i, line.strip()) for i, line in enumerate(fin, 1)]
    return metrics


def write_metrics_file(metrics: List[Dict[str, Any]], path: str):
    """
    Write metrics data to tab-separated file.

    :param metrics: metrics data.
    :param path: Path to write to.
    """
    with open(path, 'w') as metrics_out:
        for checkpoint, metric_dict in enumerate(metrics, 1):
            metrics_str = "\t".join(["{}={}".format(name, value) for name, value in sorted(metric_dict.items())])
            metrics_out.write("{}\t{}\n".format(checkpoint, metrics_str))


def get_validation_metric_points(model_path: str, metric: str):
    """
    Returns tuples of value and checkpoint for given metric from metrics file at model_path.
    :param model_path: Model path containing .metrics file.
    :param metric: Metric values to extract.
    :return: List of tuples (value, checkpoint).
    """
    metrics_path = os.path.join(model_path, C.METRICS_NAME)
    data = read_metrics_file(metrics_path)
    return [(d['%s-val' % metric], cp) for cp, d in enumerate(data, 1)]


def grouper(iterable: Iterable, size: int) -> Iterable:
    """
    Collect data into fixed-length chunks or blocks without discarding underfilled chunks or padding them.

    :param iterable: A sequence of inputs.
    :param size: Chunk size.
    :return: Sequence of chunks.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def metric_value_is_better(new: float, old: float, metric: str) -> bool:
    """
    Returns true if new value is strictly better than old for given metric.
    """
    if C.METRIC_MAXIMIZE[metric]:
        return new > old
    else:
        return new < old


_DTYPE_TO_STRING = {
    np.float16: C.DTYPE_FP16,
    np.float32: C.DTYPE_FP32,
    np.int8: C.DTYPE_INT8,
    np.int32: C.DTYPE_INT32,
    pt.bfloat16: C.DTYPE_BF16,
    pt.float16: C.DTYPE_FP16,
    pt.float32: C.DTYPE_FP32,
    pt.int8: C.DTYPE_INT8,
    pt.int32: C.DTYPE_INT32,
}


def dtype_to_str(dtype):
    return _DTYPE_TO_STRING.get(dtype, str(dtype))


_STRING_TO_TORCH_DTYPE = {
    C.DTYPE_BF16: pt.bfloat16,
    C.DTYPE_FP16: pt.float16,
    C.DTYPE_FP32: pt.float32,
    C.DTYPE_INT8: pt.int8,
    C.DTYPE_INT32: pt.int32,
}


def get_torch_dtype(dtype: Union[pt.dtype, str]) -> pt.dtype:
    if isinstance(dtype, pt.dtype):
        return dtype
    if dtype in _STRING_TO_TORCH_DTYPE:
        return _STRING_TO_TORCH_DTYPE[dtype]
    raise ValueError(f'Cannot convert to Torch dtype: {dtype}')


_STRING_TO_NUMPY_DTYPE = {
    C.DTYPE_FP16: np.float16,
    C.DTYPE_FP32: np.float32,
    C.DTYPE_INT8: np.int8,
    C.DTYPE_INT16: np.int16,
    C.DTYPE_INT32: np.int32,
}


def get_numpy_dtype(dtype: Union[np.dtype, str]):
    if isinstance(dtype, np.dtype):
        return dtype
    if dtype in _STRING_TO_NUMPY_DTYPE:
        return _STRING_TO_NUMPY_DTYPE[dtype]
    raise ValueError(f'Cannot convert to NumPy dtype: {dtype}')


def log_parameters(model: pt.nn.Module):
    """
    Logs information about model parameters.
    """
    fixed_parameter_names = []
    learned_parameter_names = []
    total_learned = 0
    total_fixed = 0
    visited = defaultdict(list)
    for name, module in model.named_modules(remove_duplicate=False):
        for param_name, param in module.named_parameters(prefix=name, recurse=False):
            repr = "%s [%s, %s]" % (name, tuple(param.shape), dtype_to_str(param.dtype))
            size = param.shape.numel()
            if not param.requires_grad:
                fixed_parameter_names.append(repr)
                total_fixed += size if param not in visited else 0
            else:
                total_learned += size if param not in visited else 0
                learned_parameter_names.append(repr)
            visited[param].append(param_name)
    shared_parameter_names = []  # type: List[str]
    total_shared = 0
    for param, names in visited.items():
        if len(names) > 1:
            total_shared += param.shape.numel()
            shared_parameter_names.append(" = ".join(names))
    total_parameters = total_learned + total_fixed
    logger.info("# of parameters: %d | trainable: %d (%.2f%%) | shared: %d (%.2f%%) | fixed: %d (%.2f%%)",
                total_parameters,
                total_learned, total_learned / total_parameters * 100,
                total_shared, total_shared / total_parameters * 100,
                total_fixed, total_fixed / total_parameters * 100)
    logger.info("Trainable parameters: \n%s", pprint.pformat(learned_parameter_names))
    logger.info("Shared parameters: \n%s", pprint.pformat(shared_parameter_names, width=120))
    logger.info("Fixed parameters:\n%s", pprint.pformat(fixed_parameter_names))


@contextmanager
def no_context():
    """
    No-op context manager that can be used in "with" statements
    """
    yield None


class SingleProcessPool:

    def map(self, func, iterable):
        return list(map(func, iterable))

    def starmap(self, func, iterable):
        return list(starmap(func, iterable))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_pool(max_processes):
    if max_processes == 1:
        return SingleProcessPool()
    else:
        return multiprocessing.pool.Pool(processes=max_processes)


def update_dict(dest: Dict[Any, Any], source: Dict[Any, Any]):
    for key in source:
        if isinstance(source[key], dict):
            if key not in dest:
                dest[key] = {}
            if not isinstance(dest[key], dict):
                raise ValueError(f'Type mismatch for key {key}: {type(source[key])} vs {type(dest[key])}')
            update_dict(dest[key], source[key])
        else:
            dest[key] = source[key]


def update_dict_with_prefix_kv(dest: Dict[Any, Any], prefix_kv: Dict[Any, Any]):
    """
    Update a dictionary in place with prefix key-value dictionary.

    :param dest: Dictionary to update.
    :param prefix_kv: Dictionary of prefix keys and values. Prefix keys have the
                      form [prefix.]key (e.g., prefix_field1.prefix_field2.key).
    """
    for keys, value in prefix_kv.items():
        split_keys = keys.split('.')
        prefix = split_keys[:-1]
        key = split_keys[-1]
        _dict = dest
        for prefix_field in prefix:
            if prefix_field not in _dict:
                _dict[prefix_field] = {}
            _dict = _dict[prefix_field]
        _dict[key] = value


def is_distributed() -> bool:
    return torch.distributed.is_initialized()


def is_primary_worker() -> bool:
    """
    True when current process is the primary worker (rank 0) or the only worker
    (not running in distributed mode)
    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def get_local_rank() -> int:
    return int(os.environ[C.DIST_ENV_LOCAL_RANK])


T = TypeVar('T')


def broadcast_object(obj: T, src: int = 0) -> T:
    """
    Broadcast a single Python object across workers (default source is primary
    worker with rank 0)
    """
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def all_gather_object(obj: T) -> List[T]:
    """Gather each worker's instance of an object, returned as a list"""
    obj_list = [None] * torch.distributed.get_world_size()  # type: List[T]
    torch.distributed.all_gather_object(obj_list, obj)
    return obj_list


# Track whether DeepSpeed has been initialized
_using_deepspeed = False

def init_deepspeed():
    """
    Make sure all of the DeepSpeed modules we use can be imported, initialize
    DeepSpeed, and set the global variable that tracks initialization.

    """
    global _using_deepspeed
    try:
        import deepspeed  # pylint: disable=E0401
        import deepspeed.utils.zero_to_fp32  # pylint: disable=E0401
        deepspeed.init_distributed()
        _using_deepspeed = True
    except:
        raise RuntimeError('To train models with DeepSpeed (https://www.deepspeed.ai/), '
                           'install the module with `pip install deepspeed`.')


def using_deepspeed() -> bool:
    """Check whether DeepSpeed has been initialized via this module"""
    return _using_deepspeed


# Track whether Faiss has been confirmed importable
_faiss_checked = False

def check_import_faiss():
    """
    Make sure the faiss module can be imported.
    """
    global _faiss_checked
    if not _faiss_checked:
        try:
            import faiss  # pylint: disable=E0401
            _faiss_checked = True
        except:
            raise RuntimeError('To run kNN-MT models, please install faiss by following '
                               'https://github.com/facebookresearch/faiss/blob/main/INSTALL.md')


def count_seq_len(sample: str, count_type: str = 'char', replace_tokens: Optional[List] = None) -> int:
    """
    Count sequence length, after replacing (optional) token/s.
    """
    if replace_tokens is not None:
        for tokens in replace_tokens:
            sample = sample.replace(tokens, '')
    if count_type == C.SEQ_LEN_IN_CHARACTERS:
        return len(sample.replace(C.TOKEN_SEPARATOR, ''))
    elif count_type == C.SEQ_LEN_IN_TOKENS:
        return len(sample.split(C.TOKEN_SEPARATOR))
    else:
        raise SockeyeError("Sequence length count type '%s' unknown. "
                           "Choices are: %s" % (count_type, [C.SEQ_LEN_IN_CHARACTERS, C.SEQ_LEN_IN_TOKENS]))


def compute_isometric_score(hypothesis: str, hypothesis_score: float, source: str,
                            isometric_metric: str = 'isometric-ratio', isometric_alpha: float = 0.5):
    """
    Compute hypothesis to source isometric score using sample char length
    and isometric metric (ratio, diff, lc).
        - isometric-diff: https://aclanthology.org/W19-5210.pdf
        - isometric-ratio/lc: https://arxiv.org/pdf/2110.03847.pdf
    :return isometric score
    """
    count_type = C.SEQ_LEN_IN_CHARACTERS  # for isometric scoring, count length in characters
    replace_tokens = C.TOKEN_SEGMENTATION_MARKERS

    hypothesis_len = count_seq_len(hypothesis, count_type, replace_tokens)
    source_len = count_seq_len(source, count_type, replace_tokens)

    if isometric_metric == C.RERANK_ISOMETRIC_LC:
        abs_len_diff = abs(hypothesis_len - source_len)
        isometric_score = (abs_len_diff*100) / source_len if source_len else abs_len_diff*100

        return isometric_score
    else:
        if isometric_metric == C.RERANK_ISOMETRIC_RATIO:
            len_ratio = hypothesis_len/source_len if source_len else hypothesis_len
            synchrony_score = float(1/(1 + len_ratio))

        if isometric_metric == C.RERANK_ISOMETRIC_DIFF:
            abs_len_diff = abs(hypothesis_len - source_len)
            synchrony_score = float(1/(1 + abs_len_diff))

        # isometric score, if alpha=0.0 takes model prediction score
        pred_sub_score = (1 - isometric_alpha) * float(hypothesis_score)
        synchrony_sub_score = isometric_alpha * synchrony_score
        isometric_score = pred_sub_score + synchrony_sub_score

        return isometric_score


def init_device(args: argparse.Namespace) -> pt.device:
    """
    Select Torch device based on CLI args:
    - When CUDA is not available, the device defaults to CPU.
    - When using CUDA, tf32 is enabled if specified.
    - When running distributed training, the CUDA device is determined by local
      rank instead of CLI args.

    :param args: Parsed CLI args including device parameters.

    :return: Torch device.
    """

    use_cpu = args.use_cpu
    if not use_cpu and not pt.cuda.is_available():
        logger.info('CUDA not available, defaulting to CPU device')
        use_cpu = True
    if use_cpu:
        return pt.device('cpu')

    device = pt.device('cuda', get_local_rank() if is_distributed() else args.device_id)
    # Ensure that GPU operations use the correct device by default
    pt.cuda.set_device(device)
    if args.tf32:
        pt.backends.cuda.matmul.allow_tf32 = True
        logger.info('CUDA: allow tf32 (float32 but with 10 bits precision)')

    return device

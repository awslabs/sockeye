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
    np.float32: 'float32',
    np.float16: 'float16',
    np.int8: 'int8',
    np.int32: 'int32',
    pt.float32: 'float32',
    pt.float16: 'float16',
    pt.int32: 'int32',
    pt.int8: 'int8',
}


def _print_dtype(dtype):
    return _DTYPE_TO_STRING.get(dtype, str(dtype))


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
            repr = "%s [%s, %s]" % (name, tuple(param.shape), _print_dtype(param.dtype))
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

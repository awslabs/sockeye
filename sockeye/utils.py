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
A set of utility methods.
"""
import binascii
import errno
import gzip
import itertools
import logging
import math
import multiprocessing
import os
import pprint
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import reduce
from typing import Any, List, Iterator, Iterable, Set, Tuple, Dict, Optional, Union, IO, TypeVar, cast
from itertools import starmap

import torch as pt
import torch.distributed

import numpy as np
import portalocker

from . import __version__, constants as C
from . import horovod_mpi
from .log import log_sockeye_version, log_mxnet_version, log_torch_version

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
    # TODO(fhieber): temporarily maintain backwards compoatibility for 2.3.x models
    if given_version[0] == '2' and given_version[1] == '3':
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
    log_mxnet_version(logger)
    log_torch_version(logger)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)


def seed_rngs(seed: int, ctx: Optional[Union['mx.Context', List['mx.Context']]] = None) -> None:  # type: ignore
    """
    Seed the random number generators (Python, Numpy and MXNet).

    :param seed: The random seed.
    :param ctx: Random number generators in MXNet are device specific.
           If None, MXNet will set the state of each generator of each device using seed and device id. This will lead
           to different results on different devices. If ctx is provided, this function will seed
           device-specific generators with a fixed offset. E.g. for 2 devices and seed=13, seed for gpu(0) will be 13,
           14 for gpu(1). See https://beta.mxnet.io/api/gluon-related/_autogen/mxnet.random.seed.html.
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
    try:
        import mxnet as mx
        if ctx is None:
            mx.random.seed(seed, ctx='all')
        else:
            if isinstance(ctx, mx.Context):
                ctx = [ctx]
            for i, c in enumerate(ctx):
                mx.random.seed(seed + i, ctx=c)
    except:
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


def average_arrays(arrays: List['np.ndarray']) -> 'np.ndarray':
    """
    Take a list of arrays of the same shape and take the element wise average.

    :param arrays: A list of ndarrays with the same shape that will be averaged.
    :return: The average of the ndarrays in the same context as arrays[0].
    """
    from mxnet import npx
    if not arrays:
        raise ValueError("arrays is empty.")
    if len(arrays) == 1:
        return arrays[0]
    check_condition(all(arrays[0].shape == a.shape for a in arrays), "nd array shapes do not match")
    return npx.add_n(*arrays) / len(arrays)


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


def get_num_gpus() -> int:
    """
    Gets the number of GPUs available on the host.

    :return: The number of GPUs on the system.
    """
    try:
        import mxnet as mx
    except ImportError:
        return 0
    try:
        return mx.context.num_gpus()
    except:
        # Some builds of MXNet will raise a CUDA error when CUDA is not
        # installed on the host.  In this case, zero GPUs are available.
        return 0


def get_gpu_memory_usage(ctx: Union['mx.context.Context', List['mx.context.Context']]) -> Dict[int, Tuple[int, int]]:  # type: ignore
    """
    Returns used and total memory for GPUs identified by the given context list.

    :param ctx: List of MXNet context devices.
    :return: Dictionary of device id mapping to a tuple of (memory used, memory total).
    """
    try:
        import mxnet as mx
    except ImportError:
        return {}
    if not isinstance(ctx, List):
        ctx = [ctx]
    ctx = [c for c in ctx if c.device_type == 'gpu']
    if not ctx:
        return {}

    memory_data = {}  # type: Dict[int, Tuple[int, int]]
    for c in ctx:
        try:
            free, total = mx.context.gpu_memory_info(device_id=c.device_id)  # in bytes
            used = total - free
            memory_data[c.device_id] = (used * 1e-06, total * 1e-06)
        except mx.MXNetError:
            logger.exception("Failed retrieving memory data for gpu%d", c.device_id)
            continue
    log_gpu_memory_usage(memory_data)
    return memory_data


def log_gpu_memory_usage(memory_data: Dict[int, Tuple[int, int]]):
    log_str = " ".join(
        "GPU %d: %d/%d MB (%.2f%%)" % (k, v[0], v[1], v[0] * 100.0 / v[1]) for k, v in memory_data.items() if v[1])
    logger.info(log_str)


def determine_context(device_ids: List[int],
                      use_cpu: bool,
                      disable_device_locking: bool,
                      lock_dir: str,
                      exit_stack: ExitStack) -> List['mx.Context']:  # type: ignore
    """
    Determine the MXNet context to run on (CPU or GPU).

    :param device_ids: List of device as defined from the CLI.
    :param use_cpu: Whether to use the CPU instead of GPU(s).
    :param disable_device_locking: Disable Sockeye's device locking feature.
    :param lock_dir: Directory to place device lock files in.
    :param exit_stack: An ExitStack from contextlib.

    :return: A list with the context(s) to run on.
    """
    try:
        import mxnet as mx
    except ImportError:
        return []
    if use_cpu:
        context = [mx.cpu()]
    else:
        num_gpus = get_num_gpus()
        check_condition(num_gpus >= 1,
                        "No GPUs found, consider running on the CPU with --use-cpu ")
        if horovod_mpi.using_horovod():
            # Running with Horovod/MPI: GPU(s) are determined by local rank
            check_condition(len(device_ids) == 1 and device_ids[0] < 0,
                            "When using Horovod, --device-ids should be a negative integer indicating the number of "
                            "GPUs each worker should use.")
            n_ids = -device_ids[0]
            context = [mx.gpu(_id + horovod_mpi.hvd.local_rank() * n_ids) for _id in range(n_ids)]
        else:
            if disable_device_locking:
                context = expand_requested_device_ids(device_ids)
            else:
                context = exit_stack.enter_context(acquire_gpus(device_ids, lock_dir=lock_dir))
            context = [mx.gpu(gpu_id) for gpu_id in context]
    return context


def expand_requested_device_ids(requested_device_ids: List[int]) -> List[int]:
    """
    Transform a list of device id requests to concrete device ids. For example on a host with 8 GPUs when requesting
    [-4, 3, 5] you will get [0, 1, 2, 3, 4, 5]. Namely you will get device 3 and 5, as well as 3 other available
    device ids (starting to fill up from low to high device ids).

    :param requested_device_ids: The requested device ids, each number is either negative indicating the number of GPUs
     that will be allocated, or positive indicating we want to acquire a specific device id.
    :return: A list of device ids.
    """
    num_gpus_available = get_num_gpus()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.warning("Sockeye currently does not respect CUDA_VISIBLE_DEVICE settings when locking GPU devices.")
    return _expand_requested_device_ids(requested_device_ids, num_gpus_available)


def _expand_requested_device_ids(requested_device_ids: List[int], num_gpus_available: int) -> List[int]:
    if num_gpus_available == 0:
        raise RuntimeError("Can not acquire GPU, as no GPUs were found on this machine.")

    num_arbitrary_device_ids = 0
    device_ids = []
    for device_id in requested_device_ids:
        if device_id < 0:
            num_gpus = -device_id
            num_arbitrary_device_ids += num_gpus
        else:
            device_ids.append(device_id)
    num_gpus_requested = len(device_ids) + num_arbitrary_device_ids
    if num_gpus_requested > num_gpus_available:
        raise ValueError("Requested %d GPUs, but only %d are available." % (num_gpus_requested, num_gpus_available))
    remaining_device_ids = set(range(num_gpus_available)) - set(device_ids)
    logger.info("Attempting to acquire %d GPUs of %d GPUs.", num_gpus_requested, num_gpus_available)
    return device_ids + list(remaining_device_ids)[:num_arbitrary_device_ids]


@contextmanager
def acquire_gpus(requested_device_ids: List[int], lock_dir: str = "/tmp",
                 retry_wait_min: int = 10, retry_wait_rand: int = 60,
                 num_gpus_available: Optional[int] = None):
    """
    Acquire a number of GPUs in a transactional way. This method should be used inside a `with` statement.
    Will try to acquire all the requested number of GPUs. If currently
    not enough GPUs are available all locks will be released and we wait until we retry. Will retry until enough
    GPUs become available.

    :param requested_device_ids: The requested device ids, each number is either negative indicating the number of GPUs
     that will be allocated, or positive indicating we want to acquire a specific device id.
    :param lock_dir: The directory for storing the lock file.
    :param retry_wait_min: The minimum number of seconds to wait between retries.
    :param retry_wait_rand: Randomly add between 0 and `retry_wait_rand` seconds to the wait time.
    :param num_gpus_available: The number of GPUs available, if None we will call get_num_gpus().
    :return: yields a list of GPU ids.
    """
    if num_gpus_available is None:
        num_gpus_available = get_num_gpus()
    if num_gpus_available == 0:
        raise RuntimeError("Can not acquire GPU, as no GPUs were found on this machine.")

    if not os.path.exists(lock_dir):
        raise IOError("Lock directory %s does not exist." % lock_dir)

    if not os.access(lock_dir, os.W_OK):
        raise IOError("Lock directory %s is not writeable." % lock_dir)

    # split the device ids into the specific ids requested and count up the number of arbitrary ids we want
    # e.g. device_ids = [-3, 2, 5, 7, -5] means we want to acquire device 2, 5 and 7 plus 8 other devices.
    specific_device_ids = set()  # type: Set[int]
    num_arbitrary_device_ids = 0
    for device_id in requested_device_ids:
        if device_id < 0:
            num_gpus = -device_id
            num_arbitrary_device_ids += num_gpus
        else:
            if device_id in specific_device_ids:
                raise ValueError("Requested GPU %d twice." % device_id)
            specific_device_ids.add(device_id)

    # make sure we have enough GPUs available
    num_gpus_requested = len(specific_device_ids) + num_arbitrary_device_ids
    if num_gpus_requested > num_gpus_available:
        raise ValueError("Requested %d GPUs, but only %d are available." % (num_gpus_requested, num_gpus_available))
    logger.info("Attempting to acquire %d GPUs of %d GPUs. The requested devices are: %s",
                num_gpus_requested, num_gpus_available, str(requested_device_ids))

    # note: it's important to first allocate the specific device ids and then the others to not deadlock ourselves.

    # for specific device ids we just have the device id itself as a candidate
    candidates_to_request = [[device_id] for device_id in specific_device_ids]

    # for the arbitrary device ids we take all remaining device ids as a list of candidates
    remaining_device_ids = [device_id for device_id in range(num_gpus_available)
                            if device_id not in specific_device_ids]
    candidates_to_request += [remaining_device_ids for _ in range(num_arbitrary_device_ids)]

    while True:

        with ExitStack() as exit_stack:
            any_failed = False
            acquired_gpus = []  # type: List[int]
            with GpuFileLock(candidates=["master_lock"], lock_dir=lock_dir) as master_lock:  # type: str
                # Only one process, determined by the master lock, can try acquiring gpu locks at a time.
                # This will make sure that we use consecutive device ids whenever possible.
                if master_lock is not None:
                    for candidates in candidates_to_request:
                        gpu_id = exit_stack.enter_context(GpuFileLock(candidates=candidates, lock_dir=lock_dir))
                        if gpu_id is not None:
                            acquired_gpus.append(cast(int, gpu_id))
                        else:
                            if len(candidates) == 1:
                                logger.info("Could not acquire GPU %d. It's currently locked.", candidates[0])
                            any_failed = True
                            break
            if master_lock is not None and not any_failed:
                try:
                    yield acquired_gpus
                except:  # pylint: disable=try-except-raise
                    raise
                return

        # randomize so that multiple processes starting at the same time don't retry at a similar point in time
        if retry_wait_rand > 0:
            retry_wait_actual = retry_wait_min + random.randint(0, retry_wait_rand)
        else:
            retry_wait_actual = retry_wait_min

        if master_lock is None:
            logger.info("Another process is acquiring GPUs at the moment will try again in %ss." % retry_wait_actual)
        else:
            logger.info("Not enough GPUs available will try again in %ss." % retry_wait_actual)
        time.sleep(retry_wait_actual)


GpuDeviceType = TypeVar('GpuDeviceType')


class GpuFileLock:
    """
    Acquires a single GPU by locking a file (therefore this assumes that everyone using GPUs calls this method and
    shares the lock directory). Sets target to a GPU id or None if none is available.

    :param candidates: List of candidate device ids to try to acquire.
    :param lock_dir: The directory for storing the lock file.
    """

    def __init__(self, candidates: List[GpuDeviceType], lock_dir: str) -> None:
        self.candidates = candidates
        self.lock_dir = lock_dir
        self.lock_file = None  # type: Optional[IO[Any]]
        self.lock_file_path = None  # type: Optional[str]
        self.gpu_id = None  # type: Optional[GpuDeviceType]
        self._acquired_lock = False

    def __enter__(self) -> Optional[GpuDeviceType]:
        for gpu_id in self.candidates:
            lockfile_path = os.path.join(self.lock_dir, "sockeye.gpu{}.lock".format(gpu_id))
            try:
                lock_file = open(lockfile_path, 'w')
            except IOError:
                if errno.EACCES:
                    logger.warning("GPU {} is currently locked by a different process "
                                   "(Permission denied).".format(gpu_id))
                    continue
            try:
                # exclusive non-blocking lock
                portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                # got the lock, let's write our PID into it:
                lock_file.write("%d\n" % os.getpid())
                lock_file.flush()

                self._acquired_lock = True
                self.gpu_id = gpu_id
                self.lock_file = lock_file
                self.lockfile_path = lockfile_path

                logger.info("Acquired GPU {}.".format(gpu_id))

                return gpu_id
            except portalocker.LockException as e:
                # portalocker packages the original exception,
                # we dig it out and raise if unrelated to us
                if e.args[0].errno != errno.EAGAIN:  # pylint: disable=no-member
                    logger.error("Failed acquiring GPU lock.", exc_info=True)
                    raise e.args[0]
                else:
                    logger.debug("GPU {} is currently locked.".format(gpu_id))
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_id is not None:
            logger.info("Releasing GPU {}.".format(self.gpu_id))
        if self.lock_file is not None:
            if self._acquired_lock:
                portalocker.lock(self.lock_file, portalocker.LOCK_UN)
            self.lock_file.close()
            os.remove(self.lockfile_path)


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


def log_parameters(params: C.ParameterDict):
    """
    Logs information about model parameters.
    """
    fixed_parameter_names = []
    learned_parameter_names = []
    total_learned = 0
    total_fixed = 0
    visited = defaultdict(list)
    for name, param in sorted(params.items()):
        repr = "%s [%s, %s]" % (name, param.shape, _print_dtype(param.dtype))
        size = reduce(lambda x, y: x * y, param.shape)
        if size == 0:
            logger.debug("Parameter shape for '%s' not yet fully inferred, using 0", name)
        if param.grad_req == 'null':
            fixed_parameter_names.append(repr)
            total_fixed += size
        else:
            total_learned += size if param not in visited else 0
            learned_parameter_names.append(repr)
        visited[param].append(name)
    shared_parameter_names = []
    for param, names in visited.items():
        if len(names) > 1:
            shared_parameter_names.append(" = ".join(names))
    total_parameters = total_learned + total_fixed
    logger.info("# of parameters: %d | trainable: %d (%.2f%%) | fixed: %d (%.2f%%)",
                total_parameters,
                total_learned, total_learned / total_parameters * 100,
                total_fixed, total_fixed / total_parameters * 100)
    logger.info("Trainable parameters: \n%s", pprint.pformat(learned_parameter_names))
    logger.info("Shared parameters: \n%s", pprint.pformat(shared_parameter_names))
    logger.info("Fixed parameters:\n%s", pprint.pformat(fixed_parameter_names))


def log_parameters_pt(model: pt.nn.Module):
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

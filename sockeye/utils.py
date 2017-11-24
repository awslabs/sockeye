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
A set of utility methods.
"""
import argparse
import collections
import errno
import fcntl
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import itertools
from contextlib import contextmanager, ExitStack
from typing import Mapping, NamedTuple, Any, List, Iterator, Iterable, Set, TextIO, Tuple, Dict, Optional

import mxnet as mx
import numpy as np

import sockeye.constants as C
from sockeye import __version__
from sockeye.log import log_sockeye_version, log_mxnet_version

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
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)


def seedRNGs(args: argparse.Namespace) -> None:
    """
    Seed the random number generators (Python, Numpy and MXNet)

    :param args: Arguments as returned by argparse.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)


def check_condition(condition: bool, error_message: str):
    """
    Check the condition and if it is not met, exit with the given error message
    and error_code, similar to assertions.

    :param condition: Condition to check.
    :param error_message: Error message to show to the user.
    """
    if not condition:
        raise SockeyeError(error_message)


def save_graph(symbol: mx.sym.Symbol, filename: str, hide_weights: bool = True):
    """
    Dumps computation graph visualization to .pdf and .dot file.

    :param symbol: The symbol representing the computation graph.
    :param filename: The filename to save the graphic to.
    :param hide_weights: If true the weights will not be shown.
    """
    dot = mx.viz.plot_network(symbol, hide_weights=hide_weights)
    dot.render(filename=filename)


def compute_lengths(sequence_data: mx.sym.Symbol) -> mx.sym.Symbol:
    """
    Computes sequence lenghts of PAD_ID-padded data in sequence_data.

    :param sequence_data: Input data. Shape: (batch_size, seq_len).
    :return: Length data. Shape: (batch_size,).
    """
    return mx.sym.sum(mx.sym.broadcast_not_equal(sequence_data, mx.sym.zeros((1,))), axis=1)


def save_params(arg_params: Mapping[str, mx.nd.NDArray], fname: str,
                aux_params: Optional[Mapping[str, mx.nd.NDArray]] = None):
    """
    Saves the parameters to a file.

    :param arg_params: Mapping from parameter names to the actual parameters.
    :param fname: The file name to store the parameters in.
    :param aux_params: Optional mapping from parameter names to the auxiliary parameters.
    """
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    if aux_params is not None:
        save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


def load_params(fname: str) -> Tuple[Dict[str, mx.nd.NDArray], Dict[str, mx.nd.NDArray]]:
    """
    Loads parameters from a file.

    :param fname: The file containing the parameters.
    :return: Mapping from parameter names to the actual parameters for both the arg parameters and the aux parameters.
    """
    save_dict = mx.nd.load(fname)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


class Accuracy(mx.metric.EvalMetric):
    """
    Calculates accuracy. Taken from MXNet and adapted to work with batch-major labels
    (reshapes (batch_size, time) -> (batch_size * time).
    Also allows defining an ignore_label/pad symbol
    """

    def __init__(self,
                 name='accuracy',
                 output_names=None,
                 label_names=None,
                 ignore_label=None):
        super(Accuracy, self).__init__(name=name,
                                       output_names=output_names,
                                       label_names=label_names,
                                       ignore_label=ignore_label)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax_channel(pred_label)
            pred_label = pred_label.asnumpy().astype('int32')
            label = mx.nd.reshape(label, shape=(pred_label.size,)).asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)
            if self.ignore_label is not None:
                correct = ((pred_label.flat == label.flat) * (label.flat != self.ignore_label)).sum()
                ignore = (label.flat == self.ignore_label).sum()
                n = pred_label.size - ignore
            else:
                correct = (pred_label.flat == label.flat).sum()
                n = pred_label.size

            self.sum_metric += correct
            self.num_inst += n


def smallest_k(matrix: np.ndarray, k: int,
               only_first_row: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Find the smallest elements in a numpy matrix.

    :param matrix: Any matrix.
    :param k: The number of smallest elements to return.
    :param only_first_row: If true the search is constrained to the first row of the matrix.
    :return: The row indices, column indices and values of the k smallest items in matrix.
    """
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()

    # args are the indices in flatten of the k smallest elements
    args = np.argpartition(flatten, k)[:k]
    # args are the indices in flatten of the sorted k smallest elements
    args = args[np.argsort(flatten[args])]
    # flatten[args] are the values for args
    return np.unravel_index(args, matrix.shape), flatten[args]


def smallest_k_mx(matrix: mx.nd.NDArray, k: int,
                  only_first_row: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Find the smallest elements in a NDarray.

    :param matrix: Any matrix.
    :param k: The number of smallest elements to return.
    :param only_first_row: If True the search is constrained to the first row of the matrix.
    :return: The row indices, column indices and values of the k smallest items in matrix.
    """
    if only_first_row:
        matrix = mx.nd.reshape(matrix[0], shape=(1, -1))

    # pylint: disable=unbalanced-tuple-unpacking
    values, indices = mx.nd.topk(matrix, axis=None, k=k, ret_typ='both', is_ascend=True)

    return np.unravel_index(indices.astype(np.int32).asnumpy(), matrix.shape), values


def chunks(some_list: List, n: int) -> Iterable[List]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(some_list), n):
        yield some_list[i:i + n]


def plot_attention(attention_matrix: np.ndarray, source_tokens: List[str], target_tokens: List[str], filename: str):
    """
    Uses matplotlib for creating a visualization of the attention matrix.

    :param attention_matrix: The attention matrix.
    :param source_tokens: A list of source tokens.
    :param target_tokens: A list of target tokens.
    :param filename: The file to which the attention visualization will be written to.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert attention_matrix.shape[0] == len(target_tokens)

    plt.imshow(attention_matrix.transpose(), interpolation="nearest", cmap="Greys")
    plt.xlabel("target")
    plt.ylabel("source")
    plt.gca().set_xticks([i for i in range(0, len(target_tokens))])
    plt.gca().set_yticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_xticklabels(target_tokens, rotation='vertical')
    plt.gca().set_yticklabels(source_tokens)
    plt.tight_layout()
    plt.savefig(filename)
    logger.info("Saved alignment visualization to " + filename)


def print_attention_text(attention_matrix: np.ndarray, source_tokens: List[str], target_tokens: List[str],
                         threshold: float):
    """
    Prints the attention matrix to standard out.

    :param attention_matrix: The attention matrix.
    :param source_tokens: A list of source tokens.
    :param target_tokens: A list of target tokens.
    :param threshold: The threshold for including an alignment link in the result.
    """
    sys.stdout.write("  ")
    for _ in target_tokens:
        sys.stdout.write("---")
    sys.stdout.write("\n")
    for i, f_i in enumerate(source_tokens):  # type: ignore
        sys.stdout.write(" |")
        for j in range(len(target_tokens)):
            align_prob = attention_matrix[j, i]
            if align_prob > threshold:
                sys.stdout.write("(*)")
            elif align_prob > 0.4:
                sys.stdout.write("(?)")
            else:
                sys.stdout.write("   ")
        sys.stdout.write(" | %s\n" % f_i)
    sys.stdout.write("  ")
    for _ in target_tokens:
        sys.stdout.write("---")
    sys.stdout.write("\n")
    for k in range(max(map(len, target_tokens))):
        sys.stdout.write("  ")
        for word in target_tokens:
            letter = word[k] if len(word) > k else " "
            sys.stdout.write(" %s " % letter)
        sys.stdout.write("\n")
    sys.stdout.write("\n")


def get_alignments(attention_matrix: np.ndarray, threshold: float = .9) -> Iterator[Tuple[int, int]]:
    """
    Yields hard alignments from an attention_matrix (target_length, source_length)
    given a threshold.

    :param attention_matrix: The attention matrix.
    :param threshold: The threshold for including an alignment link in the result.
    :return: Generator yielding strings of the form 0-0, 0-1, 2-1, 2-2, 3-4...
    """
    for src_idx in range(attention_matrix.shape[1]):
        for trg_idx in range(attention_matrix.shape[0]):
            if attention_matrix[trg_idx, src_idx] > threshold:
                yield (src_idx, trg_idx)


def average_arrays(arrays: List[mx.nd.NDArray]) -> mx.nd.NDArray:
    """
    Take a list of arrays of the same shape and take the element wise average.

    :param arrays: A list of NDArrays with the same shape that will be averaged.
    :return: The average of the NDArrays in the same context as arrays[0].
    """
    if len(arrays) == 1:
        return arrays[0]
    check_condition(all(arrays[0].shape == a.shape for a in arrays), "nd array shapes do not match")
    new_array = mx.nd.zeros(arrays[0].shape, dtype=arrays[0].dtype, ctx=arrays[0].context)
    for a in arrays:
        new_array += a.as_in_context(new_array.context)
    new_array /= len(arrays)
    return new_array


def get_num_gpus() -> int:
    """
    Gets the number of GPUs available on the host (depends on nvidia-smi).

    :return: The number of GPUs on the system.
    """
    if shutil.which("nvidia-smi") is None:
        logger.warning("Couldn't find nvidia-smi, therefore we assume no GPUs are available.")
        return 0
    sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()[0].decode("utf-8")
    num_gpus = len(out_str.rstrip("\n").split("\n"))
    return num_gpus


def get_gpu_memory_usage(ctx: List[mx.context.Context]) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Returns used and total memory for GPUs identified by the given context list.

    :param ctx: List of MXNet context devices.
    :return: Dictionary of device id mapping to a tuple of (memory used, memory total).
    """
    if isinstance(ctx, mx.context.Context):
        ctx = [ctx]
    ctx = [c for c in ctx if c.device_type == 'gpu']
    if not ctx:
        return None
    if shutil.which("nvidia-smi") is None:
        logger.warning("Couldn't find nvidia-smi, therefore we assume no GPUs are available.")
        return {}
    ids = [str(c.device_id) for c in ctx]
    query = "--query-gpu=index,memory.used,memory.total"
    format = "--format=csv,noheader,nounits"
    sp = subprocess.Popen(['nvidia-smi', query, format, "-i", ",".join(ids)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = sp.communicate()[0].decode("utf-8").rstrip().split("\n")
    memory_data = {}
    for line in result:
        gpu_id, mem_used, mem_total = line.split(",")
        memory_data[int(gpu_id)] = (int(mem_used), int(mem_total))
    return memory_data


def log_gpu_memory_usage(memory_data: Dict[int, Tuple[int, int]]):
    log_str = " ".join("GPU %d: %d/%d MB (%.2f%%)" % (k, v[0], v[1], v[0] * 100.0/v[1]) for k, v in memory_data.items())
    logger.info(log_str)


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
                 num_gpus_available: Optional[int]=None):
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
            acquired_gpus = []  # type: List[int]
            any_failed = False
            for candidates in candidates_to_request:
                gpu_id = exit_stack.enter_context(GpuFileLock(candidates=candidates, lock_dir=lock_dir))
                if gpu_id is not None:
                    acquired_gpus.append(gpu_id)
                else:
                    if len(candidates) == 1:
                        logger.info("Could not acquire GPU %d. It's currently locked.", candidates[0])
                    any_failed = True
                    break
            if not any_failed:
                try:
                    yield acquired_gpus
                except:
                    raise
                return
        # couldn't acquire all GPUs, let's wait and try again later

        # randomize so that multiple processes starting at the same time don't retry at a similar point in time
        if retry_wait_rand > 0:
            retry_wait_actual = retry_wait_min + random.randint(0, retry_wait_rand)
        else:
            retry_wait_actual = retry_wait_min
        logger.info("Not enough GPUs available will try again in %ss." % retry_wait_actual)
        time.sleep(retry_wait_actual)


class GpuFileLock:
    """
    Acquires a single GPU by locking a file (therefore this assumes that everyone using GPUs calls this method and
    shares the lock directory). Sets target to a GPU id or None if none is available.

    :param candidates: List of candidate device ids to try to acquire.
    :param lock_dir: The directory for storing the lock file.
    """

    def __init__(self, candidates: List[int], lock_dir: str) -> None:
        self.candidates = candidates
        self.lock_dir = lock_dir
        self.lock_file = None  # type: Optional[TextIO]
        self.lock_file_path = None  # type: Optional[str]
        self.gpu_id = None  # type: Optional[int]
        self._acquired_lock = False

    def __enter__(self) -> Optional[int]:
        for gpu_id in self.candidates:
            lockfile_path = os.path.join(self.lock_dir, "sockeye.gpu%d.lock" % gpu_id)
            lock_file = open(lockfile_path, 'w')
            try:
                # exclusive non-blocking lock
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # got the lock, let's write our PID into it:
                lock_file.write("%d\n" % os.getpid())
                lock_file.flush()

                self._acquired_lock = True
                self.gpu_id = gpu_id
                self.lock_file = lock_file
                self.lockfile_path = lockfile_path

                logger.info("Acquired GPU %d." % gpu_id)

                return gpu_id
            except IOError as e:
                # raise on unrelated IOErrors
                if e.errno != errno.EAGAIN:
                    logger.error("Failed acquiring GPU lock.", exc_info=True)
                    raise
                else:
                    logger.debug("GPU %d is currently locked.", gpu_id)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_id is not None:
            logger.info("Releasing GPU %d.", self.gpu_id)
        if self.lock_file is not None:
            if self._acquired_lock:
                fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            os.remove(self.lockfile_path)


def namedtuple_with_defaults(typename, field_names, default_values: Mapping[str, Any] = ()) -> NamedTuple:
    """
    Create a named tuple with default values.

    :param typename: The name of the new type.
    :param field_names: The fields the type will have.
    :param default_values: A mapping from field names to default values.
    :return: The new named tuple with default values.
    """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def read_metrics_file(path: str) -> List[Dict[str, Any]]:
    """
    Reads lines metrics file and returns list of mappings of key and values.

    :param path: File to read metric values from.
    :return: Dictionary of metric names (e.g. perplexity-train) mapping to a list of values.
    """
    metrics = []
    with open(path) as fin:
        for i, line in enumerate(fin, 1):
            fields = line.strip().split('\t')
            checkpoint = int(fields[0])
            check_condition(i == checkpoint,
                            "Line (%d) and loaded checkpoint (%d) do not align." % (i, checkpoint))
            metric = dict()
            for field in fields[1:]:
                key, value = field.split("=", 1)
                metric[key] = float(value)
            metrics.append(metric)
    return metrics


def write_metrics_file(metrics: List[Dict[str, Any]], path: str):
    """
    Write metrics data to tab-separated file.

    :param metrics: metrics data.
    :param path: Path to write to.
    """
    with open(path, 'w') as metrics_out:
        for checkpoint, metric_dict in enumerate(metrics, 1):
            metrics_str = "\t".join(["%s=%.6f" % (name, value) for name, value in sorted(metric_dict.items())])
            metrics_out.write("%d\t%s\n" % (checkpoint, metrics_str))


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


class PrintValue(mx.operator.CustomOp):
    """
    Custom operator that takes a symbol, prints its value to stdout and
    propagates the value unchanged. Useful for debugging.

    Use it as:
    my_sym = mx.sym.Custom(op_type="PrintValue", data=my_sym, print_name="My symbol")

    Additionally you can use the optional arguments 'use_logger=True' for using
    the system logger and 'print_grad=True' for printing information about the
    gradient (out_grad, i.e. "upper part" of the graph).
    """
    def __init__(self, print_name, print_grad: str, use_logger: str):
        super().__init__()
        self.print_name = print_name
        # Note that all the parameters are serialized as strings
        self.print_grad = (print_grad == "True")
        self.use_logger = (use_logger == "True")

    def __print_nd__(self, nd: mx.nd.array, label: str):
        intro = "%s %s - shape %s" % (label, self.print_name, str(nd.shape))
        if self.use_logger:
            logger.info(intro)
            logger.info(str(nd.asnumpy()))
        else:
            print(">>>>> ", intro)
            print(nd.asnumpy())

    def forward(self, is_train, req, in_data, out_data, aux):
        self.__print_nd__(in_data[0], "Symbol")
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.print_grad:
            self.__print_nd__(out_grad[0], "Grad")
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("PrintValue")
class PrintValueProp(mx.operator.CustomOpProp):
    def __init__(self, print_name: str, print_grad: bool = False, use_logger: bool = False):
        super().__init__(need_top_grad=True)
        self.print_name = print_name
        self.print_grad = print_grad
        self.use_logger = use_logger

    def list_arguments(self):
        return ["data"]

    def list_outputs(self):
        return ["output"]

    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def infer_type(self, in_type):
        return in_type, in_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return PrintValue(self.print_name,
                          print_grad=self.print_grad,
                          use_logger=self.use_logger)


def grouper(iterable: Iterable, size: int) -> Iterable:
    """
    Collect data into fixed-length chunks or blocks without discarding underfilled chunks or padding them.

    :param iterable: A sequence of inputs.
    :return: Sequence of chunks.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk

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
import errno
import glob
import gzip
import itertools
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, ExitStack
from typing import Mapping, Any, List, Iterator, Iterable, Set, Tuple, Dict, Optional, Union, IO, TypeVar, cast

import fcntl
import mxnet as mx
import numpy as np

from sockeye import __version__, constants as C
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


def seedRNGs(seed: int) -> None:
    """
    Seed the random number generators (Python, Numpy and MXNet)

    :param seed: The random seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)


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
    Computes sequence lengths of PAD_ID-padded data in sequence_data.

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
            """TODO(fhieber):
            temporary weight split for models with combined weight for keys & values
            in transformer source attention layers. This can be removed once with the next major version change."""
            if "att_enc_kv2h_weight" in name:
                logger.info("Splitting '%s' parameters into separate k & v matrices.", name)
                v_split = mx.nd.split(v, axis=0, num_outputs=2)
                arg_params[name.replace('kv2h', "k2h")] = v_split[0]
                arg_params[name.replace('kv2h', "v2h")] = v_split[1]
            else:
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


def topk(scores: mx.nd.NDArray,
         k: int,
         batch_size: int,
         offset: np.ndarray,
         use_mxnet_topk: bool) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, mx.nd.NDArray]]:
    """
    Get the lowest k elements per sentence from a `scores` matrix.

    :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
    :param k: The number of smallest scores to return.
    :param batch_size: Number of sentences being decoded at once.
    :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
    :param use_mxnet_topk: True to use the mxnet implementation or False to use the numpy one.
    :return: The row indices, column indices and values of the k smallest items in matrix.
    """
    # (batch_size, beam_size * target_vocab_size)
    folded_scores = scores.reshape((batch_size, k * scores.shape[-1]))

    if use_mxnet_topk:
        # pylint: disable=unbalanced-tuple-unpacking
        values, indices = mx.nd.topk(folded_scores, axis=1, k=k, ret_typ='both', is_ascend=True)
        best_hyp_indices, best_word_indices = np.unravel_index(indices.astype(np.int32).asnumpy().ravel(), scores.shape)
        values = values.reshape((-1,))
    else:
        folded_scores = folded_scores.asnumpy()
        # Get the scores
        # Indexes into folded_scores: (batch_size, beam_size)
        flat_idxs = np.argpartition(folded_scores, range(k))[:, :k]
        # Score values: (batch_size, beam_size)
        values = folded_scores[np.arange(folded_scores.shape[0])[:, None], flat_idxs].ravel()
        best_hyp_indices, best_word_indices = np.unravel_index(flat_idxs.ravel(), scores.shape)

    if batch_size > 1:
        # Offsetting the indices to match the shape of the scores matrix
        best_hyp_indices += offset
    return best_hyp_indices, best_word_indices, values


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


def smart_open(filename: str, mode: str = "rt", ftype: str = "auto", errors: str = 'replace'):
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open
    :param errors: Encoding error handling during reading. Defaults to 'replace'
    :return: File descriptor
    """
    if ftype == 'gzip' or ftype == 'gz' or (ftype == 'auto' and filename.endswith(".gz")):
        return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    else:
        return open(filename, mode=mode, encoding='utf-8', errors=errors)


def plot_attention(attention_matrix: np.ndarray, source_tokens: List[str], target_tokens: List[str], filename: str):
    """
    Uses matplotlib for creating a visualization of the attention matrix.

    :param attention_matrix: The attention matrix.
    :param source_tokens: A list of source tokens.
    :param target_tokens: A list of target tokens.
    :param filename: The file to which the attention visualization will be written to.
    """
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError("Please install matplotlib.")
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


def get_gpu_memory_usage(ctx: List[mx.context.Context]) -> Dict[int, Tuple[int, int]]:
    """
    Returns used and total memory for GPUs identified by the given context list.

    :param ctx: List of MXNet context devices.
    :return: Dictionary of device id mapping to a tuple of (memory used, memory total).
    """
    if isinstance(ctx, mx.context.Context):
        ctx = [ctx]
    ctx = [c for c in ctx if c.device_type == 'gpu']
    if not ctx:
        return {}
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
    log_gpu_memory_usage(memory_data)
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
                except:
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
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # got the lock, let's write our PID into it:
                lock_file.write("%d\n" % os.getpid())
                lock_file.flush()

                self._acquired_lock = True
                self.gpu_id = gpu_id
                self.lock_file = lock_file
                self.lockfile_path = lockfile_path

                logger.info("Acquired GPU {}.".format(gpu_id))

                return gpu_id
            except IOError as e:
                # raise on unrelated IOErrors
                if e.errno != errno.EAGAIN:
                    logger.error("Failed acquiring GPU lock.", exc_info=True)
                    raise
                else:
                    logger.debug("GPU {} is currently locked.".format(gpu_id))
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_id is not None:
            logger.info("Releasing GPU {}.".format(self.gpu_id))
        if self.lock_file is not None:
            if self._acquired_lock:
                fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            os.remove(self.lockfile_path)


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
    def __init__(self, print_name, print_grad: str, use_logger: str) -> None:
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
    def __init__(self, print_name: str, print_grad: bool = False, use_logger: bool = False) -> None:
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


def cleanup_params_files(output_folder: str, max_to_keep: int, checkpoint: int, best_checkpoint: int):
    """
    Deletes oldest parameter files from a model folder.

    :param output_folder: Folder where param files are located.
    :param max_to_keep: Maximum number of files to keep, negative to keep all.
    :param checkpoint: Current checkpoint (i.e. index of last params file created).
    :param best_checkpoint: Best checkpoint. The parameter file corresponding to this checkpoint will not be deleted.
    """
    if max_to_keep <= 0:
        return
    existing_files = glob.glob(os.path.join(output_folder, C.PARAMS_PREFIX + "*"))
    params_name_with_dir = os.path.join(output_folder, C.PARAMS_NAME)
    for n in range(0, max(1, checkpoint - max_to_keep + 1)):
        if n != best_checkpoint:
            param_fname_n = params_name_with_dir % n
            if param_fname_n in existing_files:
                os.remove(param_fname_n)


def cast_conditionally(data: mx.sym.Symbol, dtype: str) -> mx.sym.Symbol:
    """
    Workaround until no-op cast will be fixed in MXNet codebase.
    Creates cast symbol only if dtype is different from default one, i.e. float32.

    :param data: Input symbol.
    :param dtype: Target dtype.
    :return: Cast symbol or just data symbol.
    """
    if dtype != C.DTYPE_FP32:
        return mx.sym.cast(data=data, dtype=dtype)
    return data


def uncast_conditionally(data: mx.sym.Symbol, dtype: str) -> mx.sym.Symbol:
    """
    Workaround until no-op cast will be fixed in MXNet codebase.
    Creates cast to float32 symbol only if dtype is different from default one, i.e. float32.

    :param data: Input symbol.
    :param dtype: Input symbol dtype.
    :return: Cast symbol or just data symbol.
    """
    if dtype != C.DTYPE_FP32:
        return mx.sym.cast(data=data, dtype=C.DTYPE_FP32)
    return data

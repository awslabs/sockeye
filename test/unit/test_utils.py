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

import math
import os
import re
import tempfile

import mxnet as mx
import numpy as np
import pytest

from sockeye import __version__
from sockeye import constants as C
from sockeye import utils


@pytest.mark.parametrize("some_list, expected", [
    ([1, 2, 3, 4, 5, 6, 7, 8], [[1, 2, 3], [4, 5, 6], [7, 8]]),
    ([1, 2], [[1, 2]]),
    ([1, 2, 3], [[1, 2, 3]]),
    ([1, 2, 3, 4], [[1, 2, 3], [4]]),
])
def test_chunks(some_list, expected):
    chunk_size = 3
    chunked_list = list(utils.chunks(some_list, chunk_size))
    assert chunked_list == expected


def test_get_alignments():
    attention_matrix = np.asarray([[0.1, 0.4, 0.5],
                                   [0.2, 0.8, 0.0],
                                   [0.4, 0.4, 0.2]])
    test_cases = [(0.5, [(1, 1)]),
                  (0.8, []),
                  (0.1, [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)])]

    for threshold, expected_alignment in test_cases:
        alignment = list(utils.get_alignments(attention_matrix, threshold=threshold))
        assert alignment == expected_alignment


device_params = [([-4, 3, 5], 6, [0, 1, 2, 3, 4, 5]),
                 ([-2, 3, -2, 5], 6, [0, 1, 2, 3, 4, 5]),
                 ([-1], 1, [0]),
                 ([1], 1, [1])]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params)
def test_expand_requested_device_ids(requested_device_ids, num_gpus_available, expected):
    assert set(utils._expand_requested_device_ids(requested_device_ids, num_gpus_available)) == set(expected)


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params)
def test_aquire_gpus(tmpdir, requested_device_ids, num_gpus_available, expected):
    with utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                            num_gpus_available=num_gpus_available) as acquired_gpus:
        assert set(acquired_gpus) == set(expected)
        # make sure the master lock does not exist anymore after acquiring
        # (but rather just one lock per acquired GPU)
        assert len(tmpdir.listdir()) == len(acquired_gpus)


# We expect the following settings to raise a ValueError
device_params_expected_exception = [
    # requesting the same gpu twice
    ([-4, 3, 3, 5], 5),
    # too few GPUs available
    ([-4, 3, 5], 5),
    ([3, 5], 1),
    ([-2], 1),
    ([-1, -1], 1)]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available", device_params_expected_exception)
def test_expand_requested_device_ids_exception(requested_device_ids, num_gpus_available):
    with pytest.raises(ValueError):
        utils._expand_requested_device_ids(requested_device_ids, num_gpus_available)


@pytest.mark.parametrize("requested_device_ids, num_gpus_available", device_params_expected_exception)
def test_aquire_gpus_exception(tmpdir, requested_device_ids, num_gpus_available):
    with pytest.raises(ValueError):
        with utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                                num_gpus_available=num_gpus_available) as _:
            pass


# Let's assume GPU 1 is locked already
device_params_1_locked = [([-4, 3, 5], 7, [0, 2, 3, 4, 5, 6]),
                          ([-2, 3, -2, 5], 7, [0, 2, 3, 4, 5, 6])]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params_1_locked)
def test_aquire_gpus_1_locked(tmpdir, requested_device_ids, num_gpus_available, expected):
    gpu_1 = 1
    with utils.GpuFileLock([gpu_1], str(tmpdir)) as lock:
        with utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                                num_gpus_available=num_gpus_available) as acquired_gpus:
            assert set(acquired_gpus) == set(expected)


def test_acquire_gpus_exception_propagation(tmpdir):
    raised_exception = RuntimeError("This exception should be propagated properly.")
    caught_exception = None
    try:
        with utils.acquire_gpus([-1, 4, -1], lock_dir=str(tmpdir), num_gpus_available=12) as _:
            raise raised_exception
    except Exception as e:
        caught_exception = e
    assert caught_exception is raised_exception


def test_gpu_file_lock_cleanup(tmpdir):
    gpu_id = 0
    candidates = [gpu_id]

    # Test that the lock files get created and clean up
    with utils.GpuFileLock(candidates, str(tmpdir)) as lock:
        assert lock == gpu_id
        assert tmpdir.join("sockeye.gpu0.lock").check(), "Lock file did not exist."
        assert not tmpdir.join("sockeye.gpu1.lock").check(), "Unrelated lock file did exist"
    assert not tmpdir.join("sockeye.gpu0.lock").check(), "Lock file was not cleaned up."


def test_gpu_file_lock_exception_propagation(tmpdir):
    gpu_ids = [0]
    # Test that exceptions are properly propagated
    raised_exception = RuntimeError("This exception should be propagated properly.")
    caught_exception = None
    try:
        with utils.GpuFileLock(gpu_ids, str(tmpdir)) as lock:
            raise raised_exception
    except Exception as e:
        caught_exception = e
    assert caught_exception is raised_exception


def test_gpu_file_lock_locking(tmpdir):
    # the second time we try to acquire a lock for the same device we should not succeed
    gpu_id = 0
    candidates = [gpu_id]

    with utils.GpuFileLock(candidates, str(tmpdir)) as lock_inner:
        assert lock_inner == 0
        with utils.GpuFileLock(candidates, str(tmpdir)) as lock_outer:
            assert lock_outer is None


def test_gpu_file_lock_permission_exception(tmpdir):
    tmpdir = tmpdir.mkdir("sub")
    existing_lock = tmpdir.join("sockeye.gpu0.lock")
    # remove permissions
    existing_lock.write("")
    existing_lock.chmod(0)

    with utils.GpuFileLock([0, 1], str(tmpdir)) as acquired_lock:
        # We expect to ignore the file for which we do not have permission and acquire the other device instead
        assert acquired_lock == 1


def test_check_condition_true():
    utils.check_condition(1 == 1, "Nice")


def test_check_condition_false():
    with pytest.raises(utils.SockeyeError) as e:
        utils.check_condition(1 == 2, "Wrong")
    assert "Wrong" == str(e.value)


@pytest.mark.parametrize("version_string,expected_version", [("1.0.3", ("1", "0", "3")),
                                                             ("1.0.2.3", ("1", "0", "2.3"))])
def test_parse_version(version_string, expected_version):
    assert expected_version == utils.parse_version(version_string)


def test_check_version_disregards_minor():
    release, major, minor = utils.parse_version(__version__)
    other_minor_version = "%s.%s.%d" % (release, major, int(minor) + 1)
    utils.check_version(other_minor_version)


def _get_later_major_version():
    release, major, minor = utils.parse_version(__version__)
    return "%s.%d.%s" % (release, int(major) + 1, minor)


def test_check_version_checks_major():
    version = _get_later_major_version()
    with pytest.raises(utils.SockeyeError) as e:
        utils.check_version(version)
    assert "Given major version (%s) does not match major code version (%s)" % (version, __version__) == str(e.value)


def test_version_matches_changelog():
    """
    Tests whether the last version mentioned in CHANGELOG.md matches the sockeye version (sockeye/__init__.py).
    """
    pattern = re.compile(r'''## \[([0-9.]+)\]''')
    changelog = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "CHANGELOG.md")).read()
    last_changelog_version = pattern.findall(changelog)[0]
    assert __version__ == last_changelog_version


@pytest.mark.parametrize("samples,expected_mean, expected_variance",
                         [
                             ([1, 2], 1.5, 0.25),
                             ([4., 100., 12., -3, 1000, 1., -200], 130.57142857142858, 132975.38775510204),
                         ])
def test_online_mean_and_variance(samples, expected_mean, expected_variance):
    mean_and_variance = utils.OnlineMeanAndVariance()
    for sample in samples:
        mean_and_variance.update(sample)

    assert np.isclose(mean_and_variance.mean, expected_mean)
    assert np.isclose(mean_and_variance.variance, expected_variance)


@pytest.mark.parametrize("samples,expected_mean",
                         [
                             ([], 0.),
                             ([5.], 5.),
                         ])
def test_online_mean_and_variance_nan(samples, expected_mean):
    mean_and_variance = utils.OnlineMeanAndVariance()
    for sample in samples:
        mean_and_variance.update(sample)

    assert np.isclose(mean_and_variance.mean, expected_mean)
    assert math.isnan(mean_and_variance.variance)


get_tokens_tests = [("this is a line  \n", ["this", "is", "a", "line"]),
                    (" a  \tb \r \n", ["a", "b"])]


@pytest.mark.parametrize("line, expected_tokens", get_tokens_tests)
def test_get_tokens(line, expected_tokens):
    tokens = list(utils.get_tokens(line))
    assert tokens == expected_tokens


def test_average_arrays():
    n = 4
    shape = (12, 14)
    arrays = [np.random.uniform(0, 1, (12, 14)) for _ in range(n)]
    expected_average = np.zeros(shape)
    for array in arrays:
        expected_average += array
    expected_average /= 4

    mx_arrays = [mx.nd.array(a) for a in arrays]
    assert np.isclose(utils.average_arrays(mx_arrays).asnumpy(), expected_average).all()

    with pytest.raises(utils.SockeyeError) as e:
        other_shape = (12, 13)
        utils.average_arrays(mx_arrays + [mx.nd.zeros(other_shape)])
    assert "nd array shapes do not match" == str(e.value)


def test_save_and_load_params():
    array = mx.nd.uniform(0, 1, (10, 12))
    arg_params = {"array": array}
    aux_params = {"array": array}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "params")
        utils.save_params(arg_params, path, aux_params=aux_params)
        params = mx.nd.load(path)
        assert len(params.keys()) == 2
        assert "arg:array" in params.keys()
        assert "aux:array" in params.keys()
        loaded_arg_params, loaded_aux_params = utils.load_params(path)
        assert "array" in loaded_arg_params
        assert "array" in loaded_aux_params
        assert np.isclose(loaded_arg_params['array'].asnumpy(), array.asnumpy()).all()
        assert np.isclose(loaded_aux_params['array'].asnumpy(), array.asnumpy()).all()


def test_print_value():
    data = mx.sym.Variable("data")
    weights = mx.sym.Variable("weights")
    softmax_label = mx.sym.Variable("softmax_label")

    fc = mx.sym.FullyConnected(data=data, num_hidden=128, weight=weights, no_bias=True)
    out = mx.sym.SoftmaxOutput(data=fc, label=softmax_label, name="softmax")

    fc_print = mx.sym.Custom(op_type="PrintValue", data=fc, print_name="FullyConnected")
    out_print = mx.sym.SoftmaxOutput(data=fc_print, label=softmax_label, name="softmax")

    data_np = np.random.rand(1, 256)
    weights_np = np.random.rand(128, 256)
    label_np = np.random.rand(1, 128)

    executor_base = out.simple_bind(mx.cpu(), data=(1, 256), softmax_label=(1, 128), weights=(128, 256))
    executor_base.arg_dict["data"][:] = data_np
    executor_base.arg_dict["weights"][:] = weights_np
    executor_base.arg_dict["softmax_label"][:] = label_np

    executor_print = out_print.simple_bind(mx.cpu(), data=(1, 256), softmax_label=(1, 128), weights=(128, 256))
    executor_print.arg_dict["data"][:] = data_np
    executor_print.arg_dict["weights"][:] = weights_np
    executor_print.arg_dict["softmax_label"][:] = label_np

    output_base = executor_base.forward(is_train=True)[0]
    output_print = executor_print.forward(is_train=True)[0]
    assert np.isclose(output_base.asnumpy(), output_print.asnumpy()).all()

    executor_base.backward()
    executor_print.backward()
    assert np.isclose(executor_base.grad_arrays[1].asnumpy(), executor_print.grad_arrays[1].asnumpy()).all()


@pytest.mark.parametrize("new, old, metric, result",
                         [(0, 0, C.PERPLEXITY, False),
                          (1.0, 1.0, C.PERPLEXITY, False),
                          (1.0, 0.9, C.PERPLEXITY, False),
                          (0.99, 1.0, C.PERPLEXITY, True),
                          (C.LARGE_POSITIVE_VALUE, np.inf, C.PERPLEXITY, True),
                          (0, 0, C.BLEU, False),
                          (1.0, 1.0, C.BLEU, False),
                          (1.0, 0.9, C.BLEU, True),
                          (0.99, 1.0, C.BLEU, False),
                          (C.LARGE_POSITIVE_VALUE, np.inf, C.BLEU, False),
                         ])
def test_metric_value_is_better(new, old, metric, result):
    assert utils.metric_value_is_better(new, old, metric) == result




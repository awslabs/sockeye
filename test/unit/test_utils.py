# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import gzip
import math
import os
import re
from tempfile import TemporaryDirectory

import pytest
import numpy as np
import torch as pt

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


device_params = [([-4, 3, 5], 6, [0, 1, 2, 3, 4, 5]),
                 ([-2, 3, -2, 5], 6, [0, 1, 2, 3, 4, 5]),
                 ([-1], 1, [0]),
                 ([1], 1, [1])]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params)
def test_expand_requested_device_ids(requested_device_ids, num_gpus_available, expected):
    assert set(utils._expand_requested_device_ids(requested_device_ids, num_gpus_available)) == set(expected)


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params)
def test_acquire_gpus(tmpdir, requested_device_ids, num_gpus_available, expected):
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
def test_acquire_gpus_exception(tmpdir, requested_device_ids, num_gpus_available):
    with pytest.raises(ValueError):
        with utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                                num_gpus_available=num_gpus_available) as _:
            pass


# Let's assume GPU 1 is locked already
device_params_1_locked = [([-4, 3, 5], 7, [0, 2, 3, 4, 5, 6]),
                          ([-2, 3, -2, 5], 7, [0, 2, 3, 4, 5, 6])]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params_1_locked)
def test_acquire_gpus_1_locked(tmpdir, requested_device_ids, num_gpus_available, expected):
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


@pytest.mark.parametrize("samples, sample_means, expected_mean",
                         [
                             ([[1.23, 0.474, 9.516], [10.219, 5.31, 9, 21.90, 98]], [3.74, 28.8858], 19.456125),
                             ([[-10, 10, 4.3, -4.3], [102], [0, 1]], [0.0, 102.0, 0.5], 14.714285714285714),
                             ([[], [-1], [0, 1]], [None, -1.0, 0.5], 0.0),
                             ([[], [1.99], [], [], [0]], [None, 1.99, None, None, 0.0], 0.995),
                             ([[2.45, -5.21, -20, 81.92, 41, 1, 0.1123, 1.2], []], [12.8090375, None], 12.8090375)
                         ])
def test_combine_means(samples, sample_means, expected_mean):
    num_sents = [len(l) for l in samples]
    combined_mean = utils.combine_means(sample_means, num_sents)
    assert np.isclose(expected_mean, combined_mean)


@pytest.mark.parametrize("samples, sample_means, sample_stds, expected_std",
                         [
                             ([[-10, 10, 4.3, -4.3], [10.219, 5.31, 9, 21.90, 98], [], [4.98], [], [0, 1]],
                              [0.0, 28.8858, None, 4.98, None, 0.5], [7.697077367416805, 35.00081956983293, None, 0.0, None, 0.5],
                              26.886761799748015),
                             ([[1.23, 0.474, 9.516], [10.219, 5.31, 9, 21.90, 98]],
                              [3.74, 28.8858], [4.095893553304333, 35.00081956983293], 30.33397330732285),
                             ([[-10, 10, 4.3, -4.3], [102], [0, 1]],
                              [0.0, 102.0, 0.5], [7.697077367416805, 0.0, 0.5], 36.10779213772596),
                             ([[], [-1], [0, 1]], [None, -1.0, 0.5], [None, 0.0, 0.5], 0.816496580927726),
                             ([[], [1.99], [], [], [0]], [None, 1.99, None, None, 0.0], [None, 0.0, None, None, 0.0], 0.995),
                             ([[2.45, -5.21, -20, 81.92, 41, 1, 0.1123, 1.2], []], [12.8090375, None], [30.64904989938259, None],
                              30.64904989938259)
                         ])
def test_combine_stds(samples, sample_means, sample_stds, expected_std):
    num_sents = [len(l) for l in samples]
    combined_std = utils.combine_stds(sample_stds, sample_means, num_sents)
    assert np.isclose(expected_std, combined_std)


def test_average_arrays():
    n = 4
    shape = (12, 14)
    arrays = [pt.rand(12, 14) for _ in range(n)]
    expected_average = pt.zeros(*shape)
    for array in arrays:
        expected_average += array
    expected_average /= 4

    pt.testing.assert_allclose(utils.average_tensors(arrays), expected_average)

    with pytest.raises(utils.SockeyeError) as e:
        other_shape = (12, 13)
        utils.average_tensors(arrays + [pt.zeros(*other_shape)])
    assert "tensor shapes do not match" == str(e.value)


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


def test_get_num_gpus():
    assert utils.get_num_gpus() >= 0


def _touch_file(fname, compressed: bool, empty: bool) -> str:
    if compressed:
        open_func = gzip.open
    else:
        open_func = open
    with open_func(fname, encoding='utf8', mode='wt') as f:
        if not empty:
            for i in range(10):
                print(str(i), file=f)
    return fname


def test_is_gzip_file():
    with TemporaryDirectory() as temp:
        fname = os.path.join(temp, 'test')
        assert utils.is_gzip_file(_touch_file(fname, compressed=True, empty=True))
        assert utils.is_gzip_file(_touch_file(fname, compressed=True, empty=False))
        assert not utils.is_gzip_file(_touch_file(fname, compressed=False, empty=True))
        assert not utils.is_gzip_file(_touch_file(fname, compressed=False, empty=False))


def test_smart_open_without_suffix():
    with TemporaryDirectory() as temp:
        fname = os.path.join(temp, 'test')
        _touch_file(fname, compressed=True, empty=False)
        with utils.smart_open(fname) as fin:
            assert len(fin.readlines()) == 10
        _touch_file(fname, compressed=False, empty=False)
        with utils.smart_open(fname) as fin:
            assert len(fin.readlines()) == 10


@pytest.mark.parametrize("line_num,line,expected_metrics", [
        (1, "1\tfloat_metric=3.45\tbool_metric=True", {'float_metric': 3.45, 'bool_metric': True}),
        (3, "3\tfloat_metric=1.0\tbool_metric=False", {'float_metric': 1.00, 'bool_metric': False}),
        (3, "3\tfloat_metric=1.0\tnone_metric=None", {'float_metric': 1.00, 'none_metric': None}),
        # line_num and checkpoint are not equal, should fail
        (2, "4\tfloat_metric=1.0\tbool_metric=False", {'float_metric': 1.00, 'bool_metric': False}),
        ])
def test_parse_metrics_line(line_num, line, expected_metrics):
    if line_num == int(line.split('\t')[0]):
        parsed_metrics = utils.parse_metrics_line(line_num, line)
        for k, v in parsed_metrics.items():
            assert isinstance(v, type(expected_metrics[k]))
            assert v == expected_metrics[k]
    else:
        with pytest.raises(utils.SockeyeError) as e:
            parsed_metrics = utils.parse_metrics_line(line_num, line)


def test_write_read_metric_file():
    expected_metrics = [{'float_metric':3.45, 'bool_metric': True},
                       {'float_metric':1.0, 'bool_metric': False}]
    with TemporaryDirectory(prefix="metric_file") as work_dir:
        metric_path = os.path.join(work_dir, "metrics")
        utils.write_metrics_file(expected_metrics, metric_path)
        read_metrics = utils.read_metrics_file(metric_path)

    assert len(read_metrics) == len(expected_metrics)
    assert expected_metrics == read_metrics

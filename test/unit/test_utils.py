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

import sockeye.utils
import numpy as np
import pytest
from sockeye.utils import check_condition, SockeyeError

def test_get_alignments():
    attention_matrix = np.asarray([[0.1, 0.4, 0.5],
                                   [0.2, 0.8, 0.0],
                                   [0.4, 0.4, 0.2]])
    test_cases = [(0.5, [(1, 1)]),
                  (0.8, []),
                  (0.1, [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)])]

    for threshold, expected_alignment in test_cases:
        alignment = list(sockeye.utils.get_alignments(attention_matrix, threshold=threshold))
        assert alignment == expected_alignment


device_params = [([-4, 3, 5], 6, [0, 1, 2, 3, 4, 5]),
                 ([-2, 3, -2, 5], 6, [0, 1, 2, 3, 4, 5]),
                 ([-1], 1, [0]),
                 ([1], 1, [1])]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params)
def test_expand_requested_device_ids(requested_device_ids, num_gpus_available, expected):
    assert set(sockeye.utils._expand_requested_device_ids(requested_device_ids, num_gpus_available)) == set(expected)


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params)
def test_aquire_gpus(tmpdir, requested_device_ids, num_gpus_available, expected):
    with sockeye.utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                                    num_gpus_available=num_gpus_available) as acquired_gpus:
        assert set(acquired_gpus) == set(expected)


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
        sockeye.utils._expand_requested_device_ids(requested_device_ids, num_gpus_available)


@pytest.mark.parametrize("requested_device_ids, num_gpus_available", device_params_expected_exception)
def test_aquire_gpus_exception(tmpdir, requested_device_ids, num_gpus_available):
    with pytest.raises(ValueError):
        with sockeye.utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                                        num_gpus_available=num_gpus_available) as _:
            pass


# Let's assume GPU 1 is locked already
device_params_1_locked = [([-4, 3, 5], 7, [0, 2, 3, 4, 5, 6]),
                          ([-2, 3, -2, 5], 7, [0, 2, 3, 4, 5, 6])]


@pytest.mark.parametrize("requested_device_ids, num_gpus_available, expected", device_params_1_locked)
def test_aquire_gpus_1_locked(tmpdir, requested_device_ids, num_gpus_available, expected):
    gpu_1 = 1
    with sockeye.utils.GpuFileLock([gpu_1], str(tmpdir)) as lock:
        with sockeye.utils.acquire_gpus(requested_device_ids, lock_dir=str(tmpdir),
                          num_gpus_available=num_gpus_available) as acquired_gpus:
            assert set(acquired_gpus) == set(expected)


def test_acquire_gpus_exception_propagation(tmpdir):
    raised_exception = RuntimeError("This exception should be propagated properly.")
    caught_exception = None
    try:
        with sockeye.utils.acquire_gpus([-1, 4, -1], lock_dir=str(tmpdir), num_gpus_available=12) as _:
            raise raised_exception
    except Exception as e:
        caught_exception = e
    assert caught_exception is raised_exception


def test_gpu_file_lock_cleanup(tmpdir):
    gpu_id = 0
    candidates = [gpu_id]

    # Test that the lock files get created and clean up
    with sockeye.utils.GpuFileLock(candidates, str(tmpdir)) as lock:
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
        with sockeye.utils.GpuFileLock(gpu_ids, str(tmpdir)) as lock:
            raise raised_exception
    except Exception as e:
        caught_exception = e
    assert caught_exception is raised_exception


def test_gpu_file_lock_locking(tmpdir):
    # the second time we try to acquire a lock for the same device we should not succeed
    gpu_id = 0
    candidates = [gpu_id]

    with sockeye.utils.GpuFileLock(candidates, str(tmpdir)) as lock_inner:
        assert lock_inner == 0
        with sockeye.utils.GpuFileLock(candidates, str(tmpdir)) as lock_outer:
            assert lock_outer is None


def test_gpu_file_lock_permission_exception(tmpdir):
    with pytest.raises(PermissionError):
        tmpdir = tmpdir.mkdir("sub")
        # remove permissions
        tmpdir.chmod(0)

        with sockeye.utils.GpuFileLock([0], str(tmpdir)) as lock:
            assert False, "We expect to raise an exception when aquiring the lock and never reach this code."


def test_check_condition_true():
    check_condition(1 == 1, "Nice")


def test_check_condition_false():
    with pytest.raises(SockeyeError) as e:
        check_condition(1 == 2, "Wrong")
    assert "Wrong"  == str(e.value)

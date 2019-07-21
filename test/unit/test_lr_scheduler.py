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

import pytest

import numpy as np

from sockeye import lr_scheduler

@pytest.mark.parametrize('update_interval', [1, 2])
def test_inv_sqrt_decay_scheduler(update_interval):
    warmup = 3
    scheduler = lr_scheduler.get_lr_scheduler('inv-sqrt-decay',
                                              learning_rate_reduce_factor=0,
                                              learning_rate_reduce_num_not_improved=0,
                                              learning_rate_warmup=warmup,
                                              update_interval=update_interval,
                                              max_updates=10)
    scheduler.base_lr = 1

    def alternate_implementation(t):
        # Reference formula from Transformer paper
        return min((t / update_interval)**-0.5, (t / update_interval) * (warmup)**-1.5)

    expected_schedule = [alternate_implementation(t) for t in range(1, 11)]

    actual_schedule = [scheduler(t) for t in range(1, 11)]

    assert np.isclose(expected_schedule, actual_schedule).all()



def test_linear_decay_scheduler():
    scheduler = lr_scheduler.get_lr_scheduler('linear-decay',
                                              learning_rate_reduce_factor=0,
                                              learning_rate_reduce_num_not_improved=0,
                                              learning_rate_warmup=3,
                                              update_interval=2,
                                              max_updates=10)
    scheduler.base_lr = 1
    # Warmup term * decay term
    expected_schedule = [
        (1/3) * (9/10),
        (2/3) * (8/10),
        (3/3) * (7/10),
        (3/3) * (6/10),
        (3/3) * (5/10),
        (3/3) * (4/10),
        (3/3) * (3/10),
        (3/3) * (2/10),
        (3/3) * (1/10),
        (3/3) * (0/10),
    ]
    actual_schedule = [scheduler(t) for t in range(1, 11)]
    assert np.isclose(expected_schedule, actual_schedule).all()


@pytest.mark.parametrize('scheduler_type, expected_instance',
                         [('inv-sqrt-decay', lr_scheduler.LearningRateSchedulerInvSqrtDecay),
                          ('linear-decay', lr_scheduler.LearningRateSchedulerLinearDecay),
                          ('plateau-reduce', lr_scheduler.LearningRateSchedulerPlateauReduce)])
def test_get_lr_scheduler(scheduler_type, expected_instance):
    scheduler = lr_scheduler.get_lr_scheduler(scheduler_type,
                                              learning_rate_reduce_factor=0.5,
                                              learning_rate_reduce_num_not_improved=16,
                                              learning_rate_warmup=1000,
                                              update_interval=2,
                                              max_updates=10000)
    assert isinstance(scheduler, expected_instance)


def test_get_lr_scheduler_no_reduce():
    scheduler = lr_scheduler.get_lr_scheduler('plateau-reduce',
                                              learning_rate_reduce_factor=1.0,
                                              learning_rate_reduce_num_not_improved=16)
    assert scheduler is None

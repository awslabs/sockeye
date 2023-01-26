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

import pytest

import numpy as np
import torch

from sockeye import lr_scheduler


@pytest.mark.parametrize('learning_rate_warmup', [1, 20])
def test_inv_sqrt_decay_scheduler(learning_rate_warmup):
    # Sockeye implementation
    scheduler_class, scheduler_kwargs = lr_scheduler.get_lr_scheduler(scheduler_type='inv-sqrt-decay',
                                                                      base_learning_rate=1,
                                                                      learning_rate_reduce_factor=0,
                                                                      learning_rate_reduce_num_not_improved=0,
                                                                      learning_rate_warmup=learning_rate_warmup,
                                                                      max_updates=10)
    assert scheduler_class is not None
    scheduler = scheduler_class(optimizer=torch.optim.Adam(params=[torch.zeros(0)]), **scheduler_kwargs)
    assert isinstance(scheduler, lr_scheduler.LearningRateSchedulerInvSqrtDecay)

    # Reference formula from Transformer paper
    alternate_implementation = lambda t: min(t**-0.5, t * learning_rate_warmup**-1.5)

    expected_schedule = [alternate_implementation(t) for t in range(1, 11)]
    actual_schedule = []
    for t in range(1, 11):
        # Check manual and automatic step numbers
        scheduler.step(t if t % 2 else None)
        actual_schedule.append(scheduler.get_last_lr()[0])
    assert np.isclose(expected_schedule, actual_schedule).all()


def test_linear_decay_scheduler():
    # Sockeye implementation
    scheduler_class, scheduler_kwargs = lr_scheduler.get_lr_scheduler(scheduler_type='linear-decay',
                                                                      base_learning_rate=1,
                                                                      learning_rate_reduce_factor=0,
                                                                      learning_rate_reduce_num_not_improved=0,
                                                                      learning_rate_warmup=3,
                                                                      max_updates=10)
    assert scheduler_class is not None
    scheduler = scheduler_class(optimizer=torch.optim.Adam(params=[torch.zeros(0)]), **scheduler_kwargs)
    assert isinstance(scheduler, lr_scheduler.LearningRateSchedulerLinearDecay)

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

    actual_schedule = []
    for t in range(1, 11):
        # Check manual and automatic step numbers
        scheduler.step(t if t % 2 else None)
        actual_schedule.append(scheduler.get_last_lr()[0])
    assert np.isclose(expected_schedule, actual_schedule).all()


@pytest.mark.parametrize('scheduler_type, expected_instance',
                         [('none', None),
                          ('inv-sqrt-decay', lr_scheduler.LearningRateSchedulerInvSqrtDecay),
                          ('linear-decay', lr_scheduler.LearningRateSchedulerLinearDecay),
                          ('plateau-reduce', lr_scheduler.LearningRateSchedulerPlateauReduce)])
def test_get_lr_scheduler(scheduler_type, expected_instance):
    scheduler_class, scheduler_kwargs = lr_scheduler.get_lr_scheduler(scheduler_type=scheduler_type,
                                                                      base_learning_rate=1,
                                                                      learning_rate_reduce_factor=0.5,
                                                                      learning_rate_reduce_num_not_improved=16,
                                                                      learning_rate_warmup=1000,
                                                                      max_updates=10000)
    if expected_instance is None:
        assert scheduler_class is None
        assert scheduler_kwargs == {}
    else:
        assert scheduler_class is not None
        scheduler = scheduler_class(optimizer=torch.optim.Adam(params=[torch.zeros(0)]), **scheduler_kwargs)
        assert isinstance(scheduler, expected_instance)


def test_get_lr_scheduler_no_reduce():
    scheduler_class, scheduler_kwargs = lr_scheduler.get_lr_scheduler(scheduler_type='plateau-reduce',
                                                                      base_learning_rate=1,
                                                                      learning_rate_reduce_factor=1.0,
                                                                      learning_rate_reduce_num_not_improved=16)
    assert scheduler_class is None
    assert scheduler_kwargs == {}

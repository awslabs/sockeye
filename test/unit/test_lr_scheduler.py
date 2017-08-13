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

from sockeye import lr_scheduler
from sockeye.lr_scheduler import LearningRateSchedulerFixedStep, LearningRateSchedulerInvSqrtT, LearningRateSchedulerInvT


def test_lr_scheduler():
    updates_per_checkpoint = 13
    half_life_num_checkpoints = 3

    schedulers = [LearningRateSchedulerInvT(updates_per_checkpoint, half_life_num_checkpoints),
                  LearningRateSchedulerInvSqrtT(updates_per_checkpoint, half_life_num_checkpoints)]
    for scheduler in schedulers:
        scheduler.base_lr = 1.0
        # test correct half-life:
        assert scheduler(updates_per_checkpoint * half_life_num_checkpoints) == pytest.approx(0.5)


def test_fixed_step_lr_scheduler():
    # Parse schedule string
    schedule_str = "0.5:16,0.25:8"
    schedule = LearningRateSchedulerFixedStep.parse_schedule_str(schedule_str)
    assert schedule == [(0.5, 16), (0.25, 8)]
    # Check learning rate steps
    updates_per_checkpoint = 2
    scheduler = LearningRateSchedulerFixedStep(schedule, updates_per_checkpoint)
    t = 0
    for _ in range(16):
        t += 1
        assert scheduler(t) == 0.5
        if t % 2 == 0:
            scheduler.new_evaluation_result(False)
    assert scheduler(t) == 0.25
    for _ in range(8):
        t += 1
        assert scheduler(t) == 0.25
        if t % 2 == 0:
            scheduler.new_evaluation_result(False)


@pytest.mark.parametrize("scheduler_type, reduce_factor, expected_instance",
                         [("fixed-rate-inv-sqrt-t", 1.0, lr_scheduler.LearningRateSchedulerInvSqrtT),
                          ("fixed-rate-inv-t", 1.0, lr_scheduler.LearningRateSchedulerInvT),
                          ("plateau-reduce", 0.5, lr_scheduler.LearningRateSchedulerPlateauReduce)])
def test_get_lr_scheduler(scheduler_type, reduce_factor, expected_instance):
    scheduler = lr_scheduler.get_lr_scheduler(scheduler_type,
                                              updates_per_checkpoint=4,
                                              learning_rate_half_life=2,
                                              learning_rate_reduce_factor=reduce_factor,
                                              learning_rate_reduce_num_not_improved=16)
    assert isinstance(scheduler, expected_instance)


def test_get_lr_scheduler_no_reduce():
    scheduler = lr_scheduler.get_lr_scheduler("plateau-reduce",
                                              updates_per_checkpoint=4,
                                              learning_rate_half_life=2,
                                              learning_rate_reduce_factor=1.0,
                                              learning_rate_reduce_num_not_improved=16)
    assert scheduler is None

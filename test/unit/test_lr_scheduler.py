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
from sockeye.lr_scheduler import LearningRateSchedulerInvSqrtT, LearningRateSchedulerInvT


def test_lr_scheduler():
    updates_per_epoch = 13
    half_life_num_epochs = 3

    schedulers = [LearningRateSchedulerInvT(updates_per_epoch, half_life_num_epochs),
                  LearningRateSchedulerInvSqrtT(updates_per_epoch, half_life_num_epochs)]
    for scheduler in schedulers:
        scheduler.base_lr = 1.0
        # test correct half-life:

        assert scheduler(updates_per_epoch * half_life_num_epochs) == pytest.approx(0.5)


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

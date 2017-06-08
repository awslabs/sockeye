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

from sockeye.lr_scheduler import LearningRateSchedulerInvSqrtT, LearningRateSchedulerInvT
import pytest


def test_lr_scheduler():
    updates_per_epoch = 13
    half_life_num_epochs = 3

    schedulers = [LearningRateSchedulerInvT(updates_per_epoch, half_life_num_epochs),
                  LearningRateSchedulerInvSqrtT(updates_per_epoch, half_life_num_epochs)]
    for scheduler in schedulers:
        scheduler.base_lr = 1.0
        # test correct half-life:

        assert scheduler(updates_per_epoch * half_life_num_epochs) == pytest.approx(0.5)

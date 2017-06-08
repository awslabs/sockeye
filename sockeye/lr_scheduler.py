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

import logging
from math import sqrt
from typing import Optional

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    def new_evaluation_result(self, has_improved: bool):
        pass

    def __call__(self, num_updates):
        pass


class LearningRateSchedulerInvSqrtT(LearningRateScheduler):
    """
    Learning rate schedule: lr / sqrt(1 + factor * t).
    Note: The factor is calculated from the half life of the learning rate.

    :param updates_per_checkpoint: Number of batches between checkpoints.
    :param half_life: Half life of the learning rate in number of checkpoints.
    """

    def __init__(self, updates_per_checkpoint: int, half_life: int) -> None:
        assert updates_per_checkpoint > 0, "updates_per_checkpoint needs to be > 0."
        assert half_life > 0, "half_life needs to be > 0."
        # Note: will be overwritten by optimizer  in mxnet
        self.base_lr = None
        # 0.5 base_lr = base_lr * sqrt(1 + T * factor)
        # then factor = 3 ./ T, with T = half_life * updates_per_checkpoint
        self.factor = 3. / (half_life * updates_per_checkpoint)
        self.t_last_log = -1
        self.log_every_t = int(half_life * updates_per_checkpoint)

    def __call__(self, num_updates: int):
        lr = self.base_lr / sqrt(1 + num_updates * self.factor)

        # Note: this method is called once per parameter for the same t. Making sure to just log once.
        if num_updates > self.t_last_log and num_updates % self.log_every_t == 0:
            logger.info("Learning rate currently at %1.2e", lr)
            self.t_last_log = num_updates

        return lr


class LearningRateSchedulerInvT(LearningRateScheduler):
    """
    Learning rate schedule: lr / (1 + factor * t).
    Note: The factor is calculated from the half life of the learning rate.

    :param updates_per_checkpoint: Number of batches between checkpoints.
    :param half_life: Half life of the learning rate in number of checkpoints.
    """

    def __init__(self, updates_per_checkpoint: int, half_life: int) -> None:
        assert updates_per_checkpoint > 0, "updates_per_checkpoint needs to be > 0."
        assert half_life > 0, "half_life needs to be > 0."
        # Note: will be overwritten by optimizer
        self.base_lr = None
        # 0.5 base_lr = base_lr * (1 + T * factor)
        # then factor = 1 ./ T, with T = half_life * updates_per_checkpoint
        self.factor = 1. / (half_life * updates_per_checkpoint)
        self.t_last_log = -1
        self.log_every_t = int(half_life * updates_per_checkpoint)

    def __call__(self, num_updates: int):
        lr = self.base_lr / (1 + num_updates * self.factor)

        # Note: this method is called once per parameter for the same t. Making sure to just log once.
        if num_updates > self.t_last_log and num_updates % self.log_every_t == 0:
            logger.info("Learning rate currently at %1.2e", lr)
            self.t_last_log = num_updates

        return lr


class LearningRateSchedulerPlateauReduce(LearningRateScheduler):
    """
    Lower the learning rate as soon as the validation score plateaus.

    :param reduce_factor: Factor to reduce learning rate with.
    :param reduce_num_not_improved: Number of checkpoints with no improvement after which learning rate is reduced.
    """

    def __init__(self, reduce_factor: float, reduce_num_not_improved: int) -> None:
        self.reduce_factor = reduce_factor
        self.reduce_num_not_improved = reduce_num_not_improved
        self.num_not_improved = 0
        self.logger = logging.getLogger("LearningRateSchedulerPlateauReduce")
        # Note: will be overwritten by optimizer in mxnet
        self.base_lr = None  # type: float
        self.lr = None  # type: float
        logger.info("Will reduce the learning rate by a factor of %.2f whenever"
                    " the validation score doesn't improve %d times.",
                    reduce_factor, reduce_num_not_improved)

    def new_evaluation_result(self, has_improved: bool):
        if self.lr is None:
            assert self.base_lr is not None
            self.lr = self.base_lr
        if has_improved:
            self.num_not_improved = 0
        else:
            self.num_not_improved += 1
            if self.num_not_improved >= self.reduce_num_not_improved:
                self.lr *= self.reduce_factor
                self.logger.info("Validation score hasn't improved for %d checkpoints, "
                                 "lowering learning rate to %1.2e", self.num_not_improved, self.lr)
                self.num_not_improved = 0

    def __call__(self, t):
        if self.lr is None:
            assert self.base_lr is not None
            self.lr = self.base_lr
        return self.lr

    def __repr__(self):
        return "LearningRateSchedulerPlateauReduce(reduce_factor=%.2f, " \
               "reduce_num_not_improved=%d)" % (self.reduce_factor, self.num_not_improved)


def get_lr_scheduler(scheduler_type: str,
                     updates_per_checkpoint: int,
                     learning_rate_half_life: int,
                     learning_rate_reduce_factor: float,
                     learning_rate_reduce_num_not_improved: int) -> Optional[LearningRateScheduler]:
    """
    Returns a learning rate scheduler.

    :param scheduler_type: Scheduler type.
    :param updates_per_checkpoint: Number of batches between checkpoints.
    :param learning_rate_half_life: Half life of the learning rate in number of checkpoints.
    :param learning_rate_reduce_factor: Factor to reduce learning rate with.
    :param learning_rate_reduce_num_not_improved: Number of checkpoints with no improvement after which learning rate is
           reduced.
    :raises: ValueError if unknown scheduler_type
    :return: Learning rate scheduler.
    """
    if scheduler_type is None:
        return None
    if scheduler_type == "fixed-rate-inv-sqrt-t":
        return LearningRateSchedulerInvSqrtT(updates_per_checkpoint, learning_rate_half_life)
    elif scheduler_type == "fixed-rate-inv-t":
        return LearningRateSchedulerInvT(updates_per_checkpoint, learning_rate_half_life)
    elif scheduler_type == "plateau-reduce":
        assert learning_rate_reduce_factor is not None, "learning_rate_reduce_factor needed for plateau-reduce " \
                                                        "scheduler"
        assert learning_rate_reduce_num_not_improved is not None, "learning_rate_reduce_num_not_improved needed for " \
                                                                  "plateau-reduce scheduler"
        return LearningRateSchedulerPlateauReduce(learning_rate_reduce_factor, learning_rate_reduce_num_not_improved)
    else:
        raise ValueError("Unknown learning rate scheduler type %s." % scheduler_type)

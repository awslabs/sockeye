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
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class LearningRateScheduler:

    def __init__(self, warmup: int = 0) -> None:
        self.base_lr = None  # Note: will be overwritten by MXNet optimizer
        check_condition(warmup >= 0, "warmup needs to be >= 0.")
        self.warmup = warmup
        self.log_warmup_every_t = self.warmup // 10
        self.last_warmup_log = -1

    def new_evaluation_result(self, has_improved: bool):
        pass

    def __call__(self, num_updates):
        pass

    def _warmup(self, num_updates):
        """
        Returns linearly increasing fraction of base_lr.
        """
        assert self.base_lr is not None
        if not self.warmup:
            return self.base_lr
        fraction = (num_updates + 1) * self.base_lr / (self.warmup + 1)
        if num_updates > self.last_warmup_log and num_updates % self.log_warmup_every_t == 0:
            self.last_warmup_log = num_updates
            logger.info("Learning rate %.0f%% warmed up", fraction * 100)
        return fraction


class LearningRateSchedulerInvSqrtT(LearningRateScheduler):
    """
    Learning rate schedule: lr / sqrt(1 + factor * t).
    Note: The factor is calculated from the half life of the learning rate.

    :param updates_per_checkpoint: Number of batches between checkpoints.
    :param half_life: Half life of the learning rate in number of checkpoints.
    :param warmup: Number of (linear) learning rate increases to warm-up.
    """

    def __init__(self, updates_per_checkpoint: int, half_life: int, warmup: int = 0) -> None:
        super().__init__(warmup)
        check_condition(updates_per_checkpoint > 0, "updates_per_checkpoint needs to be > 0.")
        check_condition(half_life > 0, "half_life needs to be > 0.")
        # 0.5 base_lr = base_lr * sqrt(1 + T * factor)
        # then factor = 3 ./ T, with T = half_life * updates_per_checkpoint
        self.factor = 3. / (half_life * updates_per_checkpoint)
        self.t_last_log = -1
        self.log_every_t = int(half_life * updates_per_checkpoint)

    def __call__(self, num_updates: int):
        lr = min(self.base_lr / sqrt(1 + num_updates * self.factor), self._warmup(num_updates) if self.warmup > 0 else 99999)
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

    def __init__(self, updates_per_checkpoint: int, half_life: int, warmup: int = 0) -> None:
        super().__init__(warmup)
        check_condition(updates_per_checkpoint > 0, "updates_per_checkpoint needs to be > 0.")
        check_condition(half_life > 0, "half_life needs to be > 0.")

        # 0.5 base_lr = base_lr * (1 + T * factor)
        # then factor = 1 ./ T, with T = half_life * updates_per_checkpoint
        self.factor = 1. / (half_life * updates_per_checkpoint)
        self.t_last_log = -1
        self.log_every_t = int(half_life * updates_per_checkpoint)

    def __call__(self, num_updates: int):
        lr = min(self.base_lr / (1 + num_updates * self.factor), self._warmup(num_updates) if self.warmup > 0 else 99999)
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

    def __init__(self, reduce_factor: float, reduce_num_not_improved: int, warmup: int = 0) -> None:
        super().__init__(warmup)
        check_condition(0.0 < reduce_factor <= 1, "reduce_factor should be in ]0,1].")
        self.reduce_factor = reduce_factor
        self.reduce_num_not_improved = reduce_num_not_improved
        self.num_not_improved = 0

        self.lr = None  # type: float
        self.t_last_log = -1
        self.warmed_up = not self.warmup > 0
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
            if self.num_not_improved >= self.reduce_num_not_improved and self.reduce_factor < 1.0 and self.warmed_up:
                old_lr = self.lr
                self.lr *= self.reduce_factor
                logger.info("%d checkpoints since improvement or rate scaling, "
                            "lowering learning rate: %1.2e -> %1.2e", self.num_not_improved, old_lr, self.lr)
                self.num_not_improved = 0

    def __call__(self, t):
        if self.lr is None:
            assert self.base_lr is not None
            self.lr = self.base_lr
        lr = self._warmup(t) if self.warmup > 0 and t <= self.warmup else self.lr
        if t == self.warmup:
            self.warmed_up = True
        return lr

    def __repr__(self):
        return "LearningRateSchedulerPlateauReduce(reduce_factor=%.2f, " \
               "reduce_num_not_improved=%d)" % (self.reduce_factor, self.num_not_improved)


def get_lr_scheduler(scheduler_type: str,
                     updates_per_checkpoint: int,
                     learning_rate_half_life: int,
                     learning_rate_reduce_factor: float,
                     learning_rate_reduce_num_not_improved: int,
                     learning_rate_warmup: Optional[int] = 0) -> Optional[LearningRateScheduler]:
    """
    Returns a learning rate scheduler.

    :param scheduler_type: Scheduler type.
    :param updates_per_checkpoint: Number of batches between checkpoints.
    :param learning_rate_half_life: Half life of the learning rate in number of checkpoints.
    :param learning_rate_reduce_factor: Factor to reduce learning rate with.
    :param learning_rate_reduce_num_not_improved: Number of checkpoints with no improvement after which learning rate is
           reduced.
    :param learning_rate_warmup: Number of batches that the learning rate is linearly increased.
    :raises: ValueError if unknown scheduler_type
    :return: Learning rate scheduler.
    """
    if scheduler_type is None:
        return None
    if scheduler_type == "fixed-rate-inv-sqrt-t":
        return LearningRateSchedulerInvSqrtT(updates_per_checkpoint, learning_rate_half_life, learning_rate_warmup)
    elif scheduler_type == "fixed-rate-inv-t":
        return LearningRateSchedulerInvT(updates_per_checkpoint, learning_rate_half_life, learning_rate_warmup)
    elif scheduler_type == "plateau-reduce":
        check_condition(learning_rate_reduce_factor is not None,
                        "learning_rate_reduce_factor needed for plateau-reduce scheduler")
        check_condition(learning_rate_reduce_num_not_improved is not None,
                        "learning_rate_reduce_num_not_improved needed for plateau-reduce scheduler")
        if learning_rate_reduce_factor >= 1.0:
            logger.warning("Not using plateau-reduce learning rate scheduling: learning_rate_reduce_factor == 1.0")
            return None
        return LearningRateSchedulerPlateauReduce(learning_rate_reduce_factor, learning_rate_reduce_num_not_improved,
                                                  learning_rate_warmup)
    else:
        raise ValueError("Unknown learning rate scheduler type %s." % scheduler_type)

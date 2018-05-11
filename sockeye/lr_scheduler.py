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
from typing import List, Optional, Tuple
import sockeye.constants as C
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class LearningRateScheduler:

    def __init__(self, warmup: int = 0) -> None:
        self.base_lr = None  # Note: will be overwritten by MXNet optimizer
        check_condition(warmup >= 0, "warmup needs to be >= 0.")
        self.warmup = warmup
        self.log_warmup_every_t = self.warmup // 10
        self.last_warmup_log = -1

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
            logger.info("Learning rate warmup: %3.0f%%", fraction/self.base_lr * 100.0)
        return fraction


class AdaptiveLearningRateScheduler(LearningRateScheduler):
    """
    Learning rate scheduler that implements `new_evaluation_result` and accordingly adaptively adjust the  learning
    rate.
    """

    def new_evaluation_result(self, has_improved: bool) -> bool:
        """
        Returns true if the parameters should be reset to the ones with the best validation score.

        :param has_improved: Whether the model improved on held-out validation data.
        :return: True if parameters should be reset to the ones with best validation score.
        """
        return False


class LearningRateSchedulerFixedStep(AdaptiveLearningRateScheduler):
    """
    Use a fixed schedule of learning rate steps: lr_1 for N steps, lr_2 for M steps, etc.

    :param schedule: List of learning rate step tuples in the form (rate, num_updates).
    :param updates_per_checkpoint: Updates per checkpoint.
    """

    def __init__(self, schedule: List[Tuple[float, int]], updates_per_checkpoint: int) -> None:
        super().__init__()
        check_condition(all(num_updates > 0 for (_, num_updates) in schedule),
                        "num_updates for each step should be > 0.")
        check_condition(all(num_updates % updates_per_checkpoint == 0 for (_, num_updates) in schedule),
                        "num_updates for each step should be divisible by updates_per_checkpoint.")
        self.schedule = schedule
        self.current_step = 0
        self.current_rate = 0.
        self.current_step_num_updates = 0
        self.current_step_started_at = 0
        self.next_step_at = 0
        self.latest_t = 0
        self._update_rate(self.current_step)

    def new_evaluation_result(self, has_improved: bool) -> bool:
        """
        Returns true if the parameters should be reset to the ones with the best validation score.

        :param has_improved: Whether the model improved on held-out validation data.
        :return: True if parameters should be reset to the ones with best validation score.
        """
        logger.info("Checkpoint learning rate: %1.2e (%d/%d updates)",
                    self.current_rate,
                    self.latest_t - self.current_step_started_at,
                    self.current_step_num_updates)
        if self.latest_t >= self.next_step_at:
            self.current_step += 1
            self._update_rate(self.current_step)
        return False

    def _update_rate(self, step: int):
        if self.current_step < len(self.schedule):
            self.current_rate, self.current_step_num_updates = self.schedule[step]
            self.current_step_started_at = self.latest_t
            self.next_step_at += self.current_step_num_updates
            logger.info("Changing learning rate to %1.2e for %d updates",
                        self.current_rate,
                        self.current_step_num_updates)

    def __call__(self, t: int):
        self.latest_t = max(t, self.latest_t)
        return self.current_rate

    @staticmethod
    def parse_schedule_str(schedule_str: str) -> List[Tuple[float, int]]:
        """
        Parse learning schedule string.

        :param schedule_str: String in form rate1:num_updates1[,rate2:num_updates2,...]
        :return: List of tuples (learning_rate, num_updates).
        """
        schedule = list()
        for step in schedule_str.split(","):
            rate, num_updates = step.split(":")
            schedule.append((float(rate), int(num_updates)))
        return schedule


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
        lr = min(self.base_lr / sqrt(1 + num_updates * self.factor),
                 self._warmup(num_updates) if self.warmup > 0 else C.LARGE_POSITIVE_VALUE)
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
        lr = min(self.base_lr / (1 + num_updates * self.factor),
                 self._warmup(num_updates) if self.warmup > 0 else C.LARGE_POSITIVE_VALUE)
        # Note: this method is called once per parameter for the same t. Making sure to just log once.
        if num_updates > self.t_last_log and num_updates % self.log_every_t == 0:
            logger.info("Learning rate currently at %1.2e", lr)
            self.t_last_log = num_updates

        return lr


class LearningRateSchedulerPlateauReduce(AdaptiveLearningRateScheduler):
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

    def new_evaluation_result(self, has_improved: bool) -> bool:
        """
        Returns true if the parameters should be reset to the ones with the best validation score.

        :param has_improved: Whether the model improved on held-out validation data.
        :return: True if parameters should be reset to the ones with best validation score.
        """
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
                return True
        return False

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
                     learning_rate_schedule: Optional[List[Tuple[float, int]]] = None,
                     learning_rate_warmup: Optional[int] = 0) -> Optional[LearningRateScheduler]:
    """
    Returns a learning rate scheduler.

    :param scheduler_type: Scheduler type.
    :param updates_per_checkpoint: Number of batches between checkpoints.
    :param learning_rate_half_life: Half life of the learning rate in number of checkpoints.
    :param learning_rate_reduce_factor: Factor to reduce learning rate with.
    :param learning_rate_reduce_num_not_improved: Number of checkpoints with no improvement after which learning rate is
           reduced.
    :param learning_rate_schedule: Optional fixed learning rate schedule.
    :param learning_rate_warmup: Number of batches that the learning rate is linearly increased.
    :raises: ValueError if unknown scheduler_type
    :return: Learning rate scheduler.
    """
    check_condition(learning_rate_schedule is None or scheduler_type == C.LR_SCHEDULER_FIXED_STEP,
                    "Learning rate schedule can only be used with '%s' learning rate scheduler."
                    % C.LR_SCHEDULER_FIXED_STEP)
    if scheduler_type is None:
        return None
    if scheduler_type == C.LR_SCHEDULER_FIXED_RATE_INV_SQRT_T:
        return LearningRateSchedulerInvSqrtT(updates_per_checkpoint, learning_rate_half_life, learning_rate_warmup)
    elif scheduler_type == C.LR_SCHEDULER_FIXED_RATE_INV_T:
        return LearningRateSchedulerInvT(updates_per_checkpoint, learning_rate_half_life, learning_rate_warmup)
    elif scheduler_type == C.LR_SCHEDULER_FIXED_STEP:
        check_condition(learning_rate_schedule is not None,
                        "learning_rate_schedule needed for %s scheduler" % C.LR_SCHEDULER_FIXED_STEP)
        return LearningRateSchedulerFixedStep(learning_rate_schedule, updates_per_checkpoint)
    elif scheduler_type == C.LR_SCHEDULER_PLATEAU_REDUCE:
        check_condition(learning_rate_reduce_factor is not None,
                        "learning_rate_reduce_factor needed for %s scheduler" % C.LR_SCHEDULER_PLATEAU_REDUCE)
        check_condition(learning_rate_reduce_num_not_improved is not None,
                        "learning_rate_reduce_num_not_improved needed for %s scheduler" % C.LR_SCHEDULER_PLATEAU_REDUCE)
        if learning_rate_reduce_factor >= 1.0:
            logger.warning("Not using %s learning rate scheduling: learning_rate_reduce_factor == 1.0"
                           % C.LR_SCHEDULER_PLATEAU_REDUCE)
            return None
        return LearningRateSchedulerPlateauReduce(learning_rate_reduce_factor, learning_rate_reduce_num_not_improved,
                                                  learning_rate_warmup)
    else:
        raise ValueError("Unknown learning rate scheduler type %s." % scheduler_type)

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

import logging
from math import sqrt
from typing import Optional
import sockeye.constants as C
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class LearningRateScheduler:

    def __init__(self, warmup: int = 0, t_scale: float = 1.0) -> None:
        self.base_lr = None  # Note: will be overwritten by MXNet optimizer
        check_condition(warmup >= 0, "warmup needs to be >= 0.")
        self.warmup = warmup
        self.t_scale = t_scale
        self.lr = None  # type: Optional[float]

    def __call__(self, t):
        pass

    def _warmup(self, scaled_t):
        """
        Returns linearly increasing fraction of base_lr.  Here t is not scaled
        by t_scale, as individual schedulers should scale t prior to calling
        this method.
        """
        assert self.base_lr is not None
        if not self.warmup:
            return self.base_lr
        return self.base_lr * min(1.0, scaled_t / self.warmup)


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


class LearningRateSchedulerInvSqrtDecay(LearningRateScheduler):
    """
    Learning rate schedule: lr / sqrt(max(t, warmup_steps)).

    This is the schedule used by Vaswani et al. in the Transformer paper
    (https://arxiv.org/pdf/1706.03762.pdf)

    :param warmup: Number of initial updates during which the learning rate
                   linearly increases.
    """

    def __call__(self, t: int):
        # Time scale
        scaled_t = t * self.t_scale
        # Warmup
        warm_lr = self._warmup(scaled_t)
        # Avoid square root of zero
        warmup_steps = max(1, self.warmup)
        # Warmup first N steps, then decay
        lr = warm_lr / sqrt(max(scaled_t, warmup_steps))
        # For this scheduler, `self.lr` represents the last seen lr and is only
        # used for logging purposes.
        self.lr = lr

        return lr


class LearningRateSchedulerLinearDecay(LearningRateScheduler):
    """
    Learning rate schedule: lr * (1 - t / total_steps)
    Step grows until it reaches decay_steps then remains constant.

    This is the schedule used by Devlin et al. in the BERT paper
    (https://arxiv.org/pdf/1810.04805.pdf).

    :param max_updates: Number of total training updates.  The learning rate
                        linearly decays to zero over this period.
    :param warmup: Number of initial updates during which the learning rate
                   linearly increases.
    """

    def __init__(self, total_steps: int, warmup: int = 0, t_scale: float = 1.0) -> None:
        super().__init__(warmup, t_scale)
        check_condition(total_steps >= 0, "total_steps need to be >= 0.")
        self.total_steps = total_steps

    def __call__(self, t: int):
        # Time scale
        scaled_t = t * self.t_scale
        # Warmup
        warm_lr = self._warmup(scaled_t)
        # Linear decay
        bounded_t = min(max(scaled_t, 1), self.total_steps)
        lr = warm_lr * (1 - bounded_t / self.total_steps)
        # For this scheduler, `self.lr` represents the last seen lr and is only
        # used for logging purposes.
        self.lr = lr
        return lr


class LearningRateSchedulerPlateauReduce(AdaptiveLearningRateScheduler):
    """
    Lower the learning rate as soon as the validation score plateaus.

    :param reduce_factor: Factor to reduce learning rate with.
    :param reduce_num_not_improved: Number of checkpoints with no improvement after which learning rate is reduced.
    """

    def __init__(self, reduce_factor: float, reduce_num_not_improved: int, warmup: int = 0) -> None:
        super().__init__(warmup)
        check_condition(0.0 < reduce_factor < 1, "reduce_factor should be between (0, 1).")
        self.reduce_factor = reduce_factor
        self.reduce_num_not_improved = reduce_num_not_improved
        self.num_not_improved = 0

        self.lr = None  # type: Optional[float]
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
                     learning_rate_t_scale: float,
                     learning_rate_reduce_factor: float,
                     learning_rate_reduce_num_not_improved: int,
                     learning_rate_warmup: int = 0,
                     max_updates: Optional[int] = None) -> Optional[LearningRateScheduler]:
    """
    Returns a learning rate scheduler.

    :param scheduler_type: Scheduler type.
    :param learning_rate_reduce_factor: Factor to reduce learning rate with.
    :param learning_rate_t_scale: Scaling factor for step number.
    :param learning_rate_reduce_num_not_improved: Number of checkpoints with no
           improvement after which learning rate is reduced.
    :param learning_rate_warmup: Number of initial updates during which the
                                 learning rate linearly increases.
    :param max_updates: Number of total training updates.

    :raises: ValueError if unknown scheduler_type

    :return: Learning rate scheduler.
    """
    if scheduler_type is None or scheduler_type == C.LR_SCHEDULER_NONE:
        return None
    if scheduler_type == C.LR_SCHEDULER_INV_SQRT_DECAY:
        return LearningRateSchedulerInvSqrtDecay(warmup=learning_rate_warmup, t_scale=learning_rate_t_scale)
    if scheduler_type == C.LR_SCHEDULER_LINEAR_DECAY:
        check_condition(max_updates is not None,
                        "The total number of training updates (--max-updates) must be specified when using the linear "
                        "decay learning rate scheduler.")
        return LearningRateSchedulerLinearDecay(total_steps=max_updates,
                                                warmup=learning_rate_warmup,
                                                t_scale=learning_rate_t_scale)
    if scheduler_type == C.LR_SCHEDULER_PLATEAU_REDUCE:
        check_condition(learning_rate_reduce_factor is not None,
                        "learning_rate_reduce_factor needed for %s scheduler" % C.LR_SCHEDULER_PLATEAU_REDUCE)
        check_condition(learning_rate_reduce_num_not_improved is not None,
                        "learning_rate_reduce_num_not_improved needed for %s scheduler" % C.LR_SCHEDULER_PLATEAU_REDUCE)
        if learning_rate_reduce_factor >= 1.0:
            logger.warning("Not using %s learning rate scheduling: learning_rate_reduce_factor == 1.0",
                           C.LR_SCHEDULER_PLATEAU_REDUCE)
            return None
        return LearningRateSchedulerPlateauReduce(learning_rate_reduce_factor, learning_rate_reduce_num_not_improved,
                                                  learning_rate_warmup)
    raise ValueError("Unknown learning rate scheduler type %s." % scheduler_type)

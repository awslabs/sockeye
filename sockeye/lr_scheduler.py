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
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

import sockeye.constants as C
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """
    Learning rate scheduler base class. A scheduler operates on a specified
    optimizer instance using an API that is compatible with PyTorch and
    DeepSpeed. See https://pytorch.org/docs/stable/optim.html for more
    information on PyTorch optimizers, learning rate schedulers, and parameter
    groups.

    :param optimizer: Optimizer. If None, `LearningRateScheduler(optimizer)`
                      must be called before running `step()`.
    :param base_lr: Base learning rate.
    :param warmup: Number of initial updates during which the learning rate
                   linearly increases.
    """
    def __init__(self,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 base_lr: float = 1.0,
                 warmup: int = 0) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        check_condition(warmup >= 0, "warmup needs to be >= 0.")
        self.warmup = warmup
        self._t = 0
        self._last_lr = None  # type: Optional[List[float]]

    def __call__(self, optimizer: torch.optim.Optimizer) -> 'LearningRateScheduler':
        """
        DeepSpeed compatibility method: associate otherwise initialized learning
        rate scheduler with an optimizer.
        """
        assert self.optimizer is None, 'This learning rate scheduler is already associated with an optimizer.'
        self.optimizer = optimizer
        return self

    def __repr__(self) -> str:
        return self.__class__.__name__

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.__dict__.update(state_dict)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate for the current step for each param group.
        """
        raise NotImplementedError()

    def get_last_lr(self) -> List[float]:
        """
        Get the last computed learning rate for each param group.
        """
        assert self._last_lr is not None, '`get_last_lr()` cannot be called before `get_lr()`'
        return self._last_lr

    def step(self, t: Optional[int] = None):
        """
        Increment or specify the time step (update number) and recompute the
        learning rate for each param group by calling `get_lr()`.

        :param t: Manually specify the time step instead of automatically
                  incrementing the previous value.
        """
        assert self.optimizer is not None, 'This learning rate scheduler is not associated with an optimizer.'
        if t is None:
            t = self._t + 1
        self._t = t
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _warmup(self, t):
        """
        Returns linearly increasing fraction of base_lr.
        """
        if not self.warmup:
            return self.base_lr
        return self.base_lr * min(1.0, t / self.warmup)


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
    """

    def get_lr(self):
        # Warmup
        warm_lr = self._warmup(self._t)
        # Avoid square root of zero
        warmup_steps = max(1, self.warmup)
        # Warmup first N steps, then decay
        lr = warm_lr / sqrt(max(self._t, warmup_steps))
        return [lr for _ in self.optimizer.param_groups]


class LearningRateSchedulerLinearDecay(LearningRateScheduler):
    """
    Learning rate schedule: lr * (1 - t / total_steps)
    Step grows until it reaches decay_steps then remains constant.

    This is the schedule used by Devlin et al. in the BERT paper
    (https://arxiv.org/pdf/1810.04805.pdf).

    :param optimizer: Optimizer.
    :param base_lr: Base learning rate.
    :param total_steps: Number of total training updates.  The learning rate
                        linearly decays to zero over this period.
    :param warmup: Number of initial updates during which the learning rate
                   linearly increases.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, total_steps: int, warmup: int = 0) -> None:
        super().__init__(optimizer, base_lr, warmup)
        check_condition(total_steps >= 0, "total_steps need to be >= 0.")
        self.total_steps = total_steps

    def get_lr(self) -> List[float]:
        # Warmup
        warm_lr = self._warmup(self._t)
        # Linear decay
        bounded_t = min(max(self._t, 1), self.total_steps)
        lr = warm_lr * (1 - bounded_t / self.total_steps)
        return [lr for _ in self.optimizer.param_groups]


class LearningRateSchedulerPlateauReduce(AdaptiveLearningRateScheduler):
    """
    Lower the learning rate as soon as the validation score plateaus.

    :param optimizer: Optimizer.
    :param base_lr: Base learning rate.
    :param reduce_factor: Factor to reduce learning rate with.
    :param reduce_num_not_improved: Number of checkpoints with no improvement
                                    after which learning rate is reduced.
    :param warmup: Number of initial updates during which the learning rate
                   linearly increases.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 reduce_factor: float,
                 reduce_num_not_improved: int,
                 warmup: int = 0) -> None:
        super().__init__(optimizer, base_lr, warmup)
        self.lr = base_lr
        check_condition(0.0 < reduce_factor < 1, "reduce_factor should be between (0, 1).")
        self.reduce_factor = reduce_factor
        self.reduce_num_not_improved = reduce_num_not_improved
        self.num_not_improved = 0

        self.warmed_up = not self.warmup > 0

        logger.info("Will reduce the learning rate by a factor of %.2f whenever"
                    " the validation score doesn't improve %d times.",
                    reduce_factor, reduce_num_not_improved)

    def __repr__(self) -> str:
        return (
            "LearningRateSchedulerPlateauReduce(reduce_factor=%.2f, reduce_num_not_improved=%d, num_not_improved=%d,"
            " base_lr=%s, lr=%s, warmup=%d, warmed_up=%s)"
            %
            (self.reduce_factor, self.reduce_num_not_improved,
             self.num_not_improved, self.base_lr, self.lr, self.warmup, self.warmed_up)
        )

    def new_evaluation_result(self, has_improved: bool) -> bool:
        """
        Returns true if the parameters should be reset to the ones with the best validation score.

        :param has_improved: Whether the model improved on held-out validation data.
        :return: True if parameters should be reset to the ones with best validation score.
        """
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

    def get_lr(self) -> List[float]:
        lr = self._warmup(self._t) if self.warmup > 0 and self._t <= self.warmup else self.lr
        if self._t == self.warmup:
            self.warmed_up = True
        return [lr for _ in self.optimizer.param_groups]


def get_lr_scheduler(scheduler_type: str,
                     base_learning_rate: float,
                     learning_rate_reduce_factor: float,
                     learning_rate_reduce_num_not_improved: int,
                     learning_rate_warmup: int = 0,
                     max_updates: Optional[int] = None) -> Tuple[Optional[Type[LearningRateScheduler]], Dict[str, Any]]:
    """
    Get learning rate scheduler class and kwargs.

    :param scheduler_type: Scheduler type.
    :param base_lr: Base learning rate.
    :param learning_rate_reduce_factor: Factor to reduce learning rate with.
    :param learning_rate_reduce_num_not_improved: Number of checkpoints with no
           improvement after which learning rate is reduced.
    :param learning_rate_warmup: Number of initial updates during which the
                                 learning rate linearly increases.
    :param max_updates: Number of total training updates.

    :raises: ValueError if unknown scheduler_type

    :return: Tuple of LearningRateScheduler class and kwargs dictionary.
    """
    if scheduler_type is None or scheduler_type == C.LR_SCHEDULER_NONE:
        return None, {}
    if scheduler_type == C.LR_SCHEDULER_INV_SQRT_DECAY:
        return LearningRateSchedulerInvSqrtDecay, {'base_lr': base_learning_rate, 'warmup': learning_rate_warmup}
    if scheduler_type == C.LR_SCHEDULER_LINEAR_DECAY:
        check_condition(max_updates is not None,
                        "The total number of training updates (--max-updates) must be specified when using the linear "
                        "decay learning rate scheduler.")
        return LearningRateSchedulerLinearDecay, {'base_lr': base_learning_rate, 'total_steps': max_updates,
                                                  'warmup': learning_rate_warmup}
    if scheduler_type == C.LR_SCHEDULER_PLATEAU_REDUCE:
        check_condition(learning_rate_reduce_factor is not None,
                        "learning_rate_reduce_factor needed for %s scheduler" % C.LR_SCHEDULER_PLATEAU_REDUCE)
        check_condition(learning_rate_reduce_num_not_improved is not None,
                        "learning_rate_reduce_num_not_improved needed for %s scheduler" % C.LR_SCHEDULER_PLATEAU_REDUCE)
        if learning_rate_reduce_factor >= 1.0:
            logger.warning("Not using %s learning rate scheduling: learning_rate_reduce_factor == 1.0",
                           C.LR_SCHEDULER_PLATEAU_REDUCE)
            return None, {}
        return LearningRateSchedulerPlateauReduce, {'base_lr': base_learning_rate,
                                                    'reduce_factor': learning_rate_reduce_factor,
                                                    'reduce_num_not_improved': learning_rate_reduce_num_not_improved,
                                                    'warmup': learning_rate_warmup}
    raise ValueError("Unknown learning rate scheduler type %s." % scheduler_type)

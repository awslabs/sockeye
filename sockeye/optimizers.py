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

"""
Extra optimizers not included in MXNet.
"""

from abc import abstractmethod
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple

import math
import mxnet as mx

from . import config
from .lr_scheduler import LearningRateScheduler
from .utils import check_condition

BatchState = namedtuple("BatchState", ["metric_val"])
CheckpointState = namedtuple("CheckpointState", ["checkpoint", "metric_val"])


class OptimizerConfig(config.Config):

    def __init__(self,
                 name: str,
                 params: Dict[str, Any],
                 kvstore: str,
                 initializer: mx.initializer.Initializer,
                 gradient_clipping_type: str,
                 gradient_clipping_threshold: Optional[float]) -> None:
        super().__init__()
        self.name = name
        self.params = params
        self.kvstore = kvstore
        self.initializer = initializer
        self.gradient_clipping_type = gradient_clipping_type
        self.gradient_clipping_threshold = gradient_clipping_threshold

    @property
    def lr_scheduler(self) -> Optional[LearningRateScheduler]:
        return self.params.get("lr_scheduler", None)

    def set_lr_scheduler(self, lr_scheduler: Optional[LearningRateScheduler]):
        self.params["lr_scheduler"] = lr_scheduler


class SockeyeOptimizer(mx.optimizer.Optimizer):
    """
    Optimizer that has access to additional information from the last batch and the last checkpoint
    when updating weights.

    :param request_optimized_metric: Whether to request the optimized metric (e.g. perplexity) in
                                     place of optimizer loss (e.g. cross-entropy).
    """
    def __init__(self, request_optimized_metric: bool = False, **kwargs) -> None:
        self.request_optimized_metric = request_optimized_metric
        self.batch_state = None # type: Optional[BatchState]
        self.checkpoint_state = None # type: Optional[CheckpointState]
        super().__init__(**kwargs)

    def pre_update_batch(self, batch_state: BatchState):
        """
        Called automatically prior to `update()` for each batch.
        """
        self.batch_state = batch_state

    def pre_update_checkpoint(self, checkpoint_state: CheckpointState):
        """
        Called automatically at each checkpoint.
        """
        self.checkpoint_state = checkpoint_state

    @abstractmethod
    def update(self, index, weight, grad, state):
        """
        Called automatically as normal.
        """
        pass


class EveState:
    """
    Storage class for Eve optimizer state information.
    """
    def __init__(self, weight: mx.nd.NDArray) -> None:
        # Mean and variance for Adam
        self.mean = mx.nd.zeros_like(weight, ctx=weight.context)
        self.variance = mx.nd.zeros_like(weight, ctx=weight.context)
        # For Nadam warmup
        self.m_schedule = 1.
        # Values for computing Eve's d term (batch)
        self.batch_f_hat_prev = 0.
        self.batch_d_prev = 1.
        # Values for computing Eve's d term (checkpoint)
        self.checkpoint_prev = 0
        self.checkpoint_f_hat_prev = 0.
        self.checkpoint_d_prev = 1.


@mx.optimizer.Optimizer.register
class Eve(SockeyeOptimizer):
    """
    The Eve optimizer is an extended version of Adam that incorporates feedback from the objective
    function to further adapt the learning rate.
        * "Improving Stochastic Gradient Descent with Feedback"
          Jayanth Koushik; Hiroaki Hayashi (https://arxiv.org/abs/1611.01505)

    This version allows:
        * Using validation checkpoint loss in addition to training batch loss.
        * Using Adam or Nesterov Adam (Nadam) as the base algorithm

    Eve does not currently support rescaling gradients, clipping gradients, or weight decay.

    :param learning_rate: The initial learning rate.
    :param beta1: Exponential decay rate for the first moment estimates.
    :param beta2: Exponential decay rate for the second moment estimates.
    :param beta3_batch: Exponential decay rate for batch objective relative change.
    :param beta3_checkpoint: Exponential decay rate for checkpoint objective relative change.
    :param epsilon: Small value to avoid division by 0.
    :param k_lo: Lower threshold for relative change.
    :param k_hi: Upper threshold for relative change.
    :param use_batch_objective: Incorporate batch objective (can use both).
    :param use_checkpoint_objective: Incorporate checkpoint objective (can use both).
    :param use_nesterov_momentum: Use Nesterov-accelerated adaptive moment estimation (update rules
                                  used by "Nadam" optimizer).
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 beta3_batch: float = 0.999,
                 beta3_checkpoint: float = 0.,
                 epsilon: float = 1e-8,
                 k_lo: float = 0.1,
                 k_hi: float = 10,
                 schedule_decay: float = 0.004,
                 use_batch_objective: bool = True,
                 use_checkpoint_objective: bool = False,
                 use_nesterov_momentum: bool = False,
                 **kwargs) -> None:
        check_condition(any((use_batch_objective, use_checkpoint_objective)),
                        "Must use at least one of: batch objective, checkpoint objective")
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3_batch = beta3_batch
        self.beta3_checkpoint = beta3_checkpoint
        self.epsilon = epsilon
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.schedule_decay = schedule_decay
        self.use_batch_objective = use_batch_objective
        self.use_checkpoint_objective = use_checkpoint_objective
        self.use_nesterov_momentum = use_nesterov_momentum

    def create_state(self, index: int, weight: mx.nd.NDArray) -> EveState:
        return EveState(weight)

    def update(self, index: int, weight: mx.nd.NDArray, grad: mx.nd.NDArray, state: EveState):

        assert isinstance(weight, mx.nd.NDArray)
        assert isinstance(grad, mx.nd.NDArray)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        t = self._index_update_count[index]

        # Preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -1. * self.clip_gradient, self.clip_gradient)

        # First compute Eve's f_hat and d terms

        def compute_d(t: int, f: float, f_hat_prev: float, d_prev: float, beta: float) -> Tuple[float, float]:
            """Compute Eve's f_hat and d terms as described in paper"""
            if t > 1:
                # The original paper has a typo in the algorithm here.  The following lines are re-
                # written to reflect the actual logic presented in the authors' longer explanation.
                if f <= f_hat_prev:
                    delta_lo = 1. / (self.k_hi + 1.)
                    delta_hi = 1. / (self.k_lo + 1.)
                else:
                    delta_lo = self.k_lo + 1.
                    delta_hi = self.k_hi + 1.
                # ^ End modified section ^
                c = min(max(delta_lo, f / f_hat_prev), delta_hi)
                f_hat = c * f_hat_prev
                r = abs(f_hat - f_hat_prev) / min(f_hat, f_hat_prev)
                d = beta * d_prev + (1. - beta) * r
            else:
                f_hat = f
                d = 1.
            return f_hat, d

        batch_d, checkpoint_d = None, None

        # Computation occurs for each batch
        if self.use_batch_objective:
            batch_f_hat, batch_d = compute_d(t,
                                             self.batch_state.metric_val,
                                             state.batch_f_hat_prev,
                                             state.batch_d_prev,
                                             self.beta3_batch)
            state.batch_f_hat_prev = batch_f_hat
            state.batch_d_prev = batch_d

        # Computation occurs once per checkpoint using the checkpoint number as t.  Prior to the
        # first checkpoint, d = 1.
        if self.use_checkpoint_objective:
            # Only need to recompute if we've seen a new checkpoint since the previous batch update
            if (isinstance(self.checkpoint_state, CheckpointState) and
                self.checkpoint_state.checkpoint != state.checkpoint_prev):
                checkpoint = self.checkpoint_state.checkpoint
                checkpoint_f_hat, checkpoint_d = compute_d(checkpoint,
                                                           self.checkpoint_state.metric_val,
                                                           state.checkpoint_f_hat_prev,
                                                           state.checkpoint_d_prev,
                                                           self.beta3_checkpoint)
                state.checkpoint_prev = checkpoint
                state.checkpoint_f_hat_prev = checkpoint_f_hat
                state.checkpoint_d_prev = checkpoint_d
            else:
                checkpoint_d = state.checkpoint_d_prev

        # Batch and checkpoint contribute equally when both are used
        if self.use_batch_objective and self.use_checkpoint_objective:
            d = (batch_d + checkpoint_d) / 2.
        elif self.use_batch_objective:
            d = batch_d
        elif self.use_checkpoint_objective:
            d = checkpoint_d
        else:
            raise ValueError

        # Update mean and variance (Adam/Nadam)
        m_t, v_t = state.mean, state.variance

        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        v_t[:] = self.beta2 * v_t + (1. - self.beta2) * grad * grad

        # Finally apply either Adam or Nadam update
        if self.use_nesterov_momentum:
            # Nadam warming momentum schedule
            momentum_t = self.beta1 * (1. - 0.5 * 0.96**(t * self.schedule_decay))
            momentum_t_1 = self.beta1 * (1. - 0.5 * 0.96**((t + 1) * self.schedule_decay))
            state.m_schedule = state.m_schedule * momentum_t
            m_schedule_next = state.m_schedule * momentum_t_1
            # Nadam update terms
            grad_prime = grad / (1. - state.m_schedule)
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t_prime = v_t / (1. - self.beta2**t)
            m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime
            # Final weight update with extra d term
            weight[:] -= lr * m_t_bar / (d * mx.nd.sqrt(v_t_prime) + self.epsilon)
        else:
            # Adam warmup
            coef1 = 1. - self.beta1**t
            coef2 = 1. - self.beta2**t
            lr *= math.sqrt(coef2) / coef1
            # Final weight update with extra d term
            weight[:] = weight - lr * m_t / (d * mx.nd.sqrt(v_t) + self.epsilon)

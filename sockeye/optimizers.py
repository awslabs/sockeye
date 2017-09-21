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
import math
from typing import Optional, Tuple

from mxnet.ndarray import NDArray, sqrt, zeros_like
from mxnet.optimizer import Optimizer

from sockeye.utils import check_condition

BatchState = namedtuple("BatchState", ["metric_val"])
CheckpointState = namedtuple("CheckpointState", ["checkpoint", "metric_val"])


class SockeyeOptimizer(Optimizer):
    """
    Optimizer that has access to additional information from the last batch and the last chekpoint
    when updating weights.
    """
    def __init__(self, **kwargs) -> None:
        self.batch_state = Optional[BatchState]
        self.checkpoint_state = Optional[CheckpointState]
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
    def __init__(self, weight: NDArray) -> None:
        # Mean and variance for Adam
        self.mean = zeros_like(weight, ctx=weight.context)
        self.variance = zeros_like(weight, ctx=weight.context)
        # Values for computing Eve's d term (batch)
        self.batch_f_hat_prev = 0.
        self.batch_d_prev = 1.
        # Values for computing Eve's d term (checkpoint)
        self.checkpoint_prev = 0
        self.checkpoint_f_hat_prev = 0.
        self.checkpoint_d_prev = 1.


@Optimizer.register
class Eve(SockeyeOptimizer):
    """
    The Eve optimizer is an extended version of Adam that incorporates feedback from the objective
    function to further adapt the learning rate.
        * "Improving Stochastic Gradient Descent with Feedback"
          Jayanth Koushik; Hiroaki Hayashi (https://arxiv.org/abs/1611.01505)

    This version allows using validation checkpoint loss in addition to training batch loss.

    Eve does not currently support rescaling gradients, clipping gradients, or weight decay.

    :param learning_rate: The initial learning rate.
    :param beta1: Exponential decay rate for the first moment estimates.
    :param beta2: Exponential decay rate for the second moment estimates.
    :param beta3: Exponential decay rate for batch objective relative change.
    :param beta4: Exponential decay rate for checkpoint objective relative change.
    :param epsilon: Small value to avoid division by 0.
    :param k_lo: Lower threshold for relative change.
    :param k_hi: Upper threshold for relative change.
    :param use_batch_objective: Incorporate batch objective (can use both).
    :param use_checkpoint_objective: Incorporate checkpoint objective (can use both).
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 beta3: float = 0.999,
                 beta4: float = 0.,
                 epsilon: float = 1e-8,
                 k_lo: float = 0.1,
                 k_hi: float = 10,
                 use_batch_objective: bool = True,
                 use_checkpoint_objective: bool = False,
                 **kwargs) -> None:
        check_condition(any((use_batch_objective, use_checkpoint_objective)),
                        "Must use at least one of: batch objective, checkpoint objective")
        check_condition(kwargs.get("rescale_grad", 1.) == 1., "Eve optimizer does not support rescaling gradients.")
        check_condition(not kwargs.get("clip_gradient", False), "Eve optimizer does not support gradient clipping.")
        check_condition(kwargs.get("wd", 0.) == 0., "Eve optimizer does not support weight decay.")
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.epsilon = epsilon
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.use_batch_objective = use_batch_objective
        self.use_checkpoint_objective = use_checkpoint_objective

    def create_state(self, index: int, weight: NDArray) -> EveState:
        return EveState(weight)

    def update(self, index: int, weight: NDArray, grad: NDArray, state: EveState):

        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        lr = self._get_lr(index)
        self._update_count(index)
        t = self._index_update_count[index]

        # Standard Adam rules for updating mean and variance
        mean = state.mean
        var = state.variance

        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        var[:] = self.beta2 * var + (1. - self.beta2) * (grad**2)

        # Now compute Eve's f_hat and d terms

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
            return (f_hat, d)

        # Computation occurs for each batch
        if self.use_batch_objective:
            batch_f_hat, batch_d = compute_d(t,
                                             self.batch_state.metric_val,
                                             state.batch_f_hat_prev,
                                             state.batch_d_prev,
                                             self.beta3)
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
                                                           self.beta4)
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

        # Final weight update rule (Adam rule with extra d term)
        weight[:] = weight - lr * mean / (d * sqrt(var) + self.epsilon)

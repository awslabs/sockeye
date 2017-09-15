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

from mxnet.ndarray import NDArray, sqrt, zeros
from mxnet.optimizer import Optimizer

from sockeye.utils import check_condition

BatchState = namedtuple("BatchState", ["metric_val"])
CheckpointState = namedtuple("CheckpointState", ["checkpoint", "metric_val"])


class SockeyeOptimizer(Optimizer):
    """
    Optimizer that has access to additional information from the last batch and the last chekpoint
    when updating weights.
    """
    def __init__(self, **kwargs):
        self.batch_state = None
        self.checkpoint_state = None
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


# convenience wrapper for Optimizer.Register
register = Optimizer.register   # pylint: disable=invalid-name


@register
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
                 **kwargs):
        check_condition(any((use_batch_objective, use_checkpoint_objective)),
                        "Must use at least one of: batch objective, checkpoint objective")
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

    def create_state(self, index: int, weight: NDArray):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype),  # variance
                [0, 1],     # batch previous objective "hat", previous d
                [0, 0, 1])  # checkpoint number, previous objective "hat", previous d

    def update(self,
               index: int,
               weight: NDArray,
               grad: NDArray,
               state: object):

        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        lr = self._get_lr(index)

        mean, var, prev_batch, prev_checkpoint = state

        # Standard Adam rules for updating mean and variance
        self._update_count(index)
        t = self._index_update_count[index]

        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        var[:] = self.beta2 * var + (1. - self.beta2) * (grad**2)

        # Now compute Eve's d term
        d = 0.

        # Eve rules from paper: compute f_hat and d based on last training batch objective
        if self.use_batch_objective:
            f_hat_prev, d_prev = prev_batch
            f = self.batch_state.metric_val
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
                d_batch = self.beta3 * d_prev + (1. - self.beta3) * r
            else:
                f_hat = f
                d_batch = 1.
            prev_batch[:] = [f_hat, d_batch]
            d += d_batch

        # Extension: compute f_hat and d based on last validation checkpoint objective
        # Computation occurs once per checkpoint using the checkpoint number as t.  Prior to the
        # first checkpoint, d = 1.
        if self.use_checkpoint_objective:
            checkpoint, f_hat_prev, d_prev = prev_checkpoint
            # Only need to recompute if we've seen a new checkpoint since the previous batch update
            if self.checkpoint_state and self.checkpoint_state.checkpoint != checkpoint:
                checkpoint = self.checkpoint_state.checkpoint
                f = self.checkpoint_state.metric_val
                if checkpoint > 1:
                    if f <= f_hat_prev:
                        delta_lo = 1. / (self.k_hi + 1.)
                        delta_hi = 1. / (self.k_lo + 1.)
                    else:
                        delta_lo = self.k_lo + 1.
                        delta_hi = self.k_hi + 1.
                    c = min(max(delta_lo, f / f_hat_prev), delta_hi)
                    f_hat = c * f_hat_prev
                    r = abs(f_hat - f_hat_prev) / min(f_hat, f_hat_prev)
                    d_checkpoint = self.beta4 * d_prev + (1. - self.beta4) * r
                else:
                    f_hat = f
                    d_checkpoint = 1.
                prev_checkpoint[:] = [checkpoint, f_hat, d_checkpoint]
            else:
                d_checkpoint = d_prev
            d += d_checkpoint

        # Batch and checkpoint contribute equally when both are used
        if self.use_batch_objective and self.use_checkpoint_objective:
            d /= 2.

        # Final weight update rule (Adam rule with extra d term)
        weight[:] = weight - lr * mean / (d * sqrt(var) + self.epsilon)

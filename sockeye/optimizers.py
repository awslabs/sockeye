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
Extensions to MXNet optimizers
"""

from abc import abstractmethod
from collections import namedtuple
import math

from mxnet.ndarray import NDArray, sqrt, zeros
from mxnet.optimizer import Optimizer


CurrentTrainingState = namedtuple("CurrentTrainingState", ["metric_val"])


class SockeyeOptimizer(Optimizer):
    """
    Extended optimizer that is passed the current training state via the `pre_update()` method.  The
    `update()` method then has access to information such as the metric value for the current batch.
    """
    def __init__(self, **kwargs):
        self.current_training_state = None
        super().__init__(**kwargs)

    def pre_update(self, current_training_state: CurrentTrainingState):
        """
        Called automatically prior to `update()` for each batch.
        """
        self.current_training_state = current_training_state

    @abstractmethod
    def update(self, index, weight, grad, state):
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

    Eve currently does not support rescaling gradients, clipping gradients, or weight decay.

    :param learning_rate: The initial learning rate.
    :param beta1: Exponential decay rate for the first moment estimates.
    :param beta2: Exponential decay rate for the second moment estimates.
    :param beta3: Exponential decay rate for computing relative change.
    :param epsilon: Small value to avoid division by 0.
    :param k_lo: Lower threshold for relative change.
    :param k_hi: Upper threshold for relative change.
    :param maximize_metric: Whether the optimized metric is maximized.
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 beta3: float = 0.999,
                 epsilon: float = 1e-8,
                 k_lo: float = 0.1,
                 k_hi: float = 10,
                 maximize_metric: bool = False,
                 **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.epsilon = epsilon
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.maximize_metric = maximize_metric

    def create_state(self, index: int, weight: NDArray):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype),  # variance
                [0, 1])  # previous objective "hat", previous d

    def update(self,
               index: int,
               weight: NDArray,
               grad: NDArray,
               state: object):

        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        lr = self._get_lr(index)

        mean, var, prev = state

        # Standard Adam rules for updating mean and variance
        self._update_count(index)
        t = self._index_update_count[index]

        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        var[:] = self.beta2 * var + (1. - self.beta2) * (grad**2)

        # Rules for updating Eve's f_hat and d terms
        f_hat_prev, d_prev = prev
        f = self.current_training_state.metric_val
        if t > 1:
            if (self.maximize_metric and f < f_hat_prev) or (not self.maximize_metric and f > f_hat_prev):
                delta_lo = self.k_lo + 1.
                delta_hi = self.k_hi + 1.
            else:
                delta_lo = 1. / (self.k_hi + 1.)
                delta_hi = 1. / (self.k_lo + 1.)
            c = min(max(delta_lo, f / f_hat_prev), delta_hi)
            f_hat = c * f_hat_prev
            r = abs(f_hat - f_hat_prev) / min(f_hat, f_hat_prev)
            d = self.beta3 * d_prev + (1. - self.beta3) * r
        else:
            f_hat = f
            d = 1.
        prev[:] = [f_hat, d]

        # Final weight update rule (Adam rule with extra d term)
        weight[:] = weight - lr * mean / (d * sqrt(var) + self.epsilon)

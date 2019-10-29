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

import math
from typing import Any, Dict, Optional

import mxnet as mx

from . import config
from .lr_scheduler import LearningRateScheduler
from sockeye_contrib.optimizers import bert_adam, lamb


class OptimizerConfig(config.Config):

    def __init__(self,
                 name: str,
                 params: Dict[str, Any],
                 kvstore: str,
                 initializer: mx.initializer.Initializer,
                 gradient_clipping_type: str,
                 gradient_clipping_threshold: Optional[float],
                 update_interval: int = 1) -> None:
        super().__init__()
        self.name = name
        self.params = params
        self.kvstore = kvstore
        self.initializer = initializer
        self.gradient_clipping_type = gradient_clipping_type
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.update_interval = update_interval

    @property
    def lr_scheduler(self) -> Optional[LearningRateScheduler]:
        return self.params.get("lr_scheduler", None)

    def set_lr_scheduler(self, lr_scheduler: Optional[LearningRateScheduler]):
        self.params["lr_scheduler"] = lr_scheduler


class LAAdamState:
    '''
    State for Adam with lookahead.  In addition to Adam's mean and variance,
    this keeps a copy of the "slow" weights.
    '''
    def __init__(self, weight: mx.nd.NDArray, stype: str) -> None:
        self.mean = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        self.variance = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        self.slow_weight = weight.copy()


@mx.optimizer.register
class LAAdam(mx.optimizer.Optimizer):
    '''
    Version of the Adam optimizer with lookahead (Zhang et al. 2019,
    arxiv.org/abs/1907.08610).  Lookahead uses 2 sets of weights, a "fast" set
    that is updated at each step and a "slow" set that is periodically updated
    toward the fast weights.
    '''
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 alpha: float = 0.5,
                 k: int = 10,
                 lazy_update: bool = True,
                 **kwargs) -> None:
        super(LAAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k
        self.lazy_update = lazy_update

    def create_state(self, index: int, weight: mx.nd.NDArray):
        stype = weight.stype if self.lazy_update else 'default'
        return LAAdamState(weight, stype)

    def update(self, index: int, weight: mx.nd.NDArray, grad: mx.nd.NDArray, state: LAAdamState):
        # Every update applies the standard Adam update to the fast weights
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]
        coef1 = 1. - self.beta1 ** t
        coef2 = 1. - self.beta2 ** t
        lr *= math.sqrt(coef2) / coef1

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon, 'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        mx.nd.adam_update(weight, grad, state.mean, state.variance, out=weight, lazy_update=self.lazy_update, lr=lr,
                          wd=wd, **kwargs)

        # Every K updates:
        # 1. The fast weights are used to update the slow weights
        # 2. The slow weights are copied back over the fast weights
        # (fast mean and variance are not modified)
        if t % self.k == 0:
            state.slow_weight[:] += self.alpha * (weight - state.slow_weight)
            weight[:] = state.slow_weight

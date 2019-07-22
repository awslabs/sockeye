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

from typing import Any, Dict, Optional
import warnings

import mxnet as mx
from mxnet.ndarray.contrib import adamw_update, mp_adamw_update
import numpy as np

from . import config
from .lr_scheduler import LearningRateScheduler


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


@mx.optimizer.Optimizer.register
class BERTAdam(mx.optimizer.Optimizer):
    """The Adam optimizer with weight decay regularization for BERT.

    Updates are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * (m / (sqrt(v) + epsilon) + wd * w)

    Note that this is different from `mxnet.optimizer.Adam`, where L2 loss is added and
    accumulated in m and v. In BERTAdam, the weight decay term decoupled from gradient
    based update.

    This is also slightly different from the AdamW optimizer described in
    *Fixing Weight Decay Regularization in Adam*, where the schedule multiplier and
    learning rate is decoupled, and the bias-correction terms are removed.
    The BERTAdam optimizer uses the same learning rate to apply gradients
    w.r.t. the loss and weight decay.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`mxnet.optimizer.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 **kwargs):
        super(BERTAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state_multi_precision(self, index, weight):
        """multi-precision state creation function."""
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn('Accumulating with float16 in optimizer can lead to '
                          'poor accuracy or slow convergence. '
                          'Consider using multi_precision=True option of the '
                          'BERTAdam optimizer')
        return self.create_state(index, weight)

    def create_state(self, _, weight):
        """state creation function."""
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype), #mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)) #variance

    def update(self, index, weight, grad, state):
        """update function"""
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        """multi-precision update function"""
        use_multi_precision = self.multi_precision and weight.dtype == np.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

    def _update_impl(self, indices, weight, grad, state, multi_precision=False):
        """update function"""
        self._update_count(indices)
        lr = self._get_lr(indices)
        wd = self._get_wd(indices)

        # pylint: disable=access-member-before-definition
        if not isinstance(self.rescale_grad, mx.nd.NDArray):
            self.rescale_grad = mx.nd.full(shape=(1,), val=self.rescale_grad, ctx=weight.context)
        else:
            self.rescale_grad = self.rescale_grad.as_in_context(weight.context)

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if not multi_precision:
            mean, var = state
            adamw_update(weight, grad, mean, var, out=weight,
                         lr=1, wd=wd, eta=lr, **kwargs)
        else:
            mean, var = state[0]
            mp_adamw_update(weight, grad, mean, var, state[1], out=weight,
                            lr=1, wd=wd, eta=lr, **kwargs)

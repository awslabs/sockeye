# Copyright 2017--2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mxnet as mx
import torch as pt

from . import config
from . import constants as C
from .lr_scheduler import LearningRateScheduler
from .model_pt import PyTorchSockeyeModel

@dataclass
class OptimizerConfig(config.Config):
    name: str
    params: Dict[str, Any]
    kvstore: str
    initializer: mx.initializer.Initializer
    gradient_clipping_type: str
    gradient_clipping_threshold: Optional[float]
    update_interval: int = 1

    @property
    def lr_scheduler(self) -> Optional[LearningRateScheduler]:
        return self.params.get("lr_scheduler", None)

    def set_lr_scheduler(self, lr_scheduler: Optional[LearningRateScheduler]):
        self.params["lr_scheduler"] = lr_scheduler


@dataclass
class PyTorchOptimizerConfig(config.Config):
    # Optimizer
    name: str

    # Adam default values
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.

    # SGD default value
    momentum: float = 0.

    # Applied outside of optimizer
    gradient_clipping_type: str = C.GRADIENT_CLIPPING_TYPE_NONE
    gradient_clipping_threshold: Optional[float] = None
    update_interval: int = 1
    rescale_grad: float = 1.

    lr_scheduler: Optional[LearningRateScheduler] = None


def get_optimizer(model: PyTorchSockeyeModel, config: PyTorchOptimizerConfig) -> pt.optim.Optimizer:
    """
    Create an optimizer for a Sockeye model using the specified config settings.
    """
    if config.name == C.OPTIMIZER_ADAM:
        return pt.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                             weight_decay=config.weight_decay)
    elif config.name == C.OPTIMIZER_SGD:
        return pt.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                            weight_decay=config.weight_decay)
    raise ValueError(f'Unknown optimizer: {config.name}')

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
import logging
from typing import Any, Dict, Optional, Tuple

import torch

from . import config
from . import constants as C
from .lr_scheduler import LearningRateScheduler

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig(config.Config):
    name: str
    params: Dict[str, Any]
    kvstore: str
    initializer: 'mx.initializer.Initializer'  # type: ignore
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
    running_on_gpu: bool = False

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

    lr_scheduler: Optional[LearningRateScheduler] = None


def get_optimizer(model: torch.nn.Module, config: PyTorchOptimizerConfig) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Create an optimizer for a Sockeye model using the specified config settings.

    :param model: Sockeye model.
    :param config: Optimizer config.

    :return: Tuple of an Optimizer and the kwargs dict for calling that
             optimizer's `zero_grad()` method.
    """
    adam_impl = torch.optim.Adam
    sgd_impl = torch.optim.SGD
    # Built-in optimizers take the "set_to_none" argument. See:
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    zero_grad_kwargs = {'set_to_none': True}

    if config.running_on_gpu:
        try:
            from apex.optimizers import FusedAdam, FusedSGD
            adam_impl = FusedAdam
            sgd_impl = FusedSGD
            # Apex optimizers automatically set gradients to none instead of
            # zeroing and do not have a "set_to_none" argument. See:
            # https://nvidia.github.io/apex/optimizers.html
            zero_grad_kwargs = {}
            logging.info('Using NVIDIA Apex fused optimizers')
        except ImportError:
            logger.warning('Cannot import NVIDIA Apex optimizers (FusedAdam, FusedSGD). Consider installing Apex for '
                           'faster GPU training: https://github.com/NVIDIA/apex')

    if config.name == C.OPTIMIZER_ADAM:
        return adam_impl(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                         weight_decay=config.weight_decay), zero_grad_kwargs
    elif config.name == C.OPTIMIZER_SGD:
        return sgd_impl(model.parameters(), lr=config.lr, momentum=config.momentum,
                        weight_decay=config.weight_decay), zero_grad_kwargs
    raise ValueError(f'Unknown optimizer: {config.name}')

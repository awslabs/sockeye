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

from typing import Any, Dict, Optional

import mxnet as mx

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

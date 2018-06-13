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

from random import random

import mxnet.ndarray as nd
import pytest
from mxnet import optimizer as opt

import sockeye.constants as C
from sockeye.optimizers import BatchState, CheckpointState, SockeyeOptimizer


@pytest.mark.parametrize("optimizer, optimizer_params",
                         ((C.OPTIMIZER_ADAM, {}),
                          (C.OPTIMIZER_EVE, {}),
                          (C.OPTIMIZER_EVE, {"use_batch_objective": True, "use_checkpoint_objective": True}),
                          ))
def test_optimizer(optimizer, optimizer_params):
    # Weights
    index = 0
    weight = nd.zeros(shape=(8,))
    # Optimizer from registry
    optimizer = opt.create(optimizer, **optimizer_params)
    state = optimizer.create_state(index, weight)
    # Run a few updates
    for i in range(1, 13):
        grad = nd.random_normal(shape=(8,))
        if isinstance(optimizer, SockeyeOptimizer):
            batch_state = BatchState(metric_val=random())
            optimizer.pre_update_batch(batch_state)
        optimizer.update(index, weight, grad, state)
        # Checkpoint
        if i % 3 == 0:
            if isinstance(optimizer, SockeyeOptimizer):
                checkpoint_state = CheckpointState(checkpoint=(i % 3 + 1), metric_val=random())
                optimizer.pre_update_checkpoint(checkpoint_state)

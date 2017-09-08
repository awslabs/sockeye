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

import pytest
from random import random

import mxnet.ndarray as nd
from mxnet import optimizer as opt

import sockeye.constants as C
from sockeye.optimizers import CurrentTrainingState, SockeyeOptimizer

@pytest.mark.parametrize("optimizer", [C.OPTIMIZER_ADAM, C.OPTIMIZER_EVE])
def test_optimizer(optimizer):
    # Weights
    index = 0
    weight = nd.zeros(shape=(8,))
    # Optimizer from registry
    optimizer = opt.create(optimizer)
    state = optimizer.create_state(index, weight)
    # Run a few updates
    for _ in range(10):
        grad = nd.random_normal(shape=(8,))
        if isinstance(optimizer, SockeyeOptimizer):
            current_training_state = CurrentTrainingState(metric_val=random())
            optimizer.pre_update(current_training_state)
        optimizer.update(index, weight, grad, state)

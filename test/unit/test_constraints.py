# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
from unittest.mock import Mock

import mxnet as mx
import numpy as np
import pytest
from math import ceil

import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
from sockeye.utils import SockeyeError
from sockeye.lexical_constraints import kbest, get_bank_sizes


def mock_translator(num_source_factors: int):
    t_mock = Mock(sockeye.inference.Translator)
    t_mock.num_source_factors = num_source_factors
    return t_mock


"""
Test how the banks are allocated. Given a number of constraints (C), the beam size (k)
a number of candidates for each bank [0..C], return the allocation of the k spots of the
beam to the banks.
"""
@pytest.mark.parametrize("num_constraints, beam_size, counts, allocation",
                         [
                             # no constraints: allocate all items to bin 0
                             (0, 5, [5], [5]),
                             # 1 constraints, but no candidates, so 0 alloc
                             (1, 5, [5,0], [5,0]),
                             # 1 constraint, but 1 candidate, so 1 alloc
                             (1, 5, [5,1], [4,1]),
                             # 1 constraint, > k candidates for each (impossible, but ok), even alloc, extra goes to last
                             (1, 5, [10,10], [2,3]),
                             # 1 constraint, > k candidates for each (impossible, but ok), even alloc, extra goes to last
                             (1, 5, [1,10], [1,4]),
                             # 2 constraints, no candidates
                             (2, 5, [5,0,0], [5,0,0]),
                             # 2 constraints, some candidates
                             (2, 10, [5,0,7], [5,0,5]),
                             # 2 constraints, some candidates
                             (2, 10, [5,1,7], [5,1,4]),
                             # more constraints than beam spots: allocate to last two spots
                             (3, 2, [1,0,1,1], [0,0,1,1]),
                             # more constraints than beam spots: slots allocated empirically
                             (3, 2, [1,0,1,0], [1,0,1,0]),
                             # more constraints than beam spots: all slots to last bank
                             (3, 2, [4,2,2,3], [0,0,0,2]),
                         ])
def test_bank_allocation(num_constraints, beam_size, counts, allocation):
    assert get_bank_sizes(num_constraints, beam_size, -1, -1, counts) == allocation


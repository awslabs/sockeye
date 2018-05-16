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
from sockeye.lexical_constraints import topk, get_bank_sizes, ConstrainedHypothesis

EOS_ID = 3

def mock_translator(num_source_factors: int):
    t_mock = Mock(sockeye.inference.Translator)
    t_mock.num_source_factors = num_source_factors
    return t_mock


"""
Test how the banks are allocated. Given a number of constraints (C), the beam size (k)
a number of candidates for each bank [0..C], return the allocation of the k spots of the
beam to the banks.
"""
@pytest.mark.parametrize("num_constraints, beam_size, counts, expected_allocation",
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
def test_constraints_bank_allocation(num_constraints, beam_size, counts, expected_allocation):
    allocation = get_bank_sizes(num_constraints, beam_size, counts)
    assert sum(allocation) == beam_size
    assert allocation == expected_allocation

"""
Make sure the internal representation is correct.
For the internal representation, the list of phrasal constraints is concatenated, and then
a parallel array is used to mark which words are part of a phrasal constraint.
"""
@pytest.mark.parametrize("raw_constraints, internal_constraints, internal_is_sequence",
                         [
                             # No constraints
                             ([], [], []),
                             # One single-word constraint
                             ([[17]], [17], [0]),
                             # Multiple multiple-word constraints.
                             ([[11, 12], [13, 14]], [11, 12, 13, 14], [True, False, True, False]),
                             # Multiple constraints
                             ([[11, 12, 13], [14], [15]], [11, 12, 13, 14, 15], [True, True, False, False, False]),
                         ])
def test_constraints_setup(raw_constraints, internal_constraints, internal_is_sequence):
    hyp = ConstrainedHypothesis(raw_constraints, EOS_ID)
    # record all the words that have been met
    for phrase in raw_constraints:
        for word_id in phrase:
            hyp = hyp.advance(word_id)

    assert hyp.constraints == internal_constraints
    assert hyp.is_sequence == internal_is_sequence


"""
Ensures that advance() works correctly. advance()
"""
@pytest.mark.parametrize("raw_constraints, met, unmet",
                         [
                             # No constraints
                             ([], [], []),
                             # Single simple unmet constraint
                             ([[17]], [], [17]),
                             # Single simple met constraint
                             ([[17]], [17], []),
                             # Met first word of a phrasal constraint, return just next word of phrasal
                             ([[11, 12], [13, 14]], [11], [12]),
                             # Completed phrase, have only single-word ones
                             ([[11, 12, 13], [14], [15]], [11, 12, 13], [14, 15]),
                         ])
def test_constraints_advance(raw_constraints, met, unmet):
    hyp = ConstrainedHypothesis(raw_constraints, EOS_ID)
    # record all the words that have been met
    for word_id in met:
        hyp = hyp.advance(word_id)

    assert set(hyp.allowed()) == set(unmet)


"""
Returns the list of unmet constraints.
"""
@pytest.mark.parametrize("raw_constraints, met, unmet",
                         [
                             # No constraints
                             ([], [], []),
                             # Single simple unmet constraint
                             ([[17]], [], [17]),
                             # Single simple met constraint
                             ([[17]], [17], []),
                             # Met first word of a phrasal constraint, return just next word of phrasal
                             ([[11, 12], [13, 14]], [11], [12]),
                             # Completed phrase, have only single-word ones
                             ([[11, 12, 13], [14], [15]], [11, 12, 13], [14, 15]),
                             # Same word twice
                             ([[11], [11]], [], [11, 11]),
                         ])
def test_constraints_extend(raw_constraints, met, unmet):
    hyp = ConstrainedHypothesis(raw_constraints, EOS_ID)
    # record these ones as met
    for word_id in met:
        hyp = hyp.advance(word_id)

    assert set(hyp.allowed()) == set(unmet)

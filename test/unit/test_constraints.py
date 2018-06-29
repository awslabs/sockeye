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
from sockeye.lexical_constraints import init_batch, get_bank_sizes, topk, ConstrainedHypothesis

BOS_ID = 2
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
def test_constraints_repr(raw_constraints, internal_constraints, internal_is_sequence):
    hyp = ConstrainedHypothesis(raw_constraints, EOS_ID)
    assert hyp.constraints == internal_constraints
    assert hyp.is_sequence == internal_is_sequence


"""
Tests many of the ConstrainedHypothesis functions.
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
                             ([[11, 12], [13, 14]], [11], [12, 13, 14]),
                             # Completed phrase, have only single-word ones
                             ([[11, 12, 13], [14], [15]], [11, 12, 13], [14, 15]),
                             # Same word twice
                             ([[11], [11]], [], [11, 11]),
                             ])
def test_constraints_logic(raw_constraints, met, unmet):
    hyp = ConstrainedHypothesis(raw_constraints, EOS_ID)
    # record these ones as met
    for word_id in met:
        hyp = hyp.advance(word_id)

    assert hyp.num_needed() == len(unmet)
    assert hyp.finished() == (len(unmet) == 0)
    assert hyp.is_valid(EOS_ID) == (hyp.finished() or (len(unmet) == 1 and EOS_ID in unmet))


"""
Test the allowed() function, which returns the set of unmet constraints that can be generated.
When inside a phrase, this is only the next word of the phrase. Otherwise, it is all unmet constraints.
"""
@pytest.mark.parametrize("raw_constraints, met, allowed",
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
                             # Same word twice, nothing met, return
                             ([[11], [11]], [], [11]),
                             # Same word twice, met, still returns once
                             ([[11], [11]], [11], [11]),
                             # Same word twice, met twice
                             ([[11], [11]], [11, 11], []),
                             # EOS, allowed
                             ([[42, EOS_ID]], [42], [EOS_ID]),
                             # EOS, not allowed
                             ([[42, EOS_ID]], [], [42]),
                         ])
def test_constraints_allowed(raw_constraints, met, allowed):
    hyp = ConstrainedHypothesis(raw_constraints, EOS_ID)
    # record these ones as met
    for word_id in met:
        hyp = hyp.advance(word_id)

    assert hyp.allowed() == set(allowed)
    assert hyp.num_met() == len(met)
    assert hyp.num_needed() == hyp.size() - hyp.num_met()





"""
Ensures that batches are initialized correctly.
Each line here is a tuple containing a list (for each sentence in the batch) of RawConstraintLists,
which are lists of list of integer IDs representing the constraints for the sentence.
"""
@pytest.mark.parametrize("raw_constraint_lists",
                         [ ([None, None, None, None]),
                           ([[[17]], None]),
                           ([None, [[17]]]),
                           ([[[17], [11, 12]], [[17]], None]),
                           ([None, [[17], [11, 12]], [[17]], None]),
                         ])
def test_constraints_init_batch(raw_constraint_lists):
    beam_size = 4  # arbitrary

    constraints = init_batch(raw_constraint_lists, beam_size, BOS_ID, EOS_ID)
    assert len(raw_constraint_lists) * beam_size == len(constraints)

    # Iterate over sentences in the batch
    for raw_constraint_list, constraint in zip(raw_constraint_lists, constraints[::beam_size]):
        if raw_constraint_list is None:
            assert constraint is None
        else:
            # The number of constraints is the sum of the length of the lists in the raw constraint list
            assert constraint.size() == sum([len(phr) for phr in raw_constraint_list])

            # No constraints are met unless the start_id happened to be at the start of a constraint
            num_met = 1 if any([phr[0] == BOS_ID for phr in raw_constraint_list]) else 0
            assert constraint.num_met() == num_met

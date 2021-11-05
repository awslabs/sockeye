# Copyright 2018--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from unittest.mock import Mock

import numpy as np
import pytest

from sockeye.data_io_pt import get_tokens, strids2ids
from sockeye.inference_pt import Translator
from sockeye.lexical_constraints import init_batch, get_bank_sizes, ConstrainedHypothesis, AvoidBatch, AvoidState, \
    AvoidTrie

BOS_ID = 2
EOS_ID = 3


def mock_translator(num_source_factors: int):
    t_mock = Mock(Translator)
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


test_avoid_list_data = [ (["this", "that", "this bad phrase", "this bad phrase that is longer"]),
                         ([]),
                         (["a really bad phrase"]),
                         (["lightning crashes", "lightning bugs"]) ]

"""
Ensures that the avoid trie is built correctly.
"""
@pytest.mark.parametrize("raw_phrase_list", test_avoid_list_data)
def test_avoid_list_trie(raw_phrase_list):
    # Make sure the trie reports the right size
    raw_phrase_list = [list(get_tokens(phrase)) for phrase in raw_phrase_list]
    root_trie = AvoidTrie(raw_phrase_list)
    assert len(root_trie) == len(raw_phrase_list)

    # The last word of the phrase should be in the final() list after walking through the
    # first len(phrase) - 1 words
    for phrase in raw_phrase_list:
        trie = root_trie
        for word in phrase[:-1]:
            trie = trie.step(word)
        assert phrase[-1] in trie.final()

    oov_id = 8239
    assert root_trie.step(oov_id) is None

    root_trie.add_phrase([oov_id, 17])
    assert root_trie.step(oov_id) is not None
    assert 17 in root_trie.step(oov_id).final()

"""
Ensure that state management works correctly.
State management records where we are in the trie.
"""
@pytest.mark.parametrize("raw_phrase_list", test_avoid_list_data)
def test_avoid_list_state(raw_phrase_list):
    raw_phrase_list = [list(get_tokens(phrase)) for phrase in raw_phrase_list]
    root_trie = AvoidTrie(raw_phrase_list)

    root_state = AvoidState(root_trie)
    oov_id = 83284

    # Consuming an OOV ID from the root state should return the same state
    assert root_state == root_state.consume(oov_id)

    # The avoid lists should also be the same
    assert root_state.avoid() == root_state.consume(oov_id).avoid()

    root_ids_to_avoid = root_state.avoid()
    for phrase in raw_phrase_list:
        state = root_state
        for word in phrase[:-1]:
            state = state.consume(word)

        # The last word of the phrase should be in the avoid list
        assert phrase[-1] in state.avoid()

        # Trying to advance on an OOV from inside a multi-word constraint should return a new state
        if len(phrase) > 1:
            new_state = state.consume(oov_id)
            assert new_state != state

"""
Ensure that managing states for a whole batch works correctly.

Here we pass a list of global phrases to avoid, followed by a sentence-level list. Then, the batch, beam, and vocabulary
sizes. Finally, the prefix that the decoder is presumed to have seen, and the list of vocab-transformed IDs we should
expect (a function of the vocab size) that should be blocked.
"""
@pytest.mark.parametrize("global_raw_phrase_list, raw_phrase_list, batch_size, beam_size, prefix, expected_avoid", [
    (['5 6 7 8'], None, 1, 3, '17', []),
    (['5 6 7 12'], None, 1, 4, '5 6 7', [(0, 12), (1, 12), (2, 12), (3, 12)]),
    (['5 6 7 8', '9'], None, 1, 2, '5 6 7', [(0, 8), (0, 9), (1, 8), (1, 9)]),
    (['5 6 7 8', '13'], [[[10]]], 1, 2, '5 6 7', [(0, 8), (0, 10), (0, 13), (1, 8), (1, 10), (1, 13)]),
    # first two hypotheses blocked on 19 (= 19 and 119), next two on 20 (= 220 and 320)
    (None, [[[19]], [[20]]], 2, 2, '', [(0, 19), (1, 19), (2, 20), (3, 20)]),
    # same, but also add global block list to each row
    (['74'], [[[19]], [[20]]], 2, 2, '', [(0, 19), (0, 74), (1, 19), (1, 74), (2, 20), (2, 74), (3, 20), (3, 74)]),
])
def test_avoid_list_batch(global_raw_phrase_list, raw_phrase_list, batch_size, beam_size, prefix, expected_avoid):

    global_avoid_trie = None
    if global_raw_phrase_list:
        global_raw_phrase_list = [list(strids2ids(get_tokens(phrase))) for phrase in global_raw_phrase_list]
        global_avoid_trie = AvoidTrie(global_raw_phrase_list)

    avoid_batch = AvoidBatch(batch_size, beam_size, avoid_list=raw_phrase_list, global_avoid_trie=global_avoid_trie)

    for word_id in strids2ids(get_tokens(prefix)):
        avoid_batch.consume(np.array([word_id] * (batch_size * beam_size)))

    avoid = [(x, y) for x, y in zip(*avoid_batch.avoid())]
    assert set(avoid) == set(expected_avoid)

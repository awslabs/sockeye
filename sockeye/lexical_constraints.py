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

import copy
import logging
import re
import time

from typing import Dict, List, NamedTuple, Tuple
from operator import attrgetter

from . import constants as C
from . import utils

import mxnet as mx
import numpy as np

logger = logging.getLogger(__name__)

# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
RawConstraintList = List[List[int]]

class ConstrainedHypothesis:
    """
    Represents a set of words and phrases that must appear in the output.
    A constraint is of two types: sequence or non-sequence.
    A non-sequence constraint is a single word and can therefore be followed by anything, whereas a sequence constraint must be followed by a particular word (the next word in the sequence).
    This class also records which constraints have been met.

    :param constraint_list: A list of zero or raw constraints (each represented as a list of integers).
    :param eos_id: The end-of-sentence ID.
    """
    def __init__(self, constraint_list: RawConstraintList, eos_id: int) -> None:

        # `constraints` records the words of the constraints, as a list (duplicates allowed).
        # `is_sequence` is a parallel array that records, for each corresponding constraint,
        self.constraints = []  # type: List[int]
        self.is_sequence = []  # type: List[bool]
        for phrase in constraint_list:
            self.constraints += phrase
            self.is_sequence += [True] * len(phrase)
            self.is_sequence[-1] = False

        self.eos_id = eos_id

        # no constraints have been met
        self.met = [False for x in self.constraints]
        self.lastMet = -1

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.constraints)

    def __str__(self) -> str:
        s = []
        for i, word_id in enumerate(self.constraints):
            s.append(str(word_id) if self.met[i] is False else 'X')
            if self.is_sequence[i]:
                s.append('->')
        return ' '.join(s)

    def size(self) -> int:
        """
        :return: the number of constraints
        """
        return len(self.constraints)

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        return sum(self.met)

    def num_needed(self) -> int:
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def allowed(self) -> List[int]:
        """
        Returns the set of constrained words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is a list of unmet constraints.
        If all constraints are met, an empty set is returned.

        :return: The ID of the next required word, or -1 if any word can follow
        """
        items = []
        # Add extensions of a started-but-incomplete sequential constraint
        if self.lastMet != -1 and self.is_sequence[self.lastMet] == 1:
            word_id = self.constraints[self.lastMet + 1]
            if word_id != self.eos_id or self.num_needed() == 1:
                items.append(word_id)

        # Add all constraints that aren't non-initial sequences
        else:
            for i, word_id in enumerate(self.constraints):
                if self.met[i] is False and (i == 0 or not self.is_sequence[i - 1]):
                    if word_id != self.eos_id or self.num_needed() == 1:
                        items.append(word_id)

        return items

    def finished(self) -> bool:
        """
        Return true if all the constraints have been met.

        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def is_valid(self, wordid) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return self.finished() or wordid != self.eos_id or (self.num_needed() == 1 and self.allowed() == [self.eos_id])

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        """
        Updates the constraints object based on advancing on word_id.
        There is a complication, in that we may have started but not
        yet completed a multi-word constraint.  We need to constraints
        to be added as unconstrained words, so if the next word is a
        invalid, we need to back out of marking the current phrase as
        met constraints.

        :param word_id: The word ID to advance on.
        :return: A deep copy of the object, advanced on word_id.
        """

        obj = copy.deepcopy(self)

        # First, check if we're updating a sequential constraint.
        if obj.lastMet != -1 and obj.is_sequence[obj.lastMet] == 1:
            if word_id == obj.constraints[obj.lastMet + 1]:
                # Here, the word matches what we expect next in the constraint, so we update everything
                obj.met[obj.lastMet + 1] = True
                obj.lastMet += 1
            else:
                # Here, the word is not the expected next word of the constraint, so we back out of the constraint.
                index = obj.lastMet
                while obj.is_sequence[index]:
                    obj.met[index] = False
                    index -= 1
                obj.lastMet = -1

        # next, check if we can meet a single-word constraint
        else:
            try:
                # build a tuple of (constraint, whether it's non-initial sequential, whether it's met)
                l = list(zip(obj.constraints, [0] + obj.is_sequence[:-1], obj.met))
                q = (word_id, 0, 0)
                pos = l.index(q)
                obj.met[pos] = True
                obj.lastMet = pos

            except:
                pass

        return obj


def get_bank_sizes(num_constraints, beam_size, candidates) -> List[int]:
    """
    Evenly distributes the beam across the banks, where each bank is a portion of the beam devoted
    to hypotheses having met the same number of constraints, 0..num_constraints.
    After the assignment, banks with more slots than candidates are adjusted.

    :param num_constraints: The number of constraints.
    :param beam_size: The beam size.
    :param candidates: The empirical counts of number of candidates in each bank.
    :return: A distribution over banks.
    """

    num_banks = num_constraints + 1
    bank_size = beam_size // num_banks
    remainder = beam_size - bank_size * num_banks

    # Distribute any remainder to the end
    assigned = [bank_size for x in range(num_banks)]
    assigned[-1] += remainder

    # Now, moving right to left, push extra allocation to earlier buckets.
    # This encodes a bias for higher buckets, but if no candidates are found, space
    # will be made in lower buckets. This may not be the best strategy, but it is important
    # that you start pushing from the bucket that is assigned the remainder, for cases where
    # num_constraints >= beam_size.
    for i in reversed(range(num_banks)):
        overfill = assigned[i] - candidates[i]
        if overfill > 0:
            assigned[i] -= overfill
            assigned[(i - 1) % num_banks] += overfill

    return assigned


class ConstrainedCandidate(NamedTuple('Candidate', [
    ('row', int),
    ('col', int),
    ('score', float),
    ('hypothesis', ConstrainedHypothesis)
])):
    __slots__ = ()

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


def topk(beam_size: int,
         inactive: mx.ndarray,
         scores: mx.ndarray,
         hypotheses: List[ConstrainedHypothesis],
         best_ids: mx.ndarray,
         best_word_ids: mx.ndarray,
         sequence_scores: mx.ndarray,
         context: mx.context.Context) -> Tuple[np.array, np.array, np.array, List[ConstrainedHypothesis], mx.nd.array]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param beam_size: The length of the kbest list to produce.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects.
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :param context: The MXNet device context.
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    num_constraints = hypotheses[0].size()

    _start_time = time.time()

    candidates = set()
    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row = int(row.asscalar())
        col = int(col.asscalar())
        seq_score = float(seq_score.asscalar())
        if hypotheses[row].is_valid(col):
            new_item = hypotheses[row].advance(col)
            cand = ConstrainedCandidate(row, col, seq_score, new_item)
            candidates.add(cand)

    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = mx.ndarray.argmin(scores, axis=1)
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.allowed()

        # (3) add the single-best item after this (if it's valid)
        col = int(best_next[row].asscalar())
        if hyp.is_valid(col):
            nextones.append(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            new_item = hyp.advance(col)
            score = scores[row,col].asscalar()
            cand = ConstrainedCandidate(row, col, score, new_item)
            candidates.add(cand)

    # Sort the candidates. After allocating the beam across the banks, we will pick the top items
    # for each bank from this list
    sorted_candidates = sorted(candidates, key=attrgetter('score'))

    # The number of hypotheses in each bank
    counts = [0 for x in range(num_constraints + 1)]
    for cand in sorted_candidates:
        counts[cand.hypothesis.num_met()] += 1

    # Adjust allocated bank sizes if there are too few candidates in any of them
    bank_sizes = get_bank_sizes(num_constraints, beam_size, counts)

    # Sort the candidates into the allocated banks
    pruned_candidates = []
    for i, cand in enumerate(sorted_candidates):
        bank = cand.hypothesis.num_met()

        if bank_sizes[bank] > 0:
            pruned_candidates.append(cand)
            bank_sizes[bank] -= 1

    inactive[:len(pruned_candidates)] = 0

    # Pad the beam so array assignment still works
    if len(pruned_candidates) < beam_size:
        inactive[len(pruned_candidates):] = 1
        pruned_candidates += [pruned_candidates[len(pruned_candidates)-1]] * (beam_size - len(pruned_candidates))

    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([[x.score] for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            inactive)


def main():
    """
    Usage: python3 -m sockeye.lexical_constraints [--bpe BPE_MODEL]

    Reads sentences and constraints on STDIN (tab-delimited) and generates the JSON format that can be used when passing `--json-input`
    to sockeye.translate.

    e.g.,

        echo -e "Dies ist ein Test .\tThis is\ttest" | python3 -m sockeye.lexical_constraints

    will produce the following JSON object:

        { "text": "Dies ist ein Test .", "constraints": ["This is", "test"] }

    If you have a BPE model

        echo -e "Dies ist ein Test .\tThis is\ttest" | python3 -m sockeye.lexical_constraints --bpe /path/to/bpe.model

    You might get instead:

        { "text": "Dies ist ein Te@@ st .", "constraints": ["This is", "te@@ st"] }

    You can then translate this object by passing it to Sockeye on STDIN as follows:

        python3 -m sockeye.translate -m /path/to/model --json-input --beam-size 20 --beam-prune 20

    (Note the recommended Sockeye parameters).
    """

    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description='Generate lexical constraint JSON format for Sockeye')
    parser.add_argument('--constraints', '-c', nargs='*', type=str, default=[], help="List of target-side constraints")
    args = parser.parse_args()

    glossary = [C.BOS_SYMBOL, C.EOS_SYMBOL]

    for line in sys.stdin:
        line = line.rstrip()

        # Constraints are in fields 2+
        source, *constraints = line.split('\t')

        obj = { 'text': source }
        if len(constraints) > 0:
            obj['constraints'] = constraints

        print(json.dumps(obj, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()

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

import logging
import time
import copy
import re
from typing import Dict, List, NamedTuple, Tuple
from operator import attrgetter

from . import constants as C
from . import utils

import mxnet as mx
import numpy as np

logger = logging.getLogger(__name__)

ConstraintSet = NamedTuple('Constraint', [
    ('word_ids', List[int]),
    ('sequence_markers', List[int])
])

class ConstrainedHypothesis:
    """
    Represents a set of words and phrases that must appear in the output.
    A constraint is of two types: sequence or non-sequence.
    A non-sequence constraint is a single word and can therefore be followed by anything, whereas a sequence constraint must be followed by a particular word (the next word in the sequence).
    This class also records which constraints have been met.
    """
    def __init__(self, constraint_list: List[Tuple[int,int]], eos_id) -> None:
        # constraints records the words of the constraints, as a list
        # is_sequence records, for each constraint, whether it is the non-final word of a phrasal constraint.
        self.constraints, self.is_sequence = constraint_list
        self.eos_id = eos_id

        # no constraints have been met
        self.met = [0 for x in self.constraints]
        self.lastMet = -1

    def __len__(self):
        """
        :return: The number of constraints.
        """
        return len(self.constraints)

    def __str__(self):
        s = []
        for i, id in enumerate(self.constraints):
            s.append(str(id) if self.met[i] == 0 else 'X')
            if self.is_sequence[i]: s.append('->')
        return ' '.join(s)

    def size(self):
        """
        :return: the number of constraints
        """
        return len(self.constraints)

    def numMet(self):
        """
        :return: the number of constraints that have been met.
        """
        return sum(self.met)

    def numNeeded(self):
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.numMet()

    def allowed(self) -> List[int]:
        """
        Returns the set of *constrained* words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is a list of unmet constraints.
        If all constraints are met, an empty set is returned.

        :return: The ID of the next required word, or -1 if any word can follow
        """
        items = []
        # Add extensions of sequential constraints
        if self.lastMet != -1 and self.is_sequence[self.lastMet] == 1:
            items.append(self.constraints[self.lastMet + 1])

        # Add all constraints that aren't non-initial sequences
        else:
            for i, id in enumerate(self.constraints):
                if self.met[i] == 0 and (i == 0 or not self.is_sequence[i - 1]):
                    items.append(self.constraints[i])

        return items

    def finished(self) -> bool:
        return self.numNeeded() == 0

    def isValid(self, wordid) -> bool:
        """Ensure </s> is only generated when the hypothesis is completed."""
        return self.finished() or wordid != self.eos_id

    def copy(self):
        newobj = ConstrainedHypothesis((self.constraints, self.is_sequence), self.eos_id)
        newobj.lastMet = self.lastMet
        newobj.met = [x for x in self.met]
        return newobj

    def advance(self, word_id: int):
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

        obj = self.copy()

        # first, check if we're updating a sequential constraint
        if obj.lastMet != -1 and obj.is_sequence[obj.lastMet] == 1:
            if word_id == obj.constraints[obj.lastMet + 1]:
                # Here, the word matches what we expect next in the constraint, so we update everything
                obj.met[obj.lastMet + 1] = 1
                obj.lastMet += 1
            else:
                # Here, the word is not the expected next word of the constraint, so we back out of the constraint.
                index = obj.lastMet
                while obj.is_sequence[index]:
                    obj.met[index] = 0
                    index -= 1
                obj.lastMet = -1

        # next, check if we can meet a single-word constraint
        else:
            try:
                # build a tuple of (constraint, whether it's non-initial sequential, whether it's met)
                l = list(zip(obj.constraints, [0] + obj.is_sequence[:-1], obj.met))
                q = (word_id, 0, 0)
                pos = l.index(q)
                obj.met[pos] = 1
                obj.lastMet = pos

            except:
                pass

        return obj


class NullHypothesis(ConstrainedHypothesis):
    """
    Represents an invalid state in the beam.
    """

    def __init__(self):
        super().__init__(([], []), 0)

    def isValid(self, word_id):
        return False

    def finished(self):
        return True


def get_bank_sizes(num_constraints, beam_size, time_step, max_length, candidates):
    """
    :param C: The number of constraints.
    :param k: The beam size.
    :param t: The timestep.
    :param N: The maximum sentence length.
    :param candidates: The empirical counts of number of candidates in each bank.
    :return: A distribution over banks.
    """

    num_banks = num_constraints + 1
    bank_size = beam_size // num_banks
    remainder = beam_size - bank_size * num_banks

    if BANK_ADJUSTMENT == 'even':
        assigned = [bank_size for x in range(num_banks)]
        assigned[-1] += remainder

        for i in range(len(assigned)):
            deficit = candidates[i] - assigned[i]
            while deficit < 0:
                # sort whole list by distance from i
                for j in sorted(list(range(i - 1, -1, -1)) + list(range(i + 1, len(assigned))),
                                key=lambda x: abs(x - i)):
                    capacity = candidates[j] - assigned[j]
                    if capacity > 0:
                        transfer = min(abs(deficit), capacity)
                        deficit += transfer
                        assigned[i] -= transfer
                        assigned[j] += transfer
    elif BANK_ADJUSTMENT == 'empirical':
        total = sum(candidates)
        assigned = [max(1 if x > 0.0 else 0, int(x / total * beam_size)) for x in candidates]
        # small adjustment if needed
        if sum(assigned) < beam_size:
            assigned[0] += (beam_size - sum(assigned))
        elif sum(assigned) > beam_size:
            i = sorted(enumerate(assigned), key=lambda x: x[1], reverse=True)[0][0]
            assigned[i] -= sum(assigned) - beam_size

    elif BANK_ADJUSTMENT == 'none':
        assigned = [min(bank_size, candidates[i]) for i in range(len(candidates))]
        shortage = beam_size - sum(assigned)
        assigned[0] += shortage

    return assigned


class Candidate(NamedTuple('Candidate', [
    ('row', int),
    ('col', int),
    ('score', float),
    ('constraints', ConstrainedHypothesis)
])):
    __slots__ = ()

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.constraints.numMet())


def kbest(time_step: int,
          active_beam_size: int,
          beam_size: int,
          max_output_length: int,
          scores: mx.ndarray,
          hypotheses: ConstrainedHypothesis,
          best_ids,
          best_word_ids,
          best_scores,
          context):
    """
    Builds a list of candidates. These candidates are pulled from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param time_step: The timestep.
    :param active_beam_size: the active beam size
    :param beam_size: The length of the kbest list to produce.
    :param max_output_length: The sentence maximum length.
    :param hypotheses: the list of hypothesis objects
    :param scores: the numpy array containing scores
    :param best_ids:
    :param best_word_ids:
    :param best_scores
    """

    def get_scalar(obj, dtype):
        if context == mx.cpu():
            return dtype(obj)
        else:
            return dtype(obj.asscalar())

    argmin = np.argmin if context == mx.cpu() else mx.ndarray.argmin

    num_constraints = hypotheses[0].size()

    _start_time = time.time()

    candidates = set()
    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, score in zip(best_ids, best_word_ids, best_scores):
        row = int(row.asscalar())
        col = int(col.asscalar())
        score = float(score.asscalar())
        if hypotheses[row].isValid(col):
            new_constraint = hypotheses[row].advance(col)
            cand = Candidate(row, col, score, new_constraint)
            candidates.add(cand)

    # (2,3) For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = argmin(scores, axis=1)
    for row in range(active_beam_size):
        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.allowed()

        # (3) add the single-best item after this (if it's valid)
        col = get_scalar(best_next[row], dtype=int)
        if hypotheses[row].isValid(col):
            nextones.append(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            new_constraint = hypotheses[row].advance(col)
            score = get_scalar(scores[row,col], dtype=float)
            cand = Candidate(row, col, score, new_constraint)
            candidates.add(cand)

    new_kbest = sorted(candidates, key=attrgetter('score'))

    # pad the beam if not full
    while len(new_kbest) < beam_size:
        new_kbest.append(Candidate(0, 0, np.inf, NullHypothesis()))

    # The number of hypotheses in each bank
    counts = [0 for x in range(num_constraints + 1)]
    for cand in new_kbest:
        counts[cand.constraints.numMet()] += 1

    pruned_candidates = []

    # Adjust allocated bank sizes if there are too few candidates in any of them
    bank_sizes = get_bank_sizes(num_constraints, beam_size, time_step, max_output_length, counts)

    # logger.info("Time: bank_sizes: %f", time.time() - _start_time)
    # _start_time = time.time()

    counts = [x for x in bank_sizes]
    # sort candidates into the allocated banks
    for i, cand in enumerate(new_kbest):
        bank = cand.constraints.numMet()
        counts[bank] -= 1

        if counts[bank] >= 0:
            pruned_candidates.append(cand)

    new_active_beam_size = len(pruned_candidates)

    while len(pruned_candidates) < beam_size:
        pruned_candidates.append(pruned_candidates[new_active_beam_size-1])

    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([[x.score] for x in pruned_candidates]),
            [x.constraints for x in pruned_candidates],
            new_active_beam_size)


def main():
    """
    Usage: python3 -m sockeye.lexical_constraints [--bpe BPE_MODEL] [-c "constraint1" ["constraint2"]]

    Reads sentences on STDIN and generates the JSON format that can be used when passing `--json-input`
    to sockeye.translate.

    Constraints can be provided as a list of quoted arguments to -c. For a stream of sentence / constraint
    pairs, you can specify the constraints following the sentence on STDIN, tab-delimited.

    e.g.,

        echo "This is a test" | python3 -m sockeye.lexical_constraints -c "test" "longer test"

    is equivalent to

        echo -e "This is a test\ttest\tlonger test" | python3 -m sockeye.lexical_constraints
    """


    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description='Generate lexical constraint JSON format for Sockeye')
    parser.add_argument('--bpe', default=None, help='Location of BPE model to apply')
    parser.add_argument('--add-sos', action='store_true', help='add <s> token')
    parser.add_argument('--add-eos', action='store_true', help='add </s> token')
    parser.add_argument('--constraints', '-c', nargs='*', type=str, default=[], help="List of target-side constraints")
    args = parser.parse_args()

    bpe = None
    if args.bpe is not None:
        try:
            import apply_bpe
        except:
            print("Couldn't load BPE module. Is it in your PYTHONPATH?", file=sys.stderr)

        bpe = apply_bpe.BPE(open(args.bpe))

    def maybe_segment(phr: str) -> str:
        """
        Applies BPE if enabled.
        """
        if bpe is not None:
            return bpe.segment(phr)
        else:
            return phr

    for line in sys.stdin:
        line = line.rstrip()

        if '\t' in line:
            # Constraints are in fields 2+
            source, *constraints = line.split('\t')
        else:
            source = line
            constraints = args.constraints

        for i, constraint in enumerate(constraints):
            phrase = ''
            if args.add_sos:
                phrase += C.BOS_SYMBOL + ' '
            phrase += maybe_segment(constraint)
            if args.add_eos:
                phrase += ' ' + C.EOS_SYMBOL

            constraints[i] = phrase

        obj = { 'text': maybe_segment(source) }
        if len(constraints) > 0:
            obj['constraints'] = constraints

        print(json.dumps(obj, ensure_ascii=False))


if __name__ == '__main__':
    main()

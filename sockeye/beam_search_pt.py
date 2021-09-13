# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Optional, Tuple, List
import functools
import sockeye.constants as C
import operator

import torch as pt

logger = logging.getLogger(__name__)


class UpdateScores(pt.nn.Module):
    """
    A Module that updates the scores from the decoder step with accumulated scores.
    Inactive hypotheses receive score inf. Finished hypotheses receive their accumulated score for C.PAD_ID.
    Hypotheses at maximum length are forced to produce C.EOS_ID.
    All other options are set to infinity.
    """

    def __init__(self):
        super().__init__()
        assert C.PAD_ID == 0, "This block only works with PAD_ID == 0"

    def forward(self, target_dists, finished, inactive,
                      scores_accumulated, lengths, max_lengths,
                      unk_dist, pad_dist, eos_dist):
        # make sure to avoid generating <unk> if unk_dist is specified
        if unk_dist is not None:
            target_dists = target_dists + unk_dist
        # broadcast hypothesis score to each prediction.
        # scores_accumulated. Shape: (batch*beam, 1)
        # target_dists. Shape: (batch*beam, vocab_size)
        scores = target_dists + scores_accumulated

        # Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
        # finished rows are inf everywhere except column zero (pad_id), which holds the accumulated model score.
        # Items that are finished (but not inactive) get their previous accumulated score for the <pad> symbol,
        # infinity otherwise.
        # pad_dist. Shape: (batch*beam, vocab_size)
        pad_dist = pt.cat((scores_accumulated, pad_dist), dim=1)
        scores = pt.where(pt.logical_or(finished, inactive), pad_dist, scores)

        # Update lengths of all items, except those that were already finished. This updates
        # the lengths for inactive items, too, but that doesn't matter since they are ignored anyway.
        lengths = lengths + (1 - finished)

        # Items that are at their maximum length and not finished now are forced to produce the <eos> symbol.
        # That is, we keep scores for hypotheses below max length or finished, and 'force-eos' the rest.
        below_max_length = lengths < max_lengths
        scores = pt.where(pt.logical_or(below_max_length, finished), scores, eos_dist + scores)

        return scores, lengths


class LengthPenalty(pt.nn.Module):
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def forward(self, lengths):
        if self.alpha == 0.0:
            if isinstance(lengths, (int, float)):
                return 1.0
            else:
                return pt.ones_like(lengths)
        else:
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator


class BrevityPenalty(pt.nn.Module):
    """
    Calculates the logarithmic brevity penalty as:
      weight * log min(1, exp(1 - ref_len / hyp_len)) = weight * min(0, 1 - ref_len / hyp_len).

    :param weight: Linear weight.
    """

    def __init__(self, weight: float = 0.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, hyp_lengths, reference_lengths):
        if self.weight == 0.0:
            if isinstance(hyp_lengths, (int, float)):
                return 0.0
            else:
                # subtract to avoid MxNet's warning of not using both arguments
                # this branch should not and is not used during inference
                return pt.zeros_like(hyp_lengths - reference_lengths)
        else:
            # log_bp is always <= 0.0
            if isinstance(hyp_lengths, (int, float)):
                log_bp = min(0.0, 1.0 - reference_lengths / hyp_lengths)
            else:
                log_bp = pt.minimum(pt.zeros_like(hyp_lengths, dtype=pt.float), 1.0 - reference_lengths / hyp_lengths)
            return self.weight * log_bp


class CandidateScorer(pt.nn.Module):

    def __init__(self,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 brevity_penalty_weight: float = 0.0) -> None:
        super().__init__()
        self._lp = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)
        self._bp = None  # type: Optional[BrevityPenalty]
        if brevity_penalty_weight > 0.0:
            self._bp = BrevityPenalty(weight=brevity_penalty_weight)

    def forward(self, scores, lengths, reference_lengths):
        lp = self._lp(lengths)
        if self._bp is not None:
            bp = self._bp(lengths, reference_lengths)
        else:
            if isinstance(scores, (int, float)):
                bp = 0.0
            else:
                # avoid warning for unused input
                bp = pt.zeros_like(reference_lengths) if reference_lengths is not None else 0.0
        return scores / lp - bp

    def unnormalize(self, scores, lengths, reference_lengths):
        bp = 0.0 if self._bp is None else self._bp(lengths, reference_lengths)
        return (scores + bp) * self._lp(lengths)


class SortNormalizeAndUpdateFinished(pt.nn.Module):
    """
    A Module for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self,
                 dtype: str,
                 pad_id: int,
                 eos_id: int,
                 scorer: CandidateScorer) -> None:
        super().__init__()
        self.dtype = dtype
        self.pad_id = pad_id
        self.eos_id = eos_id
        self._scorer = scorer

    def forward(self, best_hyp_indices, best_word_indices,
                finished, scores_accumulated, lengths, reference_lengths,
                factors=None):

        # Reorder fixed-size beam data according to best_hyp_indices (ascending)
        finished = finished.gather(0, best_hyp_indices)
        lengths = lengths.take(0, best_hyp_indices)
        reference_lengths = reference_lengths.gather(0, best_hyp_indices)

        # Normalize hypotheses that JUST finished
        all_finished = pt.logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id).unsqueeze(1)
        newly_finished = pt.logical_xor(all_finished, finished)

        scores_accumulated = pt.where(newly_finished,
                                      self._scorer(scores_accumulated,
                                                   lengths,  # TODO cast int lengths to dtype required?
                                                   reference_lengths),
                                      scores_accumulated)

        # Recompute finished. Hypotheses are finished if they are extended with <pad> or <eos>
        finished = pt.logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        finished = finished.unsqueeze(1)  # TODO cast to int32 required?

        # Concatenate sorted secondary target factors to best_word_indices. Shape: (batch*beam, num_factors)
        best_word_indices = best_word_indices.unsqueeze(1)

        if factors is not None:
            secondary_factors = factors.gather(0, best_hyp_indices)
            best_word_indices = pt.cat((best_word_indices, secondary_factors), dim=1)

        return best_word_indices, finished, scores_accumulated, lengths, reference_lengths


# TODO: make this a proper (jitted) op
def unravel_index(indices: pt.LongTensor, shape: Tuple[int, ...]) -> pt.LongTensor:
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    coord = pt.stack(coord[::-1], dim=0)
    return coord


class TopK(pt.nn.Module):
    """
    Batch-wise topk operation.
    Forward method uses imperative shape inference, since both batch_size and vocab_size are dynamic
    during translation (due to variable batch size and potential vocabulary selection).
    """

    def __init__(self, k: int) -> None:
        """
        :param k: The number of smallest scores to return.
        """
        super().__init__()
        self.k = k

    def forward(self, scores, offset):
        """
        Get the lowest k elements per sentence from a `scores` matrix.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """
        batch_times_beam, vocab_size = scores.size()
        batch_size = batch_times_beam // self.k
        # Shape: (batch size, beam_size * vocab_size)
        scores = scores.view(batch_size, self.k * vocab_size)

        values, indices = pt.topk(scores, k=self.k, dim=1, largest=False, sorted=True)

        # Project indices back into original shape (which is different for t==1 and t>1)
        values, indices = values.view(-1, 1), indices.view(-1)

        best_hyp_indices, best_word_indices = unravel_index(indices, (batch_size * self.k, vocab_size))
        if batch_size > 1:
            # Offsetting the indices to match the shape of the scores matrix
            best_hyp_indices = best_hyp_indices + offset
        return best_hyp_indices, best_word_indices, values


# class SampleK(pt.nn.Module):
#     """
#     A Module for selecting a random word from each hypothesis according to its distribution.
#     """
#     def __init__(self, n) -> None:
#         super().__init__()
#         self.n = n
#
#     def forward(self, scores, target_dists, finished, best_hyp_indices):
#         """
#         Choose an extension of each hypothesis from its softmax distribution.
#
#         :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
#         :param target_dists: The non-cumulative target distributions (ignored).
#         :param finished: The list of finished hypotheses.
#         :param best_hyp_indices: Best hypothesis indices constant.
#         :return: The row indices, column indices, and values of the sampled words.
#         """
#         # Map the negative logprobs to probabilities so as to have a distribution
#         target_dists = pt.exp(-target_dists)
#
#         # n == 0 means sample from the full vocabulary. Otherwise, we sample from the top n.
#         if self.n != 0:
#             # select the top n in each row, via a mask
#             masked_items = pt.topk(target_dists, k=self.n, ret_typ='mask', dim=1, largest=False, sorted=True)
#             # set unmasked items to 0
#             masked_items = np.where(masked_items, target_dists, masked_items)
#             # renormalize
#             target_dists = masked_items / np.sum(masked_items, axis=1, keepdims=True)
#
#         # Sample from the target distributions over words, then get the corresponding values from the cumulative scores
#         best_word_indices = npx.random.categorical(target_dists, get_prob=False)
#         # Zeroes for finished hypotheses.
#         best_word_indices = np.where(finished, np.zeros_like(best_word_indices), best_word_indices)
#         values = npx.pick(scores, best_word_indices, axis=1, keepdims=True)
#
#         best_hyp_indices = npx.slice_like(best_hyp_indices, best_word_indices, axes=(0,))
#
#         return best_hyp_indices, best_word_indices, values


def _repeat_states(states: List, beam_size: int, state_structure: List) -> List:
    repeated_states = []
    flat_structure = functools.reduce(operator.add, state_structure)
    assert len(states) == len(flat_structure), "Number of states do not match the defined state structure"
    num_repeats = beam_size
    for state, state_format in zip(states, flat_structure):
        if state_format == C.STEP_STATE or state_format == C.BIAS_STATE:
            # Steps and source_bias have batch dimension on axis 0
            repeats = [num_repeats] + [0] * (state.dim() - 1)
        elif state_format == C.DECODER_STATE or state_format == C.ENCODER_STATE:
            # Decoder and encoder layer states have batch dimension on axis 1
            repeats = [0, num_repeats] + [0] * (state.dim() - 2)
        else:
            raise ValueError("Provided state format %s not recognized." % state_format)
        repeated_states.append(state.repeat(*repeats))
    return repeated_states


class SortStates(pt.nn.Module):

    def __init__(self, state_structure):
        super().__init__()
        self.flat_structure = functools.reduce(operator.add, state_structure)

    def forward(self, best_hyp_indices, *states):
        sorted_states = []
        assert len(states) == len(self.flat_structure), "Number of states do not match the defined state structure"
        for state, state_format in zip(states, self.flat_structure):
            if state_format == C.STEP_STATE or state_format == C.BIAS_STATE:
                # Steps and source_bias have batch dimension on axis 0
                sorted_state = state.gather(0, best_hyp_indices)
            elif state_format == C.DECODER_STATE:
                # Decoder and encoder layer states have batch dimension on axis 1
                sorted_state = state.gather(1, best_hyp_indices)
            elif state_format == C.ENCODER_STATE:
                # No need for takes on encoder layer states
                sorted_state = state
            else:
                raise ValueError("Provided state format %s not recognized." % state_format)
            sorted_states.append(sorted_state)
        return sorted_states
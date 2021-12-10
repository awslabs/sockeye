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

from typing import List, Optional
from typing import Tuple

import numpy as onp
import pytest
import torch as pt

import sockeye.beam_search_pt
import sockeye.constants as C
import sockeye.lexicon
import sockeye.utils


def test_length_penalty_default():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    lengths = np.array([[1], [2], [3]])
    length_penalty = sockeye.beam_search.LengthPenalty(1.0, 0.0)
    expected_lp = np.array([[1.0], [2.], [3.]])

    assert np.allclose(length_penalty(lengths), expected_lp)
    length_penalty.hybridize()
    assert np.allclose(length_penalty(lengths), expected_lp)


def test_pytorch_length_penalty_default():
    lengths = pt.tensor([[1], [2], [3]])
    length_penalty = sockeye.beam_search_pt.LengthPenalty(1.0, 0.0)
    expected_lp = pt.tensor([[1.0], [2.], [3.]])
    pt.testing.assert_allclose(length_penalty(lengths), expected_lp)


def test_length_penalty():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    lengths = np.array([[1], [2], [3]])
    length_penalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp = np.array([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])

    assert np.allclose(length_penalty(lengths), expected_lp)
    length_penalty.hybridize()
    assert np.allclose(length_penalty(lengths), expected_lp)


def test_pytorch_length_penalty():
    lengths = pt.tensor([[1], [2], [3]])
    length_penalty = sockeye.beam_search_pt.LengthPenalty(.2, 5.0)
    expected_lp = pt.tensor([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])
    pt.testing.assert_allclose(length_penalty(lengths), expected_lp)


def test_length_penalty_int_input():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    length = 1
    length_penalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp = [6 ** 0.2 / 6 ** 0.2]

    assert np.isclose(length_penalty(length), expected_lp)


def test_pytorch_length_penalty_int_input():
    length = 1
    length_penalty = sockeye.beam_search_pt.LengthPenalty(.2, 5.0)
    expected_lp = [6 ** 0.2 / 6 ** 0.2]
    assert onp.isclose(length_penalty(length), expected_lp)


def test_brevity_penalty_default():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    hyp_lengths = np.array([[1], [2], [3]])
    ref_lengths = np.array([[2], [3], [2]])
    brevity_penalty = sockeye.beam_search.BrevityPenalty(0.0)
    expected_bp = np.array([[0.0], [0.0], [0.0]])
    expected_bp_np = np.array([0.0, 0.0, 0.0])

    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)
    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp_np)
    brevity_penalty.hybridize()
    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)


def test_pytorch_brevity_penalty_default():
    hyp_lengths = pt.tensor([[1], [2], [3]])
    ref_lengths = pt.tensor([[2], [3], [2]])
    brevity_penalty = sockeye.beam_search_pt.BrevityPenalty(0.0)
    expected_bp = pt.tensor([[0.0], [0.0], [0.0]], dtype=pt.long)

    pt.testing.assert_allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)


def test_brevity_penalty():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    hyp_lengths = np.array([[1], [2], [3]])
    ref_lengths = np.array([[7], [2], [91]])
    brevity_penalty = sockeye.beam_search.BrevityPenalty(3.5)
    expected_bp = np.array([[3.5 * (1 - 7 / 1)], [0.0], [3.5 * (1 - 91 / 3)]])

    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)
    brevity_penalty.hybridize()
    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)


def test_pytorch_brevity_penalty():
    hyp_lengths = pt.tensor([[1], [2], [3]])
    ref_lengths = pt.tensor([[7], [2], [91]])
    brevity_penalty = sockeye.beam_search_pt.BrevityPenalty(3.5)
    expected_bp = pt.tensor([[3.5 * (1 - 7 / 1)], [0.0], [3.5 * (1 - 91 / 3)]])

    pt.testing.assert_allclose(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)


def test_brevity_penalty_int_input():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    hyp_length = 3
    ref_length = 5
    brevity_penalty = sockeye.beam_search.BrevityPenalty(2.0)
    expected_bp = [2.0 * (1 - 5 / 3)]

    assert np.isclose(brevity_penalty(hyp_length, ref_length), expected_bp)


def test_pytorch_brevity_penalty_int_input():
    hyp_length = 3
    ref_length = 5
    brevity_penalty = sockeye.beam_search_pt.BrevityPenalty(2.0)
    expected_bp = [2.0 * (1 - 5 / 3)]

    assert onp.isclose(brevity_penalty(hyp_length, ref_length), expected_bp)


def test_candidate_scorer():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    scorer = sockeye.beam_search.CandidateScorer(length_penalty_alpha=1.0,
                                                 length_penalty_beta=0.0,
                                                 brevity_penalty_weight=0.1)
    scorer.initialize()
    scorer.hybridize(static_alloc=True)

    # np.array input
    raw_scores = np.random.uniform(0, 1, (5,))
    lengths = np.array([1, 2, 3, 4, 5])
    reference_lengths = np.array([2, 3, 4, 5, 6])

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    assert np.allclose(unnormalized_scores, raw_scores)

    # int/float input
    raw_scores = 5.6
    lengths = 3
    reference_lengths = 4

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    assert np.allclose(unnormalized_scores, raw_scores)


def test_pytorch_candidate_scorer():
    scorer = sockeye.beam_search_pt.CandidateScorer(length_penalty_alpha=1.0,
                                                    length_penalty_beta=0.0,
                                                    brevity_penalty_weight=0.1)

    raw_scores = pt.rand(5).unsqueeze(1)
    lengths = pt.tensor([1, 2, 3, 4, 5])
    reference_lengths = pt.tensor([2, 3, 4, 5, 6])

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    pt.testing.assert_allclose(unnormalized_scores, raw_scores)

    # int/float input
    raw_scores = 5.6
    lengths = 3
    reference_lengths = 4

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    assert onp.allclose(unnormalized_scores, raw_scores)


def numpy_topk(scores: onp.ndarray,
               k: int,
               offset: onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Get the lowest k elements per sentence from a `scores` matrix using an intermediary Numpy conversion.
    This should be equivalent to sockeye.utils.topk() and is used as a comparative implementation in testing.

    :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
    :param k: The number of smallest scores to return.
    :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
    :return: The row indices, column indices and values of the k smallest items in matrix.
    """
    # (batch_size, beam_size * target_vocab_size)
    folded_scores = scores.reshape((-1, k * scores.shape[-1]))
    batch_size = folded_scores.shape[0]

    # Get the scores
    # Indexes into folded_scores: (batch_size, beam_size)
    flat_idxs = onp.argpartition(folded_scores, range(k))[:, :k]
    # Score values: (batch_size, beam_size)
    values = onp.array(folded_scores[onp.arange(folded_scores.shape[0])[:, None], flat_idxs])
    best_hyp_indices, best_word_indices = onp.array(onp.unravel_index(onp.ravel(flat_idxs), scores.shape),
                                                      dtype='int32')

    if batch_size > 1:
        # Offsetting the indices to match the shape of the scores matrix
        best_hyp_indices += offset

    values = values.reshape((-1, 1))
    return best_hyp_indices, best_word_indices, values


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size",
                        [(1, 5, 200),
                         (5, 5, 200),
                         (1, 1, 200),
                         (5, 1, 200),
                         (10, 10, 100)])
def test_topk_func(batch_size, beam_size, target_vocab_size):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores = np.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    # offset for batch sizes > 1
    offset = np.repeat(np.arange(0, batch_size * beam_size, beam_size, dtype='int32'), beam_size)

    np_hyp, np_word, np_values = numpy_topk(scores.asnumpy(), k=beam_size, offset=offset)

    topk = sockeye.beam_search.TopK(k=beam_size)
    topk.initialize()

    mx_hyp, mx_word, mx_values = topk(scores, offset)
    assert np.allclose(mx_hyp, np_hyp)
    assert np.allclose(mx_word, np_word)
    assert np.allclose(mx_values, np_values)

    topk.hybridize()
    mx_hyp, mx_word, mx_values = topk(scores, offset)
    assert np.allclose(mx_hyp, np_hyp)
    assert np.allclose(mx_word, np_word)
    assert np.allclose(mx_values, np_values)


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size",
                        [(1, 5, 200),
                         (5, 5, 200),
                         (1, 1, 200),
                         (5, 1, 200),
                         (10, 10, 100)])
def test_pytorch_topk_func(batch_size, beam_size, target_vocab_size):
    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores = onp.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    # offset for batch sizes > 1
    offset = onp.repeat(onp.arange(0, batch_size * beam_size, beam_size, dtype='int32'), beam_size)

    np_hyp, np_word, np_values = numpy_topk(scores, k=beam_size, offset=offset)

    topk = sockeye.beam_search_pt.TopK(k=beam_size)
    pt_hyp, pt_word, pt_values = topk(pt.tensor(scores), pt.tensor(offset))
    assert onp.allclose(pt_hyp.detach().numpy(), np_hyp)
    assert onp.allclose(pt_word.detach().numpy(), np_word)
    assert onp.allclose(pt_values.detach().numpy(), np_values)


@pytest.mark.parametrize("target_vocab_size", [2, 10, 500, 1024])
def test_greedytop1(target_vocab_size):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    batch_size = 1
    beam_size = 1
    target_vocab_size = 50
    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores = np.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    expected_hyp_index, expected_word_index, expected_value = numpy_topk(scores, k=beam_size, offset=None)
    assert expected_hyp_index[0] == 0
    assert expected_value.shape == (1, 1)

    greedy_top1 = sockeye.beam_search.GreedyTop1()
    greedy_top1.initialize()

    best_word_index = greedy_top1(scores, None, None)
    assert best_word_index.shape == (1, 1)
    assert best_word_index[0, 0] == expected_word_index[0]

    target_factors = np.ones((1, 1), dtype='int32')
    best_word_index_with_factors = greedy_top1(scores, None, target_factors)
    assert best_word_index_with_factors.shape == (1, 2)
    assert best_word_index_with_factors[0, 0] == expected_word_index[0]
    assert best_word_index_with_factors[0, 1] == target_factors.item()


@pytest.mark.parametrize("target_vocab_size", [2, 10, 500, 1024])
def test_pytorch_greedytop1(target_vocab_size):
    batch_size = 1
    beam_size = 1
    target_vocab_size = 50
    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores = onp.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    expected_hyp_index, expected_word_index, expected_value = numpy_topk(scores, k=beam_size, offset=None)
    assert expected_hyp_index[0] == 0
    assert expected_value.shape == (1, 1)

    greedy_top1 = sockeye.beam_search_pt.GreedyTop1()

    best_word_index = greedy_top1(pt.tensor(scores), None, None).detach().numpy()
    assert best_word_index.shape == (1, 1)
    assert best_word_index[0, 0] == expected_word_index[0]

    target_factors = pt.ones(1, 1, 2, dtype=pt.int32)
    best_word_index_with_factors = greedy_top1(pt.tensor(scores), None, target_factors).detach().numpy()
    assert best_word_index_with_factors.shape == (1, 2)
    assert best_word_index_with_factors[0, 0] == expected_word_index[0]
    assert best_word_index_with_factors[0, 1] == target_factors[:, :, 1].item()


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size, top_n",
                        [(1, 5, 200, 0),
                         (5, 5, 200, 0),
                         (1, 100, 200, 5),
                         (5, 100, 200, 5)])
def test_samplek_func(batch_size, beam_size, target_vocab_size, top_n):
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    # arrange scores increasing values from left to right, so the best item is always index 0, next-best 1, and so on
    scores = np.array([list(range(1, target_vocab_size + 1)) for _ in range(batch_size * beam_size)])
    # normalize
    target_dists = scores / scores.sum(axis=1, keepdims=True)

    samplek = sockeye.beam_search.SampleK(n=top_n)
    samplek.initialize()

    sample_best_hyp_indices = np.arange(0, batch_size * beam_size, dtype='int32')

    # 0..(batch_size * beam_size)-1
    expected_hyps = np.array(range(batch_size * beam_size), dtype='int32')
    finished = (np.random.uniform(0, 1, (batch_size * beam_size)) > 0.5).astype('int32')

    for i in [1, 2]:
        if i == 2:
            samplek.hybridize()

        hyps, words, values = samplek(scores, scores, finished, sample_best_hyp_indices)
        assert hyps.shape[0] == batch_size * beam_size

        # The indices should always be the integers from 0 to batch*beam-1
        assert sum(hyps == expected_hyps).item() == (batch_size * beam_size)
        if top_n != 0:
            # Scores are increasing left-to-right, so best items are all the lowest word IDs.
            # No word id greater than the cap (top_n) should be selected
            assert np.sum(words >= top_n).item() == 0

        # word index should be zero for all finished hypotheses
        assert np.sum(np.where(finished, words, finished)).item() == 0


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size, top_n",
                        [(1, 5, 200, 0),
                         (5, 5, 200, 0),
                         (1, 100, 200, 5),
                         (5, 100, 200, 5)])
def test_pytorch_samplek_func(batch_size, beam_size, target_vocab_size, top_n):
    # arrange scores increasing values from left to right, so the best item is always index 0, next-best 1, and so on
    scores = pt.tensor([list(range(1, target_vocab_size + 1)) for _ in range(batch_size * beam_size)])

    samplek = sockeye.beam_search_pt.SampleK(n=top_n)

    # 0..(batch_size * beam_size)-1
    expected_hyps = pt.tensor(range(batch_size * beam_size), dtype=pt.int32)
    finished = pt.rand(batch_size * beam_size) > 0.5

    for i in [1, 2]:
        hyps, words, values = samplek(scores, scores, finished)
        assert hyps.shape[0] == batch_size * beam_size

        # The indices should always be the integers from 0 to batch*beam-1
        assert (hyps == expected_hyps).sum().item() == (batch_size * beam_size)
        if top_n != 0:
            # Scores are increasing left-to-right, so best items are all the lowest word IDs.
            # No word id greater than the cap (top_n) should be selected
            assert (words >= top_n).sum().item() == 0

        # word index should be zero for all finished hypotheses
        assert pt.where(finished, words, finished.long()).sum().item() == 0


def test_update_scores():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    vocab_size = 10
    batch_beam_size = 3
    us = sockeye.beam_search.UpdateScores()
    pad_dist = np.full((batch_beam_size, vocab_size - 1), fill_value=np.inf, dtype='float32')
    eos_dist = np.full((batch_beam_size, vocab_size), fill_value=np.inf, dtype='float32')
    eos_dist[:, C.EOS_ID] = 0
    unk_dist = None

    lengths = np.array([[0], [1], [0]], dtype='int32')
    max_lengths = np.array([[1], [2], [3]], dtype='int32')  # first on reaches max length
    scores_accumulated = np.ones((3, 1), dtype='float32')
    finished = np.array([[0],  # not finished
                         [1],  # finished
                         [0]],  # not finished
                        dtype='int32')
    inactive = np.zeros_like(finished)
    target_dists = np.random.uniform(0, 1, (3, vocab_size))

    scores, lengths = us(target_dists, finished, inactive, scores_accumulated, lengths, max_lengths,
                         unk_dist, pad_dist, eos_dist)
    scores = scores
    lengths = lengths.reshape((-1,))

    assert (lengths == np.array([[1], [1], [1]])).all()  # all lengths but finished updated + 1
    assert (scores[0] == (1. + target_dists[0] + eos_dist)).all()  # 1 reached max length, force eos
    assert (scores[1] == np.array([1.] + pad_dist[1].tolist())).all()  # 2 finished, force pad, keep score
    assert (scores[2] == (1. + target_dists[2])).all()  # 3 scores + previous scores


@pytest.mark.parametrize("use_unk_dist", [False, True])
def test_pytorch_update_scores(use_unk_dist):
    vocab_size = 10
    batch_beam_size = 3
    us = sockeye.beam_search_pt.UpdateScores(prevent_unk=use_unk_dist)
    pad_dist = onp.full((1, vocab_size), fill_value=onp.inf, dtype='float32')
    pad_dist[0, 0] = 0
    eos_dist = onp.full((batch_beam_size, vocab_size), fill_value=onp.inf, dtype='float32')
    eos_dist[:, C.EOS_ID] = 0

    lengths = onp.array([0, 1, 0], dtype='int32')
    max_lengths = onp.array([1, 2, 3], dtype='int32')  # first on reaches max length
    scores_accumulated = onp.ones((3, 1), dtype='float32')
    finished = pt.tensor([False,  # not finished
                          True,  # finished
                          False],  # not finished
                         dtype=pt.bool)
    target_dists = onp.random.uniform(0, 1, (3, vocab_size)).astype('float32')

    scores, lengths = us(pt.tensor(target_dists), finished,
                         pt.tensor(scores_accumulated), pt.tensor(lengths), pt.tensor(max_lengths),
                         pt.tensor(pad_dist), pt.tensor(eos_dist))
    scores = scores.detach().numpy()
    lengths = lengths
    pt.testing.assert_allclose(lengths, pt.tensor([1, 1, 1]))  # all lengths but finished updated + 1
    assert (scores[0] == (1. + target_dists[0] + eos_dist)).all()  # 1 reached max length, force eos
    assert (scores[1] == (1. + pad_dist[0]).tolist()).all()  # 2 finished, force pad, keep score
    if use_unk_dist:
        assert scores[2, C.UNK_ID] == onp.inf  # 3 scores of <unk> should be np.inf
        target_dists[2, C.UNK_ID] = onp.inf
        assert (scores[2] == (1. + target_dists[2])).all()  # 3 scores + previous scores
    else:
        assert (scores[2] == (1. + target_dists[2])).all()  # 3 scores + previous scores


def test_prevent_unk_update_scores():
    pytest.importorskip("mxnet")
    from mxnet import np
    import sockeye.beam_search

    vocab_size = 10
    batch_beam_size = 3
    us = sockeye.beam_search.UpdateScores()
    pad_dist = np.full((batch_beam_size, vocab_size - 1), fill_value=np.inf, dtype='float32')
    eos_dist = np.full((batch_beam_size, vocab_size), fill_value=np.inf, dtype='float32')
    eos_dist[:, C.EOS_ID] = 0
    unk_dist = np.zeros_like(eos_dist)
    unk_dist[:, C.UNK_ID] = np.inf  # pylint: disable=E1137

    lengths = np.array([[0], [1], [0]], dtype='int32')
    max_lengths = np.array([[1], [2], [3]], dtype='int32')  # first on reaches max length
    scores_accumulated = np.ones((3, 1), dtype='float32')
    finished = np.array([[0],  # not finished
                         [1],  # finished
                         [0]],  # not finished
                        dtype='int32')
    inactive = np.zeros_like(finished)
    target_dists = np.random.uniform(0, 1, (3, vocab_size))

    scores, lengths = us(target_dists, finished, inactive, scores_accumulated, lengths, max_lengths,
                         unk_dist, pad_dist, eos_dist)
    scores = scores
    lengths = lengths.reshape((-1,))

    assert (lengths == np.array([[1], [1], [1]])).all()  # all lengths but finished updated + 1
    assert (scores[0] == (1. + target_dists[0] + eos_dist)).all()  # 1 reached max length, force eos
    assert (scores[1] == np.array([1.] + pad_dist[1].tolist())).all()  # 2 finished, force pad, keep score
    assert scores[2, C.UNK_ID] == np.inf    # 3 scores of <unk> should be np.inf
    assert (scores[2] == (1. + target_dists[2] + unk_dist[2])).all()  # 3 scores + previous scores


class _TestInference(sockeye.beam_search_pt._Inference):

    def __init__(self, output_vocab_size: int):
        self.output_vocab_size = output_vocab_size
        self.states = []

    def state_structure(self):
        return C.STEP_STATE + C.STEP_STATE  # is this the correct structure to use for self.states?

    def encode_and_initialize(self,
                              inputs: pt.Tensor,
                              valid_length: Optional[pt.Tensor] = None):
        batch_size = inputs.shape[0]
        # 'lengths'
        internal_lengths = pt.zeros(batch_size, 1, dtype=pt.int)
        num_decode_step_calls = pt.zeros(1, dtype=pt.int)
        self.states = [internal_lengths, num_decode_step_calls]  # TODO add nested states
        predicted_output_length = pt.ones(batch_size, 1)  # does that work?
        return self.states, predicted_output_length

    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List,
                    vocab_slice_ids: Optional[pt.Tensor] = None):
        batch_beam_size, num_target_factors = step_input.size()
        print('step_input', step_input)

        internal_lengths, num_decode_step_calls = states
        num_decode_step_calls = num_decode_step_calls.item()
        if num_decode_step_calls == 0:  # first call to decode_step, we expect step input to be all <bos>
            assert (step_input == C.BOS_ID).all()

        if step_input[:, 0].item() == C.BOS_ID:
            # predict word id 4 given <bos>
            scores = pt.tensor([0, 0, 0, 0, 1])
        elif step_input[:, 0].item() == C.EOS_ID:
            # predict pad given <eos>
            scores = pt.tensor([1, 0, 0, 0, 0])
        else:
            # otherwise always predict pad
            scores = pt.tensor([0, 0, 0, 0, 1])

        # topk is minimizing
        scores *= -1

        internal_lengths += 1
        num_decode_step_calls += 1

        self.states = states = [internal_lengths, pt.tensor([num_decode_step_calls], dtype=pt.int)]
        return scores, states, None


# TODO make this a useful test
# TODO: add vocabulary selection test
def test_beam_search():
    device = pt.device('cpu')
    dtype = pt.float32
    num_source_factors = 1
    num_target_factors = 1
    vocab_size = len(C.VOCAB_SYMBOLS) + 1  # 1 actual word: word id 4
    beam_size = 1
    bos_id = 2
    eos_id = 3

    inference = _TestInference(output_vocab_size=vocab_size)
    bs = sockeye.beam_search_pt.BeamSearch(
        beam_size=beam_size,
        dtype=dtype,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        output_vocab_size=vocab_size,
        scorer=sockeye.beam_search_pt.CandidateScorer(),
        num_source_factors=num_source_factors,
        num_target_factors=num_target_factors,
        inference=inference,
        beam_search_stop=C.BEAM_SEARCH_STOP_ALL,
        sample=None)

    # inputs
    batch_size = 1
    max_length = 3
    source = pt.tensor([[C.BOS_ID, 4, C.EOS_ID, C.PAD_ID, C.PAD_ID]], dtype=dtype).reshape(1, -1, 1)
    source_length = (source != C.PAD_ID).sum(1).reshape(-1)  # (batch_size,)

    restrict_lexicon = None
    max_output_lengths = pt.tensor([max_length], dtype=pt.int)

    bs_out = bs(source, source_length, restrict_lexicon, max_output_lengths)
    r = bs_out

    print('beam search lengths', r.lengths)
    print('internal lengths', inference.states[0])
    pt.testing.assert_allclose(r.lengths, inference.states[0].squeeze(1))
    assert inference.states[1] == max_length

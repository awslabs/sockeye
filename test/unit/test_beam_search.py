# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import Tuple

import mxnet as mx
import numpy as np
import pytest
from unittest.mock import patch, Mock
from typing import List, Optional

import sockeye.constants as C
import sockeye.beam_search
import sockeye.data_io
import sockeye.inference
import sockeye.lexical_constraints
import sockeye.lexicon
import sockeye.model
import sockeye.utils


def test_length_penalty_default():
    lengths = mx.nd.array([[1], [2], [3]])
    length_penalty = sockeye.beam_search.LengthPenalty(1.0, 0.0)
    expected_lp = np.array([[1.0], [2.], [3.]])

    assert np.allclose(length_penalty(lengths).asnumpy(), expected_lp)
    assert np.allclose(length_penalty(lengths).asnumpy(), expected_lp)
    length_penalty.hybridize()
    assert np.allclose(length_penalty(lengths).asnumpy(), expected_lp)


def test_length_penalty():
    lengths = mx.nd.array([[1], [2], [3]])
    length_penalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp = np.array([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])

    assert np.allclose(length_penalty(lengths).asnumpy(), expected_lp)
    assert np.allclose(length_penalty(lengths).asnumpy(), expected_lp)
    length_penalty.hybridize()
    assert np.allclose(length_penalty(lengths).asnumpy(), expected_lp)


def test_length_penalty_int_input():
    length = 1
    length_penalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp = [6 ** 0.2 / 6 ** 0.2]

    assert np.isclose(length_penalty(length), expected_lp)


def test_brevity_penalty_default():
    hyp_lengths = mx.nd.array([[1], [2], [3]])
    ref_lengths = mx.nd.array([[2], [3], [2]])
    brevity_penalty = sockeye.beam_search.BrevityPenalty(0.0)
    expected_bp = mx.nd.array([[0.0], [0.0], [0.0]])
    expected_bp_np = np.array([0.0, 0.0, 0.0])

    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths).asnumpy(), expected_bp.asnumpy())
    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths).asnumpy(), expected_bp_np)
    brevity_penalty.hybridize()
    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths).asnumpy(), expected_bp.asnumpy())


def test_brevity_penalty():
    hyp_lengths = mx.nd.array([[1], [2], [3]])
    ref_lengths = mx.nd.array([[7], [2], [91]])
    brevity_penalty = sockeye.beam_search.BrevityPenalty(3.5)
    expected_bp = np.array([[3.5 * (1 - 7 / 1)], [0.0], [3.5 * (1 - 91 / 3)]])

    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths).asnumpy(), expected_bp)
    brevity_penalty.hybridize()
    assert np.allclose(brevity_penalty(hyp_lengths, ref_lengths).asnumpy(), expected_bp)


def test_brevity_penalty_int_input():
    hyp_length = 3
    ref_length = 5
    brevity_penalty = sockeye.beam_search.BrevityPenalty(2.0)
    expected_bp = [2.0 * (1 - 5 / 3)]

    assert np.isclose(brevity_penalty(hyp_length, ref_length), expected_bp)


def test_candidate_scorer():
    scorer = sockeye.beam_search.CandidateScorer(length_penalty_alpha=1.0,
                                                 length_penalty_beta=0.0,
                                                 brevity_penalty_weight=0.1)
    scorer.initialize()
    scorer.hybridize(static_alloc=True)

    # NDArray input
    raw_scores = mx.nd.random.uniform(0, 1, (5,))
    lengths = mx.nd.array([1, 2, 3, 4, 5])
    reference_lengths = mx.nd.array([2, 3, 4, 5, 6])

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    assert np.allclose(unnormalized_scores.asnumpy(), raw_scores.asnumpy())

    # int/float input
    raw_scores = 5.6
    lengths = 3
    reference_lengths = 4

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    assert np.allclose(unnormalized_scores, raw_scores)


def test_sort_by_index():
    data = [mx.nd.random.uniform(0, 1, (3, i)) for i in range(1, 5)]
    indices = mx.nd.array([2, 0, 1], dtype='int32')
    expected = [d.asnumpy()[indices.asnumpy()] for d in data]

    sort_by_index = sockeye.beam_search.SortByIndex()
    sort_by_index.initialize()

    out = sort_by_index(indices, *data)
    assert len(out) == len(data) == len(expected)
    for o, e in zip(out, expected):
        assert np.allclose(o.asnumpy(), e)

    sort_by_index.hybridize()
    out = sort_by_index(indices, *data)
    assert len(out) == len(data) == len(expected)
    for o, e in zip(out, expected):
        assert np.allclose(o.asnumpy(), e)


def numpy_topk(scores: mx.nd.NDArray,
               k: int,
               offset: mx.nd.NDArray) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray]:
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

    folded_scores = folded_scores.asnumpy()
    # Get the scores
    # Indexes into folded_scores: (batch_size, beam_size)
    flat_idxs = np.argpartition(folded_scores, range(k))[:, :k]
    # Score values: (batch_size, beam_size)
    values = mx.nd.array(folded_scores[np.arange(folded_scores.shape[0])[:, None], flat_idxs], ctx=scores.context)
    best_hyp_indices, best_word_indices = mx.nd.array(np.unravel_index(flat_idxs.ravel(), scores.shape),
                                                      dtype='int32', ctx=scores.context)

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
    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores = mx.nd.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    # offset for batch sizes > 1
    offset = mx.nd.repeat(mx.nd.arange(0, batch_size * beam_size, beam_size, dtype='int32'), beam_size)

    np_hyp, np_word, np_values = numpy_topk(scores, k=beam_size, offset=offset)
    np_hyp, np_word, np_values = np_hyp.asnumpy(), np_word.asnumpy(), np_values.asnumpy()

    topk = sockeye.beam_search.TopK(k=beam_size)
    topk.initialize()

    mx_hyp, mx_word, mx_values = topk(scores, offset)
    mx_hyp, mx_word, mx_values = mx_hyp.asnumpy(), mx_word.asnumpy(), mx_values.asnumpy()
    assert np.allclose(mx_hyp, np_hyp)
    assert np.allclose(mx_word, np_word)
    assert np.allclose(mx_values, np_values)

    topk.hybridize()
    mx_hyp, mx_word, mx_values = topk(scores, offset)
    mx_hyp, mx_word, mx_values = mx_hyp.asnumpy(), mx_word.asnumpy(), mx_values.asnumpy()
    assert np.allclose(mx_hyp, np_hyp)
    assert np.allclose(mx_word, np_word)
    assert np.allclose(mx_values, np_values)


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size, top_n",
                        [(1, 5, 200, 0),
                         (5, 5, 200, 0),
                         (1, 100, 200, 5),
                         (5, 100, 200, 5)])
def test_samplek_func(batch_size, beam_size, target_vocab_size, top_n):
    # arrange scores increasing values from left to right, so the best item is always index 0, next-best 1, and so on
    scores = mx.nd.array([list(range(1, target_vocab_size + 1)) for _ in range(batch_size * beam_size)])
    # normalize
    target_dists = mx.nd.broadcast_div(scores, scores.sum(axis=1, keepdims=True))

    samplek = sockeye.beam_search.SampleK(n=top_n)
    samplek.initialize()

    sample_best_hyp_indices = mx.nd.arange(0, batch_size * beam_size, dtype='int32')

    # 0..(batch_size * beam_size)-1
    expected_hyps = mx.nd.array(range(batch_size * beam_size), dtype='int32')
    finished = mx.nd.cast(mx.nd.random.uniform(0, 1, (batch_size * beam_size)) > 0.5, dtype='int32')

    for i in [1, 2]:
        if i == 2:
            samplek.hybridize()

        hyps, words, values = samplek(scores, scores, finished, sample_best_hyp_indices)
        assert hyps.shape[0] == batch_size * beam_size

        # The indices should always be the integers from 0 to batch*beam-1
        assert sum(hyps == expected_hyps).asscalar() == (batch_size * beam_size)
        if top_n != 0:
            # Scores are increasing left-to-right, so best items are all the lowest word IDs.
            # No word id greater than the cap (top_n) should be selected
            assert mx.nd.sum(words >= top_n)[0].asscalar() == 0

        # word index should be zero for all finished hypotheses
        assert mx.nd.sum(mx.nd.where(finished, words, finished))[0].asscalar() == 0


class _TestInference(sockeye.beam_search._Inference):

    def __init__(self, output_vocab_size: int, predictor = dict()):
        self.output_vocab_size = output_vocab_size
        self.predictor = predictor
        self.states = []

    def encode_and_initialize(self,
                              inputs: mx.nd.NDArray,
                              valid_length: Optional[mx.nd.NDArray] = None):
        batch_size = inputs.shape[0]
        # 'lengths'
        internal_lengths = mx.nd.zeros((batch_size, 1), dtype='int32')
        self.states = [internal_lengths]  # TODO add nested states
        predicted_output_length = mx.nd.ones((batch_size, 1))  # does that work?
        return self.states, predicted_output_length

    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        batch_beam_size = step_input.shape[0]
        print('step_input', step_input.asnumpy())

        outputs = mx.nd.array([self.predictor.get(inp, C.PAD_ID) for inp in step_input.asnumpy().tolist()], ctx=step_input.context)
        scores = mx.nd.one_hot(outputs, depth=self.output_vocab_size)
        print('scores', scores.asnumpy())

        #scores = mx.nd.random.uniform(0, 1, (batch_beam_size, self.output_vocab_size))
        states[0] += 1
        self.states = states
        return scores, states


# TODO make this a useful test
# TODO: add vocabulary selection test
def test_beam_search():
    context = mx.cpu()
    dtype='float32'
    num_source_factors = 1
    output_vocab_size = 4
    beam_size = 5
    bos_id = 2
    eos_id = 3

    inference = _TestInference(output_vocab_size=output_vocab_size, predictor={2: 3})
    bs = sockeye.beam_search.BeamSearch(
        beam_size=beam_size,
        bos_id=bos_id,
        eos_id=eos_id,
        context=context,
        output_vocab_size=output_vocab_size,
        scorer=sockeye.beam_search.CandidateScorer(),
        num_source_factors=num_source_factors,
        inference=inference,
        beam_search_stop=C.BEAM_SEARCH_STOP_ALL,
        global_avoid_trie=None,
        sample=None)

    # inputs
    batch_size = 1
    max_length = 5
    source = mx.nd.array([[1, 2, 3, 4, 5]], ctx=context, dtype=dtype).reshape((0, -1, 1))
    source_length = mx.nd.array([max_length], ctx=context, dtype=dtype)

    restrict_lexicon = None
    raw_constraints = [None] * batch_size
    raw_avoid_list = [None] * batch_size
    max_output_lengths = mx.nd.array([max_length], ctx=context, dtype='int32')  # TODO

    bs_out = bs(source, source_length, restrict_lexicon, raw_constraints, raw_avoid_list, max_output_lengths)

    best_hyp_indices, best_word_indices, scores, lengths, estimated_ref_lengths, constraints = bs_out

    #best_ids = np.arange(0, batch_size * beam_size, beam_size, dtype='int32')
    #sockeye.inference.Translator._get_best_word_indices_for_kth_hypotheses(best_ids, best_hyp_indices)

    print(lengths)
    print(inference.states[0].asnumpy())
    #assert np.allclose(lengths, inference.states[0].asnumpy())

    #assert False


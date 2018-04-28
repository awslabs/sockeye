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

import mxnet as mx
import numpy as np
import pytest

import unittest

import sockeye.inference
from unittest.mock import patch

"""
Unit testing for inference.Translator._beam_prune()

Tests: take in
- accumulated_scores and finished
- a dummy inactive (not read from)
- best_word_indices (maybe dummy?)
and check
- values of finished and invalid
- maybe values of best_word_indices
"""

"""
Test pruning. The best score is computed from the best finished item; all other items
whose scores are outside (best_item - threshold) are pruned, which means their spot in
`inactive` is set to 1
"""
# batch size, beam size, prune thresh, accumulated scores, finished, expected_inactive
prune_tests = [
    # no pruning because nothing is finished
    (1, 10, 0, list(range(10)), [0] * 10, [0] * 10),
    # top item finished, threshold of 0.5, so one everything except top inactive
    (1, 10, 0.5, list(range(10)), [1] + [0] * 9, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # same but here the threshold doesn't include the second item
    (1, 10, 1.5, list(range(10)), [1] + [0] * 9, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
    # finished item is in the middle
    (1, 5, 1.5, [10, 16, 4, 5, 8], [0, 0, 1, 0, 0], [1, 1, 0, 0, 1]),
    # multiple finished items, lowest in last position
    (1, 5, 1.5, [10, 16, 4, 5, 8], [1, 0, 0, 0, 1], [1, 1, 0, 0, 0]),
    # batch setting, so pruning only applies to the first sentence
    (2, 10, 1.5, list(range(20)), [1] + [0] * 19, [0, 0] + [1] * 8 + [0] * 10),
]

@pytest.fixture
def mock_translator(batch_size, beam_size, beam_prune):
    """
    Creates a fake translator object but with real values for things that we need.
    This lets us avoid a messy call to the constructor.
    """
    with patch.object(sockeye.inference.Translator, '__init__', lambda self, **kwargs: None):
        translator = sockeye.inference.Translator(context=None,
                                                  ensemble_mode=None,
                                                  bucket_source_width=None,
                                                  length_penalty=None,
                                                  beam_prune=None,
                                                  beam_search_stop=None,
                                                  models=None,
                                                  source_vocabs=None,
                                                  target_vocab=None,
                                                  restrict_lexicon=None,
                                                  store_beam=None,
                                                  strip_unknown_words=None)
        translator.batch_size = batch_size
        translator.beam_size = beam_size
        translator.beam_prune = beam_prune
        translator.zeros_array = mx.nd.zeros((beam_size,), dtype='int32')
        translator.inf_array_long = mx.nd.full((batch_size * beam_size,), val=np.inf, dtype='float32')
        translator.inf_array = mx.nd.slice(translator.inf_array_long, begin=(0), end=(beam_size))
        return translator

@pytest.mark.parametrize("batch, beam, prune, scores, finished, expected_inactive", prune_tests)
def test_prune(batch, beam, prune, scores, finished, expected_inactive):
    translator = mock_translator(batch, beam, prune)

    orig_finished = [x for x in finished]

    # these are passed by reference and changed, so create them here
    scores = mx.nd.array(scores).expand_dims(axis=1)
    inactive = mx.nd.array([0] * (batch * beam), dtype='int32')
    best_words = mx.nd.array([10] * (batch * beam), dtype='int32')
    finished = mx.nd.array(finished, dtype='int32')

    translator._prune(scores, best_words, inactive, finished)

    # Make sure inactive is set as expected
    assert inactive.asnumpy().tolist() == expected_inactive

    # Ensure that scores for inactive items are set to 'inf'
    zeros = mx.nd.zeros((beam * batch,), dtype='float32')
    assert mx.nd.where(inactive, scores[:, 0], zeros).asnumpy().tolist() == [np.inf if x == 1 else 0 for x in expected_inactive]

    # Inactive items should also be marked as finished
    assert finished.asnumpy().tolist() == np.clip(np.array(orig_finished) + np.array(expected_inactive), 0, 1).tolist()

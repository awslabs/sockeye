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

import json
from unittest.mock import patch, Mock

import mxnet as mx
import numpy as np
import pytest
from math import ceil

import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
from sockeye.utils import SockeyeError

_BOS = 0
_EOS = -1


@pytest.fixture
def mock_translator(batch_size: int = 1,
                    beam_size: int = 5,
                    beam_prune: float = 0,
                    num_source_factors: int = 1):
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

        # This is needed for returning the right number of source factors
        def mock_model():
            t_mock = Mock(sockeye.inference.InferenceModel)
            t_mock.num_source_factors = num_source_factors
            return t_mock
        translator.models = [ mock_model() ]

        translator.batch_size = batch_size
        translator.beam_size = beam_size
        translator.beam_prune = beam_prune
        translator.zeros_array = mx.nd.zeros((beam_size,), dtype='int32')
        translator.inf_array_long = mx.nd.full((batch_size * beam_size,), val=np.inf, dtype='float32')
        translator.inf_array = mx.nd.slice(translator.inf_array_long, begin=(0), end=(beam_size))
        return translator


def test_concat_translations():
    expected_target_ids = [0, 1, 2, 8, 9, 3, 4, 5, -1]
    num_src = 7

    def length_penalty(length):
        return 1. / length

    expected_score = (1 + 2 + 3) / length_penalty(len(expected_target_ids))

    translations = [sockeye.inference.Translation([0, 1, 2, -1], np.zeros((4, num_src)), 1.0 / length_penalty(4)),
                    # Translation without EOS
                    sockeye.inference.Translation([0, 8, 9], np.zeros((3, num_src)), 2.0 / length_penalty(3)),
                    sockeye.inference.Translation([0, 3, 4, 5, -1], np.zeros((5, num_src)), 3.0 / length_penalty(5))]
    combined = sockeye.inference._concat_translations(translations, start_id=_BOS, stop_ids={_EOS},
                                                      length_penalty=length_penalty)

    assert combined.target_ids == expected_target_ids
    assert combined.attention_matrix.shape == (len(expected_target_ids), len(translations) * num_src)
    assert np.isclose(combined.score, expected_score)


def test_length_penalty_default():
    lengths = mx.nd.array([[1], [2], [3]])
    length_penalty = sockeye.inference.LengthPenalty(1.0, 0.0)
    expected_lp = np.array([[1.0], [2.], [3.]])

    assert np.isclose(length_penalty(lengths).asnumpy(), expected_lp).all()


def test_length_penalty():
    lengths = mx.nd.array([[1], [2], [3]])
    length_penalty = sockeye.inference.LengthPenalty(.2, 5.0)
    expected_lp = np.array([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])

    assert np.isclose(length_penalty(lengths).asnumpy(), expected_lp).all()


def test_length_penalty_int_input():
    length = 1
    length_penalty = sockeye.inference.LengthPenalty(.2, 5.0)
    expected_lp = [6 ** 0.2 / 6 ** 0.2]

    assert np.isclose(np.asarray([length_penalty(length)]),
                      np.asarray(expected_lp)).all()


@pytest.mark.parametrize("sentence_id, sentence, factors, chunk_size",
                         [(1, "a test", None, 4),
                          (1, "a test", None, 2),
                          (1, "a test", None, 1),
                          (0, "", None, 1),
                          (1, "a test", [['h', 'l']], 4),
                          (1, "a test", [['h', 'h'], ['x', 'y']], 1)])
def test_translator_input(sentence_id, sentence, factors, chunk_size):
    tokens = sentence.split()
    trans_input = sockeye.inference.TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)

    assert trans_input.sentence_id == sentence_id
    assert trans_input.tokens == tokens
    assert len(trans_input) == len(tokens)
    assert trans_input.factors == factors
    if factors is not None:
        for factor in trans_input.factors:
            assert len(factor) == len(tokens)
    assert trans_input.chunk_id == -1

    chunked_inputs = list(trans_input.chunks(chunk_size))
    assert len(chunked_inputs) == ceil(len(tokens) / chunk_size)
    for chunk_id, chunk_input in enumerate(chunked_inputs):
        assert chunk_input.sentence_id == sentence_id
        assert chunk_input.chunk_id == chunk_id
        assert chunk_input.tokens == trans_input.tokens[chunk_id * chunk_size: (chunk_id + 1) * chunk_size]
        if factors:
            assert len(chunk_input.factors) == len(factors)
            for factor, expected_factor in zip(chunk_input.factors, factors):
                assert len(factor) == len(chunk_input.tokens)
                assert factor == expected_factor[chunk_id * chunk_size: (chunk_id + 1) * chunk_size]


@pytest.mark.parametrize("supported_max_seq_len_source, supported_max_seq_len_target, training_max_seq_len_source, "
                         "forced_max_input_len, length_ratio_mean, length_ratio_std, "
                         "expected_max_input_len, expected_max_output_len",
                         [
                             (100, 100, 100, None, 0.9, 0.2, 89, 100),
                             (100, 100, 100, None, 1.1, 0.2, 75, 100),
                             # No source length constraints.
                             (None, 100, 100, None, 0.9, 0.1, 98, 100),
                             # No target length constraints.
                             (80, None, 100, None, 1.1, 0.4, 80, 122),
                             # No source/target length constraints. Source is max observed during training and target
                             # based on length ratios.
                             (None, None, 100, None, 1.0, 0.1, 100, 113),
                             # Force a maximum input length.
                             (100, 100, 100, 50, 1.1, 0.2, 50, 67),
                         ])
def test_get_max_input_output_length(
        supported_max_seq_len_source,
        supported_max_seq_len_target,
        training_max_seq_len_source,
        forced_max_input_len,
        length_ratio_mean,
        length_ratio_std,
        expected_max_input_len,
        expected_max_output_len):
    max_input_len, get_max_output_len = sockeye.inference.get_max_input_output_length(
        supported_max_seq_len_source=supported_max_seq_len_source,
        supported_max_seq_len_target=supported_max_seq_len_target,
        training_max_seq_len_source=training_max_seq_len_source,
        forced_max_input_len=forced_max_input_len,
        length_ratio_mean=length_ratio_mean,
        length_ratio_std=length_ratio_std,
        num_stds=1)
    max_output_len = get_max_output_len(max_input_len)

    if supported_max_seq_len_source is not None:
        assert max_input_len <= supported_max_seq_len_source
    if supported_max_seq_len_target is not None:
        assert max_output_len <= supported_max_seq_len_target
    if expected_max_input_len is not None:
        assert max_input_len == expected_max_input_len
    if expected_max_output_len is not None:
        assert max_output_len == expected_max_output_len


@pytest.mark.parametrize("sentence, num_expected_factors, delimiter, expected_tokens, expected_factors",
                         [
                             # sentence with single factor
                             ("this is a test", 1, "|", ["this", "is", "a", "test"], None),
                             # sentence with additional factor-like tokens, but no additional factors expected
                             ("this|X is| a|X test|", 1, "|", ["this|X", "is|", "a|X", "test|"], None),
                             # multiple spaces between token sequence
                             ("space   space", 1, "|", ["space", "space"], None),
                             # empty token sequence
                             ("", 1, "|", [], None),
                             ("", 2, "|", [], [[]]),
                             # proper factored sequences
                             ("a|l b|l C|u", 2, "|", ["a", "b", "C"], [["l", "l", "u"]]),
                             ("a-X-Y b-Y-X", 3, "-", ["a", "b"], [["X", "Y"], ["Y", "X"]]),
                             ("a-X-Y ", 3, "-", ["a"], [["X"], ["Y"]])
                         ])
def test_make_input_from_factored_string(sentence, num_expected_factors, delimiter,
                                         expected_tokens, expected_factors):
    sentence_id = 1
    translator = mock_translator(num_source_factors=num_expected_factors)

    inp = sockeye.inference.make_input_from_factored_string(sentence_id=sentence_id, factored_string=sentence,
                                                            translator=translator, delimiter=delimiter)
    assert isinstance(inp, sockeye.inference.TranslatorInput)
    assert inp.sentence_id == sentence_id
    assert inp.chunk_id == -1
    assert inp.tokens == expected_tokens
    assert inp.factors == expected_factors
    if num_expected_factors > 1:
        assert len(inp.factors) == num_expected_factors - 1


@pytest.mark.parametrize("sentence, num_expected_factors, delimiter",
                         [
                             ("this is a test", 2, "|"),  # expecting additional factor
                             ("this|X is a test", 2, "|"),  # expecting additional factor
                             ("this|X is|X a|X test", 2, "|"),  # fail on last token without factor
                             ("this| is|X a|X test|", 2, "|"),  # first token with delimiter but no factor
                             ("this|X is|X a|X test|", 2, "|"),  # last token with delimiter but no factor
                             ("w1||w2||f22", 2, "|"),
                             ("this", 2, "|"),  # single token without factor
                             ("this|", 2, "|"),  # single token with delimiter but no factor
                             ("this||", 3, "|"),  # double delimiter
                             ("this|| another", 2, "|"),  # double delimiter followed by token
                             ("this|||", 2, "|"),  # triple delimiter
                             ("|this", 2, "|"),  # empty token with 1 additional factor
                             ("|this|that", 3, "|"),  # empty token with 2 additional factors
                             ("|this|that|", 4, "|")  # empty token with 3 additional factors
                         ])
def test_factor_parsing(sentence, num_expected_factors, delimiter):
    """
    Test to ensure we fail on parses with invalid factors.
    """
    sentence_id = 1
    translator = mock_translator(num_source_factors=num_expected_factors)
    inp = sockeye.inference.make_input_from_factored_string(sentence_id=sentence_id,
                                                            factored_string=sentence,
                                                            translator=translator, delimiter=delimiter)
    assert isinstance(inp, sockeye.inference.BadTranslatorInput)


@pytest.mark.parametrize("delimiter", ["\t", "\t \t", "\t\t", "\n", "\r", "\r\n", "\u0020",
                                       "\n\n", "  ", " \t", "\f", "\v", "\u00a0", "\u1680",
                                       "\u2000", None, "", "\u200a", "\u205f", "\u3000"])
def test_make_input_whitespace_delimiter(delimiter):
    """
    Test to ensure we disallow a variety of whitespace strings as factor delimiters.
    """
    sentence_id = 1
    translator = mock_translator(num_source_factors=2)
    sentence = "foo"
    with pytest.raises(SockeyeError) as e:
        sockeye.inference.make_input_from_factored_string(sentence_id=sentence_id,
                                                          factored_string=sentence,
                                                          translator=translator, delimiter=delimiter)
    assert str(e.value) == 'Factor delimiter can not be whitespace or empty.'


@pytest.mark.parametrize("text, factors",
                         [("this is a test without factors", None),
                          ("", None),
                          ("test", ["X", "X"]),
                          ("a b c", ["x y z"]),
                          ("a", [])])
def test_make_input_from_valid_json_string(text, factors):
    sentence_id = 1
    expected_tokens = list(sockeye.data_io.get_tokens(text))
    inp = sockeye.inference.make_input_from_json_string(sentence_id, json.dumps({C.JSON_TEXT_KEY: text,
                                                                                 C.JSON_FACTORS_KEY: factors}))
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    if factors is not None:
        assert len(inp.factors) == len(factors)
    else:
        assert inp.factors is None


@pytest.mark.parametrize("text, text_key, factors, factors_key", [("a", "blub", None, "")])
def test_failed_make_input_from_valid_json_string(text, text_key, factors, factors_key):
    sentence_id = 1
    inp = sockeye.inference.make_input_from_json_string(sentence_id, json.dumps({text_key: text, factors_key: factors}))
    assert isinstance(inp, sockeye.inference.BadTranslatorInput)


@pytest.mark.parametrize("strings",
                         [
                             ["a b c"],
                             ["a b c", "f1 f2 f3", "f3 f3 f3"]
                         ])
def test_make_input_from_multiple_strings(strings):
    inp = sockeye.inference.make_input_from_multiple_strings(1, strings)

    expected_tokens = list(sockeye.data_io.get_tokens(strings[0]))
    expected_factors = [list(sockeye.data_io.get_tokens(f)) for f in strings[1:]]
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    assert inp.factors == expected_factors

"""
Test pruning via inference.Translator._beam_prune(). The best score is computed from the best
finished item; all other items whose scores are outside (best_item - threshold) are pruned, which
means their spot in `inactive` is set to 1.

Tests: take in
- accumulated_scores and finished
- a dummy inactive (not read from)
- best_word_indices (maybe dummy?)
and check
- values of finished and invalid
- maybe values of best_word_indices
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


@pytest.mark.parametrize("batch, beam, prune, scores, finished, expected_inactive", prune_tests)
def test_beam_prune(batch, beam, prune, scores, finished, expected_inactive):
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

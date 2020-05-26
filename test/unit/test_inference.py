# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import itertools
import json
from math import ceil
from unittest.mock import patch, Mock

import mxnet as mx
import numpy as np
import pytest

import sockeye.beam_search
import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
import sockeye.lexical_constraints
import sockeye.lexicon
import sockeye.model
import sockeye.utils

_BOS = 0
_EOS = -1


def mock_translator(batch_size: int = 1,
                    beam_size: int = 5,
                    nbest_size: int = 1,
                    num_source_factors: int = 1):
    """
    Creates a fake translator object but with real values for things that we need.
    This lets us avoid a messy call to the constructor.
    """
    with patch.object(sockeye.inference.Translator, '__init__', lambda self, **kwargs: None):
        translator = sockeye.inference.Translator(context=None,
                                                  batch_size=None,
                                                  beam_size=None,
                                                  ensemble_mode=None,
                                                  scorer=None,
                                                  beam_search_stop=None,
                                                  nbest_size=None,
                                                  models=None,
                                                  source_vocabs=None,
                                                  target_vocab=None,
                                                  restrict_lexicon=None,
                                                  strip_unknown_words=None)

        # This is needed for returning the right number of source factors
        def mock_model():
            t_mock = Mock(sockeye.model.SockeyeModel)
            t_mock.num_source_factors = num_source_factors
            return t_mock

        translator.batch_size = batch_size
        translator.beam_size = beam_size
        translator.nbest_size = nbest_size
        translator.models = [mock_model()]
        translator.zeros_array = mx.nd.zeros((beam_size,), dtype='int32')
        translator.inf_array = mx.nd.full((batch_size * beam_size,), val=np.inf, dtype='float32')
        translator.inf_array = mx.nd.slice(translator.inf_array, begin=(0,), end=(beam_size,))
        translator.restrict_lexicon = None
        return translator


@pytest.mark.parametrize("lp_alpha, lp_beta, bp_weight",
                         [(1.0, 0.0, 0.0),  # no LP and no BP (default)
                          (1.0, 2.0, 0.0),  # LP and no BP
                          (1.0, 2.0, 4.0),  # LP and BP
                          (1.0, 0.0, 5.0)]) # no LP and BP
def test_concat_translations(lp_alpha: float, lp_beta: float, bp_weight: float):
    beam_history1 = {"id": [1]}
    beam_history2 = {"id": [2]}
    beam_history3 = {"id": [3]}
    expected_beam_histories = [beam_history1, beam_history2, beam_history3]
    expected_target_ids = [0, 1, 2, 0, 8, 9, 0, 3, 4, 5, -1]

    scorer = sockeye.beam_search.CandidateScorer(lp_alpha, lp_beta, bp_weight)

    raw_score = (1 + 2 + 3)
    length = len(expected_target_ids)
    reference_length = (10 + 11 + 12)
    expected_score = scorer(raw_score, length, reference_length)
    # expected_score = (1 + 2 + 3) / length_penalty.get(len(expected_target_ids)) - \
    #                  brevity_penalty.get(len(expected_target_ids), 10 + 11 + 12)
    translations = [sockeye.inference.Translation([0, 1, 2, -1],
                                                  scorer(1.0, 4, 10),
                                                  [beam_history1],
                                                  None,
                                                  10),
                    # Translation without EOS
                    sockeye.inference.Translation([0, 8, 9],
                                                  scorer(2.0, 3, 11),
                                                  [beam_history2],
                                                  None,
                                                  11),
                    sockeye.inference.Translation([0, 3, 4, 5, -1],
                                                  scorer(3.0, 5, 12),
                                                  [beam_history3],
                                                  None,
                                                  12)]
    combined = sockeye.inference._concat_translations(translations, stop_ids={_EOS}, scorer=scorer)

    assert combined.target_ids == expected_target_ids
    assert np.isclose(combined.score, expected_score)
    assert combined.beam_histories == expected_beam_histories


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

    chunked_inputs = list(trans_input.chunks(chunk_size))
    assert len(chunked_inputs) == ceil(len(tokens) / chunk_size)
    for chunk_id, chunk_input in enumerate(chunked_inputs):
        assert chunk_input.sentence_id == sentence_id
        assert chunk_input.tokens == trans_input.tokens[chunk_id * chunk_size: (chunk_id + 1) * chunk_size]
        if factors:
            assert len(chunk_input.factors) == len(factors)
            for factor, expected_factor in zip(chunk_input.factors, factors):
                assert len(factor) == len(chunk_input.tokens)
                assert factor == expected_factor[chunk_id * chunk_size: (chunk_id + 1) * chunk_size]


@pytest.mark.parametrize("supported_max_seq_len_source, supported_max_seq_len_target, "
                         "forced_max_input_len, forced_max_output_len, length_ratio_mean, length_ratio_std, "
                         "expected_max_input_len, expected_max_output_len",
                         [
                             (99 + 1, 99 + 1, None, None, 1.0, 0.0, 100, 100),  # copy/sort test cases
                             (99 + 1, 99 + 1, None, None, 0.9, 0.2, 100, 111),  # target shorter than source
                             (99 + 1, 99 + 1, None, None, 1.1, 0.2, 100, 130),  # target longer than source
                             (99 + 1, 99 + 1, 50, None, 1.1, 0.2, 51, 67),  # force a maximum input length
                             (99 + 1, 99 + 1, 50, None, 1.1, 0.2, 51, 67),  # force a maximum input length
                             (99 + 1, 99 + 1, 50, 80, 1.1, 0.2, 51, 81),  # force a maximum input length
                         ])
def test_get_max_input_output_length(
        supported_max_seq_len_source,
        supported_max_seq_len_target,
        forced_max_input_len,
        forced_max_output_len,
        length_ratio_mean,
        length_ratio_std,
        expected_max_input_len,
        expected_max_output_len):
    max_input_len, get_max_output_len = sockeye.inference.get_max_input_output_length(
        supported_max_seq_len_source=supported_max_seq_len_source,
        supported_max_seq_len_target=supported_max_seq_len_target,
        forced_max_input_len=forced_max_input_len,
        forced_max_output_len=forced_max_output_len,
        length_ratio_mean=length_ratio_mean,
        length_ratio_std=length_ratio_std,
        num_stds=1)
    max_output_len = get_max_output_len(max_input_len)

    assert max_input_len <= supported_max_seq_len_source
    assert max_input_len == expected_max_input_len
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
    with pytest.raises(sockeye.utils.SockeyeError) as e:
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
    translator = mock_translator()
    expected_tokens = list(sockeye.data_io.get_tokens(text))
    inp = sockeye.inference.make_input_from_json_string(sentence_id,
                                                        json.dumps({C.JSON_TEXT_KEY: text,
                                                                    C.JSON_FACTORS_KEY: factors}),
                                                        translator)
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    if factors is not None:
        assert len(inp.factors) == len(factors)
    else:
        assert inp.factors is None


def test_make_input_from_valid_json_string_restrict_lexicon():
    sentence_id = 1
    text = 'this is a test'
    translator = mock_translator()

    lexicon1 = Mock(sockeye.lexicon.TopKLexicon)
    lexicon2 = Mock(sockeye.lexicon.TopKLexicon)
    translator.restrict_lexicon = {'lexicon1': lexicon1, 'lexicon2': lexicon2}
    assert translator.restrict_lexicon['lexicon1'] is not translator.restrict_lexicon['lexicon2']

    restrict_lexicon1 = 'lexicon1'
    inp1 = sockeye.inference.make_input_from_json_string(sentence_id,
                                                         json.dumps({C.JSON_TEXT_KEY: text,
                                                                     C.JSON_RESTRICT_LEXICON_KEY: restrict_lexicon1}),
                                                         translator)
    assert inp1.restrict_lexicon is lexicon1

    restrict_lexicon2 = 'lexicon2'
    inp2 = sockeye.inference.make_input_from_json_string(sentence_id,
                                                         json.dumps({C.JSON_TEXT_KEY: text,
                                                                     C.JSON_RESTRICT_LEXICON_KEY: restrict_lexicon2}),
                                                         translator)
    assert inp2.restrict_lexicon is lexicon2

    assert inp1.restrict_lexicon is not inp2.restrict_lexicon


@pytest.mark.parametrize("text, text_key, factors, factors_key", [("a", "blub", None, "")])
def test_failed_make_input_from_valid_json_string(text, text_key, factors, factors_key):
    sentence_id = 1
    translator = mock_translator()
    inp = sockeye.inference.make_input_from_json_string(sentence_id,
                                                        json.dumps({text_key: text, factors_key: factors}),
                                                        translator)
    assert isinstance(inp, sockeye.inference.BadTranslatorInput)


@pytest.mark.parametrize("text, factors",
                         [("this is a test without factors", None),
                          ("", None),
                          ("test", ["X", "X"]),
                          ("a b c", ["x y z"]),
                          ("a", [])])
def test_make_input_from_valid_dict(text, factors):
    sentence_id = 1
    translator = mock_translator()
    expected_tokens = list(sockeye.data_io.get_tokens(text))
    inp = sockeye.inference.make_input_from_dict(sentence_id, {C.JSON_TEXT_KEY: text,
                                                               C.JSON_FACTORS_KEY: factors}, translator)
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    if factors is not None:
        assert len(inp.factors) == len(factors)
    else:
        assert inp.factors is None


@pytest.mark.parametrize("text, text_key, factors, factors_key", [("a", "blub", None, "")])
def test_failed_make_input_from_valid_dict(text, text_key, factors, factors_key):
    sentence_id = 1
    translator = mock_translator()
    inp = sockeye.inference.make_input_from_dict(sentence_id, {text_key: text, factors_key: factors}, translator)
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


def test_get_best_word_indices_for_kth_hypotheses():
    # data
    all_hyp_indices = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 4, 3],
                                [0, 2, 2, 0, 1, 0, 0, 2, 1, 1, 3, 1, 1, 0, 1, 4, 0, 4],
                                [0, 1, 0, 1, 2, 1, 4, 3, 2, 3, 0, 4, 3, 1, 2, 1, 1, 0],
                                [0, 1, 0, 0, 3, 2, 2, 1, 3, 4, 4, 2, 2, 3, 3, 2, 2, 1],
                                [0, 2, 4, 1, 4, 2, 3, 4, 4, 2, 0, 3, 4, 4, 4, 3, 3, 2]], dtype='int32')
    ks = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    expected_indices = [np.array([[2, 1, 0, 0, 0, 0, 1, 3, 3, 2, 0, 0, 0, 1, 1, 2, 3]], dtype='int32'),
                        np.array([[1, 2, 1, 2, 2, 3, 4, 4, 4, 3, 1, 1, 1, 2, 2, 3, 4]], dtype='int32'),
                        np.array([[2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 2, 3, 3, 3, 4, 0]], dtype='int32'),
                        np.array([[2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 3, 2, 0, 0, 0, 1]], dtype='int32'),
                        np.array([[2, 1, 0, 1, 1, 2, 3, 2, 2, 4, 3, 4, 4, 4, 4, 1, 2]], dtype='int32')]

    # extract individually
    for k, expected_result in zip(ks, expected_indices):
        result = sockeye.inference.Translator._get_best_word_indices_for_kth_hypotheses(k, all_hyp_indices)
        assert result.shape == expected_result.shape
        assert (result == expected_result).all()

    # extract all at once
    ks = np.concatenate(ks, axis=0)
    expected_indices = np.concatenate(expected_indices, axis=0)
    result = sockeye.inference.Translator._get_best_word_indices_for_kth_hypotheses(ks, all_hyp_indices)
    assert result.shape == expected_indices.shape
    assert (result == expected_indices).all()


@pytest.mark.parametrize("raw_constraints, beam_histories, expected_best_ids, expected_best_indices",
                         [([[], [], [], []], [None, None], np.array([0, 2], dtype='int32'), np.array([[1, 1, 1], [3, 3, 3]], dtype='int32')),
                          ([[[1]], [], [[3]], []], [None, None], np.array([1, 3], dtype='int32'), np.array([[1, 0, 0], [3, 2, 2]], dtype='int32'))
                          ])
def test_get_best_from_beam(raw_constraints, beam_histories, expected_best_ids, expected_best_indices):
    best_hyp_indices = np.array([[0, 1, 0, 1],
                                 [0, 1, 1, 0],
                                 [2, 3, 2, 3],
                                 [2, 3, 3, 2]],
                                dtype='int32')
    best_word_indices = np.array([[3, 3, 0],
                                  [4, 4, 3],
                                  [3, 3, 0],
                                  [4, 5, 3]],
                                 dtype='int32')
    seq_scores = np.array([[3.8197377],
                           [5.081118 ],
                           [3.8068485],
                           [5.0746527]],
                          dtype='float32')
    lengths = np.array([[3], [2], [3], [2]], dtype='int32')

    translator = mock_translator(beam_size=2, batch_size=2)

    expected_result = [sockeye.inference.Translator._assemble_translation(*x) for x in zip(
                            best_word_indices[expected_best_indices, np.arange(expected_best_indices.shape[1])],
                            lengths[expected_best_ids],
                            seq_scores[expected_best_ids],
                            beam_histories,
                            itertools.repeat(None))]

    constraints = [sockeye.lexical_constraints.ConstrainedHypothesis(rc, _EOS) for rc in raw_constraints]

    actual_result = sockeye.inference.Translator._get_best_from_beam(translator,
                                                                     best_hyp_indices,
                                                                     best_word_indices,
                                                                     seq_scores,
                                                                     lengths,
                                                                     None,
                                                                     constraints,
                                                                     beam_histories)

    for expected_translation, actual_translation in zip(expected_result, actual_result):
        assert expected_translation.target_ids == actual_translation.target_ids
        assert expected_translation.score == actual_translation.score
        assert expected_translation.beam_histories == actual_translation.beam_histories

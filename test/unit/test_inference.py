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

import itertools
import json
from math import ceil
from unittest.mock import patch, Mock

import numpy as np
import pytest
import torch as pt

import sockeye.beam_search_pt
import sockeye.constants as C
import sockeye.data_io_pt
import sockeye.inference_pt
import sockeye.lexicon
import sockeye.model_pt
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
    with patch.object(sockeye.inference_pt.Translator, '__init__', lambda self, **kwargs: None):
        translator = sockeye.inference_pt.Translator(device=None,
                                                     batch_size=None,
                                                     beam_size=None,
                                                     ensemble_mode=None,
                                                     scorer=None,
                                                     beam_search_stop=None,
                                                     nbest_size=None,
                                                     models=None,
                                                     source_vocabs=None,
                                                     target_vocabs=None,
                                                     restrict_lexicon=None,
                                                     strip_unknown_words=None)

        # This is needed for returning the right number of source factors
        def mock_model():
            t_mock = Mock(sockeye.model_pt.PyTorchSockeyeModel)
            t_mock.num_source_factors = num_source_factors
            return t_mock

        translator.batch_size = batch_size
        translator.beam_size = beam_size
        translator.nbest_size = nbest_size
        translator.models = [mock_model()]
        translator.zeros_array = pt.zeros(beam_size, dtype=pt.int)
        translator.inf_array = pt.full((batch_size * beam_size,), fill_value=np.inf, dtype=pt.float32)
        translator.inf_array = translator.inf_array[:beam_size]
        translator.restrict_lexicon = None
        return translator


@pytest.mark.parametrize("lp_alpha, lp_beta, bp_weight",
                         [(1.0, 0.0, 0.0),  # no LP and no BP (default)
                          (1.0, 2.0, 0.0),  # LP and no BP
                          (1.0, 2.0, 4.0),  # LP and BP
                          (1.0, 0.0, 5.0)])  # no LP and BP
def test_concat_translations(lp_alpha: float, lp_beta: float, bp_weight: float):
    expected_target_ids = [[0], [1], [2], [0], [8], [9], [0], [3], [4], [5], [-1]]

    scorer = sockeye.beam_search_pt.CandidateScorer(lp_alpha, lp_beta, bp_weight)

    raw_score = (1 + 2 + 3)
    length = len(expected_target_ids)
    reference_length = (10 + 11 + 12)
    expected_score = [scorer(raw_score, length, reference_length)]
    # expected_score = (1 + 2 + 3) / length_penalty.get(len(expected_target_ids)) - \
    #                  brevity_penalty.get(len(expected_target_ids), 10 + 11 + 12)
    translations = [sockeye.inference_pt.Translation([[0], [1], [2], [-1]],
                                                     [scorer(1.0, 4, 10)],
                                                     None,
                                                     10),
                    # Translation without EOS
                    sockeye.inference_pt.Translation([[0], [8], [9]],
                                                     [scorer(2.0, 3, 11)],
                                                     None,
                                                     11),
                    sockeye.inference_pt.Translation([[0], [3], [4], [5], [-1]],
                                                     [scorer(3.0, 5, 12)],
                                                     None,
                                                     12)]
    combined = sockeye.inference_pt._concat_translations(translations, stop_ids={_EOS}, scorer=scorer)

    assert combined.target_ids == expected_target_ids
    assert np.isclose(combined.scores, expected_score)


@pytest.mark.parametrize("sentence_id, sentence, factors, chunk_size",
                         [(1, "a test", None, 4),
                          (1, "a test", None, 2),
                          (1, "a test", None, 1),
                          (0, "", None, 1),
                          (1, "a test", [['h', 'l']], 4),
                          (1, "a test", [['h', 'h'], ['x', 'y']], 1)])
def test_translator_input(sentence_id, sentence, factors, chunk_size):
    tokens = sentence.split()
    trans_input = sockeye.inference_pt.TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)

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
    max_input_len, get_max_output_len = sockeye.inference_pt.get_max_input_output_length(
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

    inp = sockeye.inference_pt.make_input_from_factored_string(sentence_id=sentence_id, factored_string=sentence,
                                                               translator=translator, delimiter=delimiter)
    assert isinstance(inp, sockeye.inference_pt.TranslatorInput)
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
    inp = sockeye.inference_pt.make_input_from_factored_string(sentence_id=sentence_id,
                                                               factored_string=sentence,
                                                               translator=translator, delimiter=delimiter)
    assert isinstance(inp, sockeye.inference_pt.BadTranslatorInput)


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
        sockeye.inference_pt.make_input_from_factored_string(sentence_id=sentence_id,
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
    expected_tokens = list(sockeye.data_io_pt.get_tokens(text))
    inp = sockeye.inference_pt.make_input_from_json_string(sentence_id,
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
    inp1 = sockeye.inference_pt.make_input_from_json_string(sentence_id,
                                                            json.dumps({C.JSON_TEXT_KEY: text,
                                                                        C.JSON_RESTRICT_LEXICON_KEY: restrict_lexicon1}),
                                                            translator)
    assert inp1.restrict_lexicon is lexicon1

    restrict_lexicon2 = 'lexicon2'
    inp2 = sockeye.inference_pt.make_input_from_json_string(sentence_id,
                                                            json.dumps({C.JSON_TEXT_KEY: text,
                                                                        C.JSON_RESTRICT_LEXICON_KEY: restrict_lexicon2}),
                                                            translator)
    assert inp2.restrict_lexicon is lexicon2

    assert inp1.restrict_lexicon is not inp2.restrict_lexicon


@pytest.mark.parametrize("text, text_key, factors, factors_key", [("a", "blub", None, "")])
def test_failed_make_input_from_valid_json_string(text, text_key, factors, factors_key):
    sentence_id = 1
    translator = mock_translator()
    inp = sockeye.inference_pt.make_input_from_json_string(sentence_id,
                                                           json.dumps({text_key: text, factors_key: factors}),
                                                           translator)
    assert isinstance(inp, sockeye.inference_pt.BadTranslatorInput)


@pytest.mark.parametrize("text, factors",
                         [("this is a test without factors", None),
                          ("", None),
                          ("test", ["X", "X"]),
                          ("a b c", ["x y z"]),
                          ("a", [])])
def test_make_input_from_valid_dict(text, factors):
    sentence_id = 1
    translator = mock_translator()
    expected_tokens = list(sockeye.data_io_pt.get_tokens(text))
    inp = sockeye.inference_pt.make_input_from_dict(sentence_id, {C.JSON_TEXT_KEY: text,
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
    inp = sockeye.inference_pt.make_input_from_dict(sentence_id, {text_key: text, factors_key: factors}, translator)
    assert isinstance(inp, sockeye.inference_pt.BadTranslatorInput)


@pytest.mark.parametrize("strings",
                         [
                             ["a b c"],
                             ["a b c", "f1 f2 f3", "f3 f3 f3"]
                         ])
def test_make_input_from_multiple_strings(strings):
    inp = sockeye.inference_pt.make_input_from_multiple_strings(1, strings)

    expected_tokens = list(sockeye.data_io_pt.get_tokens(strings[0]))
    expected_factors = [list(sockeye.data_io_pt.get_tokens(f)) for f in strings[1:]]
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
        result = sockeye.inference_pt.Translator._get_best_word_indices_for_kth_hypotheses(k, all_hyp_indices)
        assert result.shape == expected_result.shape
        assert (result == expected_result).all()

    # extract all at once
    ks = np.concatenate(ks, axis=0)
    expected_indices = np.concatenate(expected_indices, axis=0)
    result = sockeye.inference_pt.Translator._get_best_word_indices_for_kth_hypotheses(ks, all_hyp_indices)
    assert result.shape == expected_indices.shape
    assert (result == expected_indices).all()


@pytest.mark.parametrize("expected_best_ids, expected_best_indices",
                         [(np.array([0, 2], dtype='int32'),
                           np.array([[1, 1, 1], [3, 3, 3]], dtype='int32'))
                          ])
def test_get_best_translations(expected_best_ids, expected_best_indices):
    best_hyp_indices = pt.tensor([[0, 1, 0, 1],
                                  [0, 1, 1, 0],
                                  [2, 3, 2, 3],
                                  [2, 3, 3, 2]],
                                 dtype=pt.int32)
    best_word_indices = pt.tensor([[[3, 3, 0]],
                                   [[4, 4, 3]],
                                   [[3, 3, 0]],
                                   [[4, 5, 3]]],
                                  dtype=pt.int32)
    seq_scores = pt.tensor([[3.8197377],
                            [5.081118],
                            [3.8068485],
                            [5.0746527]],
                           dtype=pt.float32)
    lengths = pt.tensor([[3], [2], [3], [2]], dtype=pt.int32)

    translator = mock_translator(beam_size=2, batch_size=2)

    expected_result = [sockeye.inference_pt.Translator._assemble_translation(*x) for x in zip(
        best_word_indices[expected_best_indices, :, np.arange(expected_best_indices.shape[1])],
        lengths[expected_best_ids],
        seq_scores[expected_best_ids],
        itertools.repeat(None))]

    search_result = sockeye.beam_search_pt.SearchResult(best_hyp_indices=best_hyp_indices,
                                                        best_word_indices=best_word_indices,
                                                        accumulated_scores=seq_scores,
                                                        lengths=lengths,
                                                        estimated_reference_lengths=None)
    actual_result = sockeye.inference_pt.Translator._get_best_translations(translator, search_result)

    for expected_translation, actual_translation in zip(expected_result, actual_result):
        assert expected_translation.target_ids == actual_translation.target_ids
        assert expected_translation.scores == actual_translation.scores


@pytest.mark.parametrize("sequence, fill_with, expected_sequence",
                         [
                             (np.array([1, 2, 3]), C.EOS_ID, [1, 2, 3]),
                             (np.array([[1], [2], [3]]), C.EOS_ID, [[1], [2], [3]]),
                             (np.array([[1, 0], [2, 1], [3, 2]]), C.EOS_ID, [(1, 1), (2, 2), (3, C.EOS_ID)]),
                             (np.array([[1, 0], [2, 1], [3, 2]]), C.PAD_ID, [(1, 1), (2, 2), (3, C.PAD_ID)]),
                         ])
def test_unshift_target_factors(sequence, fill_with, expected_sequence):
    sequence = sockeye.inference_pt._unshift_target_factors(sequence, fill_last_with=fill_with)
    assert sequence == expected_sequence

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
from math import ceil

import sockeye.inference
from sockeye.utils import SockeyeError

_BOS = 0
_EOS = -1


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
                         [(1, "a test", [], 4),
                          (1, "a test", [], 2),
                          (1, "a test", [], 1),
                          (0, "", [], 1),
                          (1, "a test", [['h', 'l']], 4),
                          (1, "a test", [['h', 'h'], ['x', 'y']], 1)])
def test_translator_input(sentence_id, sentence, factors, chunk_size):
    tokens = sentence.split()
    trans_input = sockeye.inference.TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)

    assert trans_input.sentence_id == sentence_id
    assert trans_input.tokens == tokens
    assert trans_input.factors == factors
    for factor in trans_input.factors:
        assert len(factor) == len(tokens)
    assert trans_input.chunk_id == -1

    chunked_inputs = list(trans_input.chunks(chunk_size))
    assert len(chunked_inputs) == ceil(len(tokens) / chunk_size)
    for chunk_id, chunk_input in enumerate(chunked_inputs):
        assert chunk_input.sentence_id == sentence_id
        assert chunk_input.chunk_id == chunk_id
        assert chunk_input.tokens == trans_input.tokens[chunk_id * chunk_size: (chunk_id + 1) * chunk_size]
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


@pytest.mark.parametrize("sentence_id, raw_sentence, num_factors, delimiter, expected_tokens, expected_factors",
                         [
                             # sentence with single factor
                             (1, "this is a test", 1, "|", ["this", "is", "a", "test"], []),
                             # sentence with additional factor-like tokens, but no additional factors expected
                             (1, "this|X is| a|X test|", 1, "|", ["this|X", "is|", "a|X", "test|"], []),
                             # multiple spaces between token sequence
                             (1, "space   space", 1, "|", ["space", "space"], []),
                             # empty token sequence
                             (2, "", 1, "|", [], []),
                             (2, "", 2, "|", [], [[]]),
                             # proper factored sequences
                             (14, "a|l b|l C|u", 2, "|", ["a", "b", "C"], [["l", "l", "u"]]),
                             (1, "a-X-Y b-Y-X", 3, "-", ["a", "b"], [["X", "Y"], ["Y", "X"]]),
                             (1, "a-X-Y ", 2, "-", ["a-X"], [["Y"]])
                         ])
def test_make_input(sentence_id, raw_sentence, num_factors, delimiter, expected_tokens, expected_factors):
    translator_input = sockeye.inference.Translator.make_input(sentence_id=sentence_id,
                                                               sentence=raw_sentence,
                                                               num_factors=num_factors,
                                                               delimiter=delimiter)
    assert isinstance(translator_input, sockeye.inference.TranslatorInput)
    assert translator_input.sentence_id == sentence_id
    assert translator_input.chunk_id == -1
    assert translator_input.tokens == expected_tokens
    assert len(translator_input.factors) == num_factors - 1
    assert translator_input.factors == expected_factors


@pytest.mark.parametrize("sentence, num_factors, delimiter, expected_wrong_num_factors, expected_fail_word",
                         [("this is a test", 2, "|", 1, "this"),  # expecting additional factor
                          ("this|X is a test", 2, "|", 1, "is"),  # expecting additional factor
                          ("this|X is|X a|X test", 2, "|", 1, "test"),  # fail on last token without factor
                          ("this| is|X a|X test|", 2, "|", 1, "this"),  # first token with delimiter but no factor
                          ("this|X is|X a|X test|", 2, "|", 1, "test"),  # last token with delimiter but no factor
                          ("this", 2, "|", 1, "this"),  # single token without factor
                          ("this|", 2, "|", 1, "this"),  # single token with delimiter but no factor
                          ("this||", 3, "|", 1, "this"),  # double delimiter
                          ("this|| another", 2, "|", 1, "this|"),  # double delimiter followed by token
                          ("this|||", 2, "|", 1, "this||")])  # triple delimiter
def test_make_input_factor_parsing(sentence, num_factors, delimiter, expected_wrong_num_factors, expected_fail_word):
    """
    Test to ensure we fail on parses with invalid factors.
    """
    sentence_id = 1
    with pytest.raises(SockeyeError) as e:
        sockeye.inference.Translator.make_input(sentence_id=1,
                                                sentence=sentence,
                                                num_factors=num_factors,
                                                delimiter=delimiter)
    assert str(e.value) == 'Expecting %d factors, but got %d at sentence %d, word "%s"' % (num_factors,
                                                                                           expected_wrong_num_factors,
                                                                                           sentence_id,
                                                                                           expected_fail_word)


@pytest.mark.parametrize("sentence, num_factors, delimiter, expected_position",
                         [
                             ("|this", 2, "|", 0),  # empty token with 1 additional factor
                             ("|this|that", 3, "|", 0),  # empty token with 2 additional factors
                             ("|this|that|", 4, "|", 0)  # empty token with 3 additional factors
                         ])
def test_make_input_emtpy_token(sentence, num_factors, delimiter, expected_position):
    """
    Test to ensure we fail on parses that create empty tokens.
    """
    sentence_id = 1
    with pytest.raises(SockeyeError) as e:
        sockeye.inference.Translator.make_input(sentence_id=sentence_id,
                                                sentence=sentence,
                                                num_factors=num_factors,
                                                delimiter=delimiter)
    assert str(e.value) == 'Empty token at sentence %d, position %d' % (sentence_id, expected_position)


@pytest.mark.parametrize("delimiter", ["\t", "\t \t", "\t\t", "\n", "\r", "\r\n", "\u0020",
                                       "\n\n", "  ", " \t", "\f", "\v", "\u00a0", "\u1680",
                                       "\u2000", None, "", "\u200a", "\u205f", "\u3000"])
def test_make_input_whitespace_delimiter(delimiter):
    """
    Test to ensure we disallow a variety of whitespace strings as factor delimiters.
    """
    sentence_id = 1
    sentence = "foo"
    num_factors = 2
    with pytest.raises(SockeyeError) as e:
        sockeye.inference.Translator.make_input(sentence_id=sentence_id,
                                                sentence=sentence,
                                                num_factors=num_factors,
                                                delimiter=delimiter)
    assert str(e.value) == 'Factor delimiter can not be whitespace or empty.'

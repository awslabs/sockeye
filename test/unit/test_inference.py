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

import sockeye.inference
from sockeye.utils import SockeyeError

_BOS = 0
_EOS = -1


def test_concat_translations():
    expected_target_ids = [0, 1, 2, 8, 9, 3, 4, 5, -1]
    NUM_SRC = 7

    def length_penalty(length):
        return 1. / length

    expected_score = (1 + 2 + 3) / length_penalty(len(expected_target_ids))

    translations = [sockeye.inference.Translation([0, 1, 2, -1], np.zeros((4, NUM_SRC)), 1.0 / length_penalty(4)),
                    # Translation without EOS
                    sockeye.inference.Translation([0, 8, 9], np.zeros((3, NUM_SRC)), 2.0 / length_penalty(3)),
                    sockeye.inference.Translation([0, 3, 4, 5, -1], np.zeros((5, NUM_SRC)), 3.0 / length_penalty(5))]
    combined = sockeye.inference._concat_translations(translations, start_id=_BOS, stop_ids={_EOS},
                                                      length_penalty=length_penalty)

    assert combined.target_ids == expected_target_ids
    assert combined.attention_matrix.shape == (len(expected_target_ids), len(translations) * NUM_SRC)
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
                         [(1, "this is a test", 0, "|", ["this", "is", "a", "test"], []),
                          (1, "space   space", 0, "|", ["space", "space"], []),
                          (2, "", 0, "|", [], []),
                          (14, "a|l b|l C|u", 1, "|", ["a", "b", "C"], [["l", "l", "u"]]),
                          (1, "a-X-Y ", 2, "-", ["a"], [["X"], ["Y"]]),
                          (1, "a-X-Y ", 1, "-", ["a-X"], [["Y"]])])
def test_make_input(sentence_id, raw_sentence, num_factors, delimiter, expected_tokens, expected_factors):
    translator_input = sockeye.inference.Translator.make_input(sentence_id=sentence_id,
                                                               raw_sentence=raw_sentence,
                                                               num_factors=num_factors,
                                                               delimiter=delimiter)
    assert isinstance(translator_input, sockeye.inference.TranslatorInput)
    assert translator_input.sentence_id == sentence_id
    assert translator_input.chunk_id == 0
    assert translator_input.tokens == expected_tokens
    assert len(translator_input.factors) == num_factors
    assert translator_input.factors == expected_factors


@pytest.mark.parametrize("sentence, num_factors, delimiter, expected_wrong_num_factors, expected_fail_word",
                         [("this is a test", 1, "|", 0, "this"),
                          ("this is", 1, " ", 0, "this"),
                          ("this", 1, "|", 0, "this"),
                          ("this|X is| a|", 1, "|", 1, "is"),
                          ("this||", 1, "|", 1, "is"),
                          ("this|| another", 1, "|", 0, "another"),
                          ("this|||", 1, "|", 0, "this"),
                          ("|this", 1, "|", 0, "this"),
                          (r"this\tX is\tX", 1, r"\t", 0, "this"),
                          ])
def test_failed_make_input(sentence, num_factors, delimiter, expected_wrong_num_factors, expected_fail_word):
    sentence_id = 1
    with pytest.raises(SockeyeError) as e:
        sockeye.inference.Translator.make_input(sentence_id=sentence_id,
                                                raw_sentence=sentence,
                                                num_factors=num_factors,
                                                delimiter=delimiter)
    assert str(e.value) == 'Expecting %d factors, but got %d at sentence %d, word "%s"' % (num_factors,
                                                                                            expected_wrong_num_factors,
                                                                                            sentence_id,
                                                                                            expected_fail_word)

# TODO(fhieber): failure tests and other tests for factor parsing

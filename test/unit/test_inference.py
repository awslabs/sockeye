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
from math import ceil
from unittest.mock import patch, Mock

import mxnet as mx
import numpy as np
import pytest

import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
import sockeye.lexical_constraints
import sockeye.utils

_BOS = 0
_EOS = -1


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

        translator.models = [mock_model()]

        translator.batch_size = batch_size
        translator.beam_size = beam_size
        translator.beam_prune = beam_prune
        translator.zeros_array = mx.nd.zeros((beam_size,), dtype='int32')
        translator.inf_array = mx.nd.full((batch_size * beam_size,), val=np.inf, dtype='float32')
        translator.inf_array = mx.nd.slice(translator.inf_array, begin=(0), end=(beam_size))
        return translator


def test_concat_translations():
    beam_history1 = {"id": [1]}
    beam_history2 = {"id": [2]}
    beam_history3 = {"id": [3]}
    expected_beam_histories = [beam_history1, beam_history2, beam_history3]
    expected_target_ids = [0, 1, 2, 0, 8, 9, 0, 3, 4, 5, -1]
    num_src = 7

    length_penalty = sockeye.inference.LengthPenalty()

    expected_score = (1 + 2 + 3) / length_penalty.get(len(expected_target_ids))

    translations = [sockeye.inference.Translation([0, 1, 2, -1], np.zeros((4, num_src)), 1.0 / length_penalty.get(4),
                                                  [beam_history1]),
                    # Translation without EOS
                    sockeye.inference.Translation([0, 8, 9], np.zeros((3, num_src)), 2.0 / length_penalty.get(3),
                                                  [beam_history2]),
                    sockeye.inference.Translation([0, 3, 4, 5, -1], np.zeros((5, num_src)), 3.0 / length_penalty.get(5),
                                                  [beam_history3])]
    combined = sockeye.inference._concat_translations(translations, stop_ids={_EOS}, length_penalty=length_penalty)

    assert combined.target_ids == expected_target_ids
    assert combined.attention_matrix.shape == (len(expected_target_ids), len(translations) * num_src)
    assert np.isclose(combined.score, expected_score)
    assert combined.beam_histories == expected_beam_histories


def test_length_penalty_default():
    lengths = mx.nd.array([[1], [2], [3]])
    length_penalty = sockeye.inference.LengthPenalty(1.0, 0.0)
    expected_lp = np.array([[1.0], [2.], [3.]])

    assert np.isclose(length_penalty.get(lengths).asnumpy(), expected_lp).all()
    assert np.isclose(length_penalty(lengths).asnumpy(), expected_lp).all()
    length_penalty.hybridize()
    assert np.isclose(length_penalty(lengths).asnumpy(), expected_lp).all()


def test_length_penalty():
    lengths = mx.nd.array([[1], [2], [3]])
    length_penalty = sockeye.inference.LengthPenalty(.2, 5.0)
    expected_lp = np.array([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])

    assert np.isclose(length_penalty.get(lengths).asnumpy(), expected_lp).all()
    assert np.isclose(length_penalty(lengths).asnumpy(), expected_lp).all()
    length_penalty.hybridize()
    assert np.isclose(length_penalty(lengths).asnumpy(), expected_lp).all()


def test_length_penalty_int_input():
    length = 1
    length_penalty = sockeye.inference.LengthPenalty(.2, 5.0)
    expected_lp = [6 ** 0.2 / 6 ** 0.2]

    assert np.isclose(np.asarray([length_penalty.get(length)]), np.asarray(expected_lp)).all()


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
    scores = mx.nd.array(scores)
    finished = mx.nd.array(finished, dtype='int32')
    inf_array = mx.nd.full((batch * beam,), val=np.inf)
    zeros_array = mx.nd.zeros((batch * beam,), dtype='int32')
    best_word_indices = mx.nd.zeros((batch * beam,), dtype='int32')

    prune_hyps = sockeye.inference.PruneHypotheses(prune, beam)
    prune_hyps.initialize()
    inactive, _, _ = prune_hyps(best_word_indices, scores, finished, inf_array, zeros_array)
    assert inactive.asnumpy().tolist() == expected_inactive

    prune_hyps.hybridize()
    inactive, _, _ = prune_hyps(best_word_indices, scores, finished, inf_array, zeros_array)
    assert inactive.asnumpy().tolist() == expected_inactive


def test_sort_by_index():
    data = [mx.nd.random.uniform(0, 1, (3, i)) for i in range(1, 5)]
    indices = mx.nd.array([2, 0, 1], dtype='int32')
    expected = [d.asnumpy()[indices.asnumpy()] for d in data]

    sort_by_index = sockeye.inference.SortByIndex()
    sort_by_index.initialize()

    out = sort_by_index(indices, *data)
    assert len(out) == len(data) == len(expected)
    for o, e in zip(out, expected):
        assert (o.asnumpy() == e).all()

    sort_by_index.hybridize()
    out = sort_by_index(indices, *data)
    assert len(out) == len(data) == len(expected)
    for o, e in zip(out, expected):
        assert (o.asnumpy() == e).all()


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
    offset = mx.nd.array(np.repeat(np.arange(0, batch_size * beam_size, beam_size), beam_size), dtype='int32')

    np_hyp, np_word, np_values = sockeye.utils.topk(scores, k=beam_size,
                                                    offset=offset, use_mxnet_topk=False)
    np_hyp, np_word, np_values = np_hyp.asnumpy(), np_word.asnumpy(), np_values.asnumpy()

    mx_hyp, mx_word, mx_values = sockeye.utils.topk(scores, k=beam_size,
                                                    offset=offset, use_mxnet_topk=True)
    mx_hyp, mx_word, mx_values = mx_hyp.asnumpy(), mx_word.asnumpy(), mx_values.asnumpy()
    assert all(mx_hyp == np_hyp)
    assert all(mx_word == np_word)
    assert all(mx_values == np_values)

    topk = sockeye.inference.TopK(k=beam_size, batch_size=batch_size, vocab_size=target_vocab_size)
    topk.initialize()
    assert all(topk.offset.data() == offset)

    mx_hyp, mx_word, mx_values = topk(scores)
    mx_hyp, mx_word, mx_values = mx_hyp.asnumpy(), mx_word.asnumpy(), mx_values.asnumpy()
    assert all(mx_hyp == np_hyp)
    assert all(mx_word == np_word)
    assert all(mx_values == np_values)

    topk.hybridize()
    mx_hyp, mx_word, mx_values = topk(scores)
    mx_hyp, mx_word, mx_values = mx_hyp.asnumpy(), mx_word.asnumpy(), mx_values.asnumpy()
    assert all(mx_hyp == np_hyp)
    assert all(mx_word == np_word)
    assert all(mx_values == np_values)


def test_get_best_word_indeces_for_kth_hypotheses():
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
        result = sockeye.inference.Translator._get_best_word_indeces_for_kth_hypotheses(k, all_hyp_indices)
        assert result.shape == expected_result.shape
        assert (result == expected_result).all()

    # extract all at once
    ks = np.concatenate(ks, axis=0)
    expected_indices = np.concatenate(expected_indices, axis=0)
    result = sockeye.inference.Translator._get_best_word_indeces_for_kth_hypotheses(ks, all_hyp_indices)
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
    attentions = np.array([[[0.1748407 , 0.17223692, 0.153318  , 0.16618672, 0.15373373,
                             0.1796839 , 0.        , 0.        , 0.        , 0.        ],
                            [0.17484048, 0.17223585, 0.15332589, 0.16618879, 0.15374145,
                             0.17966755, 0.        , 0.        , 0.        , 0.        ],
                            [0.17483611, 0.17222905, 0.15335034, 0.16619477, 0.15375796,
                             0.17963174, 0.        , 0.        , 0.        , 0.        ]],
                           [[0.1748407 , 0.17223692, 0.153318  , 0.16618672, 0.15373373,
                             0.1796839 , 0.        , 0.        , 0.        , 0.        ],
                            [0.17484048, 0.17223585, 0.15332589, 0.16618879, 0.15374145,
                             0.17966755, 0.        , 0.        , 0.        , 0.        ],
                            [0.1748425 , 0.17223647, 0.15333334, 0.16618758, 0.15375413,
                             0.17964599, 0.        , 0.        , 0.        , 0.        ]],
                           [[0.20974289, 0.1808782 , 0.18161033, 0.20220006, 0.22556852,
                             0.        , 0.        , 0.        , 0.        , 0.        ],
                            [0.20973803, 0.18088503, 0.18162282, 0.20220187, 0.22555229,
                             0.        , 0.        , 0.        , 0.        , 0.        ],
                            [0.20973288, 0.18088858, 0.1816678 , 0.20219383, 0.2255169 ,
                             0.        , 0.        , 0.        , 0.        , 0.        ]],
                           [[0.20974289, 0.1808782 , 0.18161033, 0.20220006, 0.22556852,
                             0.        , 0.        , 0.        , 0.        , 0.        ],
                            [0.20973803, 0.18088503, 0.18162282, 0.20220187, 0.22555229,
                             0.        , 0.        , 0.        , 0.        , 0.        ],
                            [0.20972022, 0.1809091 , 0.18161656, 0.20222935, 0.22552474,
                             0.        , 0.        , 0.        , 0.        , 0.        ]]],
                           dtype='float32')
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
                            attentions[expected_best_ids],
                            seq_scores[expected_best_ids],
                            beam_histories)]

    constraints = [sockeye.lexical_constraints.ConstrainedHypothesis(rc, _EOS) for rc in raw_constraints]

    actual_result = sockeye.inference.Translator._get_best_from_beam(translator,
                                                                     best_hyp_indices,
                                                                     best_word_indices,
                                                                     attentions,
                                                                     seq_scores,
                                                                     lengths,
                                                                     constraints,
                                                                     beam_histories)

    for expected_translation, actual_translation in zip(expected_result, actual_result):
        assert expected_translation.target_ids == actual_translation.target_ids
        assert np.array_equal(expected_translation.attention_matrix,
                              actual_translation.attention_matrix)
        assert expected_translation.score == actual_translation.score
        assert expected_translation.beam_histories == actual_translation.beam_histories

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

import pytest
from unittest import mock
from collections import Counter

import sockeye.constants as C
from sockeye.vocab import (build_vocab, get_ordered_tokens_from_vocab, is_valid_vocab, \
    _get_sorted_source_vocab_fnames, build_raw_vocab, merge_raw_vocabs)


def test_build_raw_vocab():
    data = ["a b c", "c d e"]
    raw_vocab = build_raw_vocab(data)
    assert raw_vocab == Counter({"a": 1, "b": 1, "c": 2, "d": 1, "e": 1})


def test_merge_raw_vocabs():
    v1 = build_raw_vocab(["a b c", "c d e"])
    v2 = build_raw_vocab(["a b c", "c d g"])
    raw_vocab = merge_raw_vocabs(v1, v2)
    assert raw_vocab == Counter({"a": 2, "b": 2, "c": 4, "d": 2, "e": 1, "g": 1})


test_vocab = [
        # Example 1
        (["one two three", "one two three"], None, 1,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5, "one": 6}),
        (["one two three", "one two three"], 3, 1,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5, "one": 6}),
        (["one two three", "one two three"], 3, 2,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5, "one": 6}),
        (["one two three", "one two three"], 2, 2,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5}),
        # Example 2
        (["one one two three ", "one two three"], 3, 1,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4, "two": 5, "three": 6}),
        (["one one two three ", "one two three"], 3, 2,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4, "two": 5, "three": 6}),
        (["one one two three ", "one two three"], 3, 3,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4}),
        (["one one two three ", "one two three"], 2, 1,
         {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4, "two": 5}),
        ]


@pytest.mark.parametrize("data,size,min_count,expected", test_vocab)
def test_build_vocab(data, size, min_count, expected):
    vocab = build_vocab(data=data, num_words=size, min_count=min_count)
    assert vocab == expected


@pytest.mark.parametrize("num_types,pad_to_multiple_of,expected_vocab_size",
                         [(4, None, 8), (2, 8, 8), (4, 8, 8), (8, 8, 16), (10, 16, 16), (13, 16, 32)])
def test_padded_build_vocab(num_types, pad_to_multiple_of, expected_vocab_size):
    data = [" ".join('word%d' % i for i in range(num_types))]
    size = None
    min_count = 1
    vocab = build_vocab(data, size, min_count, pad_to_multiple_of=pad_to_multiple_of)
    assert len(vocab) == expected_vocab_size


test_constants = [
        # Example 1
        (["one two three", "one two three"], 3, 1, C.VOCAB_SYMBOLS),
        (["one two three", "one two three"], 3, 2, C.VOCAB_SYMBOLS),
        (["one two three", "one two three"], 2, 2, C.VOCAB_SYMBOLS),
        # Example 2
        (["one one two three ", "one two three"], 3, 1, C.VOCAB_SYMBOLS),
        (["one one two three ", "one two three"], 3, 2, C.VOCAB_SYMBOLS),
        (["one one two three ", "one two three"], 3, 3, C.VOCAB_SYMBOLS),
        (["one one two three ", "one two three"], 2, 1, C.VOCAB_SYMBOLS),
        ]


@pytest.mark.parametrize("data,size,min_count,constants", test_constants)
def test_constants_in_vocab(data, size, min_count, constants):
    vocab = build_vocab(data, size, min_count)
    for const in constants:
        assert const in vocab


@pytest.mark.parametrize("vocab, expected_output", [({"<pad>": 0, "a": 4, "b": 2}, ["<pad>", "b", "a"]),
                                                    ({}, [])])
def test_get_ordered_tokens_from_vocab(vocab, expected_output):
    assert get_ordered_tokens_from_vocab(vocab) == expected_output


@pytest.mark.parametrize(
    "vocab, expected_result",
    [
        ({symbol: idx for idx, symbol in enumerate(C.VOCAB_SYMBOLS + ["w1", "w2"])}, True),
        # A vocabulary with just the valid symbols doesn't make much sense but is technically valid
        ({symbol: idx for idx, symbol in enumerate(C.VOCAB_SYMBOLS)}, True),
        # Manually specifying the list of required special symbol so that we avoid making a backwards-incompatible
        # change by adding a new symbol to C.VOCAB_SYMBOLS
        ({symbol: idx for idx, symbol in enumerate([C.PAD_SYMBOL, C.UNK_SYMBOL, C.BOS_SYMBOL, C.EOS_SYMBOL])}, True),
        # PAD_ID must have word id 0
        ({symbol: idx for idx, symbol in enumerate(reversed(C.VOCAB_SYMBOLS))}, False),
        ({symbol: idx for idx, symbol in enumerate(list(reversed(C.VOCAB_SYMBOLS)) + ["w1", "w2"])}, False),
        # If there is a gap the vocabulary is not valid:
        ({symbol: idx if symbol != "w2" else idx + 1 for idx, symbol in enumerate(C.VOCAB_SYMBOLS + ["w1", "w2"])}, False),
        # There shouldn't be any duplicate word ids
        ({symbol: idx if symbol != "w2" else idx - 1 for idx, symbol in enumerate(C.VOCAB_SYMBOLS + ["w1", "w2"])}, False),
    ]
)
def test_verify_valid_vocab(vocab, expected_result):
    assert is_valid_vocab(vocab) == expected_result


def test_get_sorted_source_vocab_fnames():
    expected_fnames = [C.VOCAB_SRC_NAME % i for i in [1, 2, 10]]
    with mock.patch('os.listdir') as mocked_listdir:
        mocked_listdir.return_value = [C.VOCAB_SRC_NAME % i for i in [2, 1, 10]]
        fnames = _get_sorted_source_vocab_fnames(None)
        assert fnames == expected_fnames

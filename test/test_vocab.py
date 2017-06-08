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

import pytest

import sockeye.constants as C
from sockeye.vocab import build_vocab

test_vocab = [
        # Example 1
        (["one two three", "one two three"], 3, 1, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5, "one": 6}),
        (["one two three", "one two three"], 3, 2, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5, "one": 6}),
        (["one two three", "one two three"], 2, 2, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "two": 4, "three": 5}),
        # Example 2
        (["one one two three ", "one two three"], 3, 1, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4, "two": 5, "three": 6}),
        (["one one two three ", "one two three"], 3, 2, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4, "two": 5, "three": 6}),
        (["one one two three ", "one two three"], 3, 3, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4}),
        (["one one two three ", "one two three"], 2, 1, {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "one": 4, "two": 5}),
        ]


@pytest.mark.parametrize("data,size,min_count,expected", test_vocab)
def test_build_vocab(data, size, min_count, expected):
    vocab = build_vocab(data, size, min_count)
    assert vocab == expected

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

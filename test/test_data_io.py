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
import sockeye.data_io

define_bucket_tests = [(50, 10, [10, 20, 30, 40, 50]),
                       (50, 20, [20, 40, 60]),
                       (50, 50, [50]),
                       (5, 10, [10]),
                       (11, 10, [10, 20]),
                       (19, 10, [10, 20])]


@pytest.mark.parametrize("max_seq_len, step, expected_buckets", define_bucket_tests)
def test_define_buckets(max_seq_len, step, expected_buckets):
    buckets = sockeye.data_io.define_buckets(max_seq_len, step=step)
    assert buckets == expected_buckets


define_parallel_bucket_tests = [(50, 10, 1.0, [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]),
                                (50, 10, 0.5, [(10, 5), (20, 10), (30, 15), (40, 20), (50, 25)]),
                                (50, 10, 0.1, [(10, 2), (20, 2), (30, 3), (40, 4), (50, 5)]),
                                (50, 10, 0.01, [(10, 2), (20, 2), (30, 3), (40, 4), (50, 5)]),
                                (50, 10, 2.0, [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50)]),
                                (50, 10, 10.0, [(2, 10), (2, 20), (3, 30), (4, 40), (5, 50)]),
                                (50, 10, 11.0, [(2, 10), (2, 20), (3, 30), (4, 40), (5, 50)]),
                                (50, 50, 0.5, [(50, 25)]),
                                (50, 50, 1.5, [(33, 50)]),
                                (75, 50, 1.5, [(33, 50), (66, 100)])]


@pytest.mark.parametrize("max_seq_len, bucket_width, length_ratio, expected_buckets", define_parallel_bucket_tests)
def test_define_parallel_buckets(max_seq_len, bucket_width, length_ratio, expected_buckets):
    buckets = sockeye.data_io.define_parallel_buckets(max_seq_len, bucket_width=bucket_width, length_ratio=length_ratio)
    assert buckets == expected_buckets


get_bucket_tests = [([10, 20, 30, 40, 50], 50, 50),
                    ([10, 20, 30, 40, 50], 11, 20),
                    ([10, 20, 30, 40, 50], 9, 10),
                    ([10, 20, 30, 40, 50], 51, None),
                    ([10, 20, 30, 40, 50], 1, 10),
                    ([10, 20, 30, 40, 50], 0, 10),
                    ([], 50, None)]


@pytest.mark.parametrize("buckets, length, expected_bucket",
                         get_bucket_tests)
def test_get_bucket(buckets, length, expected_bucket):
    bucket = sockeye.data_io.get_bucket(length, buckets)
    assert bucket == expected_bucket


get_tokens_tests = [("this is a line  \n", ["this", "is", "a", "line"]),
                    (" a  \tb \r \n", ["a", "b"])]


@pytest.mark.parametrize("line, expected_tokens", get_tokens_tests)
def test_get_tokens(line, expected_tokens):
    tokens = list(sockeye.data_io.get_tokens(line))
    assert tokens == expected_tokens


tokens2ids_tests = [(["a", "b", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 0, 300]),
                    (["a", "x", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 12, 300])]


@pytest.mark.parametrize("tokens, vocab, expected_ids", tokens2ids_tests)
def test_tokens2ids(tokens, vocab, expected_ids):
    ids = sockeye.data_io.tokens2ids(tokens, vocab)
    assert ids == expected_ids


@pytest.mark.parametrize("buckets, expected_default_bucket_key",
                         [([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)], (50, 50)),
                          ([(5, 10), (10, 20), (15, 30), (25, 50), (20, 40)], (25, 50))])
def test_get_default_bucket_key(buckets, expected_default_bucket_key):
    default_bucket_key = sockeye.data_io.get_default_bucket_key(buckets)
    assert default_bucket_key == expected_default_bucket_key


get_parallel_bucket_tests = [([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)], 50, 50, 4, (50, 50)),
                             ([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)], 50, 10, 4, (50, 50)),
                             ([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)], 20, 10, 1, (20, 20)),
                             ([(10, 10)], 20, 10, None, None),
                             ([], 20, 10, None, None),
                             ([(10, 11)], 11, 10, None, None),
                             ([(11, 10)], 11, 10, 0, (11, 10))]


@pytest.mark.parametrize("buckets, source_length, target_length, expected_bucket_index, expected_bucket",
                         get_parallel_bucket_tests)
def test_get_parallel_bucket(buckets, source_length, target_length, expected_bucket_index, expected_bucket):
    bucket_index, bucket = sockeye.data_io.get_parallel_bucket(buckets, source_length, target_length)
    assert bucket_index == expected_bucket_index
    assert bucket == expected_bucket

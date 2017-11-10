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

import numpy as np
import pytest

from sockeye import constants as C
from sockeye import data_io
from sockeye import vocab
from test.common import tmp_digits_dataset

define_bucket_tests = [(50, 10, [10, 20, 30, 40, 50]),
                       (50, 20, [20, 40, 50]),
                       (50, 50, [50]),
                       (5, 10, [5]),
                       (11, 5, [5, 10, 11]),
                       (19, 10, [10, 19])]


@pytest.mark.parametrize("max_seq_len, step, expected_buckets", define_bucket_tests)
def test_define_buckets(max_seq_len, step, expected_buckets):
    buckets = data_io.define_buckets(max_seq_len, step=step)
    assert buckets == expected_buckets


define_parallel_bucket_tests = [(50, 50, 10, 1.0, [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]),
                                (50, 50, 10, 0.5,
                                 [(10, 5), (20, 10), (30, 15), (40, 20), (50, 25), (50, 30), (50, 35), (50, 40),
                                  (50, 45), (50, 50)]),
                                (10, 10, 10, 0.1,
                                 [(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]),
                                (10, 5, 10, 0.01, [(10, 2), (10, 3), (10, 4), (10, 5)]),
                                (50, 50, 10, 2.0,
                                 [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50), (30, 50), (35, 50), (40, 50),
                                  (45, 50), (50, 50)]),
                                (5, 10, 10, 10.0, [(2, 10), (3, 10), (4, 10), (5, 10)]),
                                (5, 10, 10, 11.0, [(2, 10), (3, 10), (4, 10), (5, 10)]),
                                (50, 50, 50, 0.5, [(50, 25), (50, 50)]),
                                (50, 50, 50, 1.5, [(33, 50), (50, 50)]),
                                (75, 75, 50, 1.5, [(33, 50), (66, 75), (75, 75)])]


@pytest.mark.parametrize("max_seq_len_source, max_seq_len_target, bucket_width, length_ratio, expected_buckets",
                         define_parallel_bucket_tests)
def test_define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width, length_ratio, expected_buckets):
    buckets = data_io.define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width=bucket_width,
                                              length_ratio=length_ratio)
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
    bucket = data_io.get_bucket(length, buckets)
    assert bucket == expected_bucket


get_tokens_tests = [("this is a line  \n", ["this", "is", "a", "line"]),
                    (" a  \tb \r \n", ["a", "b"])]


@pytest.mark.parametrize("line, expected_tokens", get_tokens_tests)
def test_get_tokens(line, expected_tokens):
    tokens = list(data_io.get_tokens(line))
    assert tokens == expected_tokens


tokens2ids_tests = [(["a", "b", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 0, 300]),
                    (["a", "x", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 12, 300])]


@pytest.mark.parametrize("tokens, vocab, expected_ids", tokens2ids_tests)
def test_tokens2ids(tokens, vocab, expected_ids):
    ids = data_io.tokens2ids(tokens, vocab)
    assert ids == expected_ids


@pytest.mark.parametrize("buckets, expected_default_bucket_key",
                         [([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)], (50, 50)),
                          ([(5, 10), (10, 20), (15, 30), (25, 50), (20, 40)], (25, 50))])
def test_get_default_bucket_key(buckets, expected_default_bucket_key):
    default_bucket_key = data_io.get_default_bucket_key(buckets)
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
    bucket_index, bucket = data_io.get_parallel_bucket(buckets, source_length, target_length)
    assert bucket_index == expected_bucket_index
    assert bucket == expected_bucket


@pytest.mark.parametrize("source, target, expected_mean, expected_std",
                         [([[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                           [[1, 1, 1], [2, 2, 2], [3, 3, 3]], 1.0, 0.0),
                          ([[1, 1], [2, 2], [3, 3]],
                           [[1, 1, 1], [2, 2, 2], [3, 3, 3]], 1.5, 0.0),
                          ([[1, 1, 1], [2, 2]],
                           [[1, 1, 1], [2], [3, 3, 3]], 0.75, 0.25)])
def test_length_statistics(source, target, expected_mean, expected_std):
    mean, std = data_io.length_statistics(source, target)
    assert np.isclose(mean, expected_mean)
    assert np.isclose(std, expected_std)


def test_get_training_data_iters():
    train_line_count = 100
    train_max_length = 30
    dev_line_count = 20
    dev_max_length = 30
    expected_mean = 1.1476392401276574
    expected_std = 0.2318455878853099
    batch_size = 5
    with tmp_digits_dataset("tmp_corpus",
                            train_line_count, train_max_length, dev_line_count, dev_max_length) as data:
        # tmp common vocab
        vcb = vocab.build_from_paths([data['source'], data['target']])

        train_iter, val_iter, config_data = data_io.get_training_data_iters(data['source'], data['target'],
                                                                            data['validation_source'],
                                                                            data['validation_target'],
                                                                            vocab_source=vcb,
                                                                            vocab_target=vcb,
                                                                            vocab_source_path=None,
                                                                            vocab_target_path=None,
                                                                            batch_size=batch_size,
                                                                            batch_by_words=False,
                                                                            batch_num_devices=1,
                                                                            fill_up="replicate",
                                                                            max_seq_len_source=train_max_length,
                                                                            max_seq_len_target=train_max_length,
                                                                            bucketing=True,
                                                                            bucket_width=10)
        assert config_data.source == data['source']
        assert config_data.target == data['target']
        assert config_data.validation_source == data['validation_source']
        assert config_data.validation_target == data['validation_target']
        assert config_data.vocab_source is None
        assert config_data.vocab_target is None
        assert config_data.max_observed_source_seq_len == train_max_length - 1
        assert config_data.max_observed_target_seq_len == train_max_length
        assert np.isclose(config_data.length_ratio_mean, expected_mean)
        assert np.isclose(config_data.length_ratio_std, expected_std)

        assert train_iter.batch_size == batch_size
        assert val_iter.batch_size == batch_size
        assert train_iter.default_bucket_key == (train_max_length, train_max_length)
        assert val_iter.default_bucket_key == (dev_max_length, dev_max_length)
        assert train_iter.max_observed_source_len == config_data.max_observed_source_seq_len
        assert train_iter.max_observed_target_len == config_data.max_observed_target_seq_len
        assert train_iter.pad_id == vcb[C.PAD_SYMBOL]
        assert train_iter.dtype == 'float32'
        assert not train_iter.batch_by_words
        assert train_iter.fill_up == 'replicate'

        # test some batches
        bos_id = vcb[C.BOS_SYMBOL]
        expected_first_target_symbols = np.full((batch_size,), bos_id, dtype='float32')
        for epoch in range(2):
            while train_iter.iter_next():
                batch = train_iter.next()
                assert len(batch.data) == 2
                assert len(batch.label) == 1
                assert batch.bucket_key in train_iter.buckets
                source = batch.data[0].asnumpy()
                target = batch.data[1].asnumpy()
                label = batch.label[0].asnumpy()
                assert source.shape[0] == batch_size
                assert target.shape[0] == batch_size
                assert label.shape[0] == batch_size
                # target first symbol should be BOS
                assert np.array_equal(target[:, 0], expected_first_target_symbols)
                # label first symbol should be 2nd target symbol
                assert np.array_equal(label[:, 0], target[:, 1])
                # each label sequence contains one EOS symbol
                assert np.sum(label == vcb[C.EOS_SYMBOL]) == batch_size
            train_iter.reset()

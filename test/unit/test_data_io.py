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

import os
import random
from tempfile import TemporaryDirectory
from typing import Optional, List, Tuple

import mxnet as mx
import numpy as np
import pytest

from sockeye import constants as C
from sockeye import data_io
from sockeye import vocab
from sockeye.utils import SockeyeError, get_tokens, seedRNGs
from test.common import tmp_digits_dataset

seedRNGs(12)

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


tokens2ids_tests = [(["a", "b", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 0, 300]),
                    (["a", "x", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 12, 300])]


@pytest.mark.parametrize("tokens, vocab, expected_ids", tokens2ids_tests)
def test_tokens2ids(tokens, vocab, expected_ids):
    ids = data_io.tokens2ids(tokens, vocab)
    assert ids == expected_ids


@pytest.mark.parametrize("tokens, expected_ids", [(["1", "2", "3", "0"], [1, 2, 3, 0]), ([], [])])
def test_strids2ids(tokens, expected_ids):
    ids = data_io.strids2ids(tokens)
    assert ids == expected_ids


@pytest.mark.parametrize("ids, expected_string", [([1, 2, 3, 0], "1 2 3 0"), ([], "")])
def test_ids2strids(ids, expected_string):
    string = data_io.ids2strids(ids)
    assert string == expected_string


sequence_reader_tests = [(["1 2 3", "2", "", "2 2 2"], False, False, False),
                         (["a b c", "c"], True, False, False),
                         (["a b c", ""], True, False, False),
                         (["a b c", "c"], True, True, True)]


@pytest.mark.parametrize("sequences, use_vocab, add_bos, add_eos", sequence_reader_tests)
def test_sequence_reader(sequences, use_vocab, add_bos, add_eos):
    with TemporaryDirectory() as work_dir:
        path = os.path.join(work_dir, 'input')
        with open(path, 'w') as f:
            for sequence in sequences:
                print(sequence, file=f)

        vocabulary = vocab.build_vocab(sequences) if use_vocab else None

        reader = data_io.SequenceReader(path, vocabulary=vocabulary, add_bos=add_bos, add_eos=add_eos)

        read_sequences = [s for s in reader]
        assert len(read_sequences) == len(sequences)

        if vocabulary is None:
            with pytest.raises(SockeyeError) as e:
                _ = data_io.SequenceReader(path, vocabulary=vocabulary, add_bos=True)
            assert str(e.value) == "Adding a BOS or EOS symbol requires a vocabulary"

            expected_sequences = [data_io.strids2ids(get_tokens(s)) if s else None for s in sequences]
            assert read_sequences == expected_sequences
        else:
            expected_sequences = [data_io.tokens2ids(get_tokens(s), vocabulary) if s else None for s in sequences]
            if add_bos:
                expected_sequences = [[vocabulary[C.BOS_SYMBOL]] + s if s else None for s in expected_sequences]
            if add_eos:
                expected_sequences = [s + [vocabulary[C.EOS_SYMBOL]]  if s else None for s in expected_sequences]
            assert read_sequences == expected_sequences


@pytest.mark.parametrize("source_iterables, target_iterable",
                         [
                             (
                                     [[[0], [1, 1], [2], [3, 3, 3]], [[0], [1, 1], [2], [3, 3, 3]]],
                                     [[0], [1]]
                             ),
                             (
                                     [[[0], [1, 1]], [[0], [1, 1]]],
                                     [[0], [1, 1], [2], [3, 3, 3]]
                             ),
                             (
                                     [[[0], [1, 1]]],
                                     [[0], [1, 1], [2], [3, 3, 3]]
                             ),
                         ])
def test_nonparallel_iter(source_iterables, target_iterable):
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterable))
    assert str(e.value) == "Different number of lines in source(s) and target iterables."


@pytest.mark.parametrize("source_iterables, target_iterable",
                         [
                             (
                                     [[[0], [1, 1]], [[0], [1]]],
                                     [[0], [1]]
                             )
                         ])
def test_nontoken_parallel_iter(source_iterables, target_iterable):
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterable))
    assert str(e.value).startswith("Source sequences are not token-parallel")


@pytest.mark.parametrize("source_iterables, target_iterable, expected",
                         [
                             (
                                     [[[0], [1, 1]], [[0], [1, 1]]],
                                     [[0], [1]],
                                     [(([0], [0]), [0]), (([1, 1], [1, 1]), [1])]
                             ),
                             (
                                     [[[0], None], [[0], None]],
                                     [[0], [1]],
                                     [(([0], [0]), [0])]
                             ),
                             (
                                     [[[0], [1, 1]], [[0], [1, 1]]],
                                     [[0], None],
                                     [(([0], [0]), [0])]
                             ),
                             (
                                     [[None, [1, 1]], [None, [1, 1]]],
                                     [None, [1]],
                                     [(([1, 1], [1, 1]), [1])]
                             ),
                             (
                                     [[None, [1, 1]], [None, [1, 1]]],
                                     [None, None],
                                     []
                             )
                         ])
def test_parallel_iter(source_iterables, target_iterable, expected):
    assert list(data_io.parallel_iter(source_iterables, target_iterable)) == expected


def test_sample_based_define_bucket_batch_sizes():
    batch_by_words = False
    batch_size = 32
    max_seq_len = 100
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, 10, 1.5)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets=buckets,
                                                           batch_size=batch_size,
                                                           batch_by_words=batch_by_words,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))
    for bbs in bucket_batch_sizes:
        assert bbs.batch_size == batch_size
        assert bbs.average_words_per_batch == bbs.bucket[1] * batch_size


def test_word_based_define_bucket_batch_sizes():
    batch_by_words = True
    batch_num_devices = 1
    batch_size = 200
    max_seq_len = 100
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, 10, 1.5)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets=buckets,
                                                           batch_size=batch_size,
                                                           batch_by_words=batch_by_words,
                                                           batch_num_devices=batch_num_devices,
                                                           data_target_average_len=[None] * len(buckets))
    # last bucket batch size is different
    for bbs in bucket_batch_sizes[:-1]:
        expected_batch_size = round((batch_size / bbs.bucket[1]) / batch_num_devices)
        assert bbs.batch_size == expected_batch_size
        expected_average_words_per_batch = expected_batch_size * bbs.bucket[1]
        assert bbs.average_words_per_batch == expected_average_words_per_batch


def _get_random_bucketed_data(buckets: List[Tuple[int, int]],
                              min_count: int,
                              max_count: int,
                              bucket_counts: Optional[List[Optional[int]]] = None):
    """
    Get random bucket data.

    :param buckets: The list of buckets.
    :param min_count: The minimum number of samples that will be sampled if no exact count is given.
    :param max_count: The maximum number of samples that will be sampled if no exact count is given.
    :param bucket_counts: For each bucket an optional exact example count can be given. If it is not given it will be
                         sampled.
    :return: The random source, target and label arrays.
    """
    if bucket_counts is None:
        bucket_counts = [None for _ in buckets]
    bucket_counts = [random.randint(min_count, max_count) if given_count is None else given_count
                     for given_count in bucket_counts]
    source = [mx.nd.array(np.random.randint(0, 10, (count, random.randint(1, bucket[0])))) for count, bucket in
              zip(bucket_counts, buckets)]
    target = [mx.nd.array(np.random.randint(0, 10, (count, random.randint(1, bucket[1])))) for count, bucket in
              zip(bucket_counts, buckets)]
    label = target
    return source, target, label


def test_parallel_data_set():
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    source, target, label = _get_random_bucketed_data(buckets, min_count=0, max_count=5)

    def check_equal(arrays1, arrays2):
        assert len(arrays1) == len(arrays2)
        for a1, a2 in zip(arrays1, arrays2):
            assert np.array_equal(a1.asnumpy(), a2.asnumpy())

    with TemporaryDirectory() as work_dir:
        dataset = data_io.ParallelDataSet(source, target, label)
        fname = os.path.join(work_dir, 'dataset')
        dataset.save(fname)
        dataset_loaded = data_io.ParallelDataSet.load(fname)
        check_equal(dataset.source, dataset_loaded.source)
        check_equal(dataset.target, dataset_loaded.target)
        check_equal(dataset.label, dataset_loaded.label)


def test_parallel_data_set_fill_up():
    batch_size = 32
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))
    dataset = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=1, max_count=5))

    dataset_filled_up = dataset.fill_up(bucket_batch_sizes, 'replicate')
    assert len(dataset_filled_up.source) == len(dataset.source)
    assert len(dataset_filled_up.target) == len(dataset.target)
    assert len(dataset_filled_up.label) == len(dataset.label)
    for bidx in range(len(dataset)):
        bucket_batch_size = bucket_batch_sizes[bidx].batch_size
        assert dataset_filled_up.source[bidx].shape[0] == bucket_batch_size
        assert dataset_filled_up.target[bidx].shape[0] == bucket_batch_size
        assert dataset_filled_up.label[bidx].shape[0] == bucket_batch_size


def test_get_permutations():
    data = [list(range(3)), list(range(1)), list(range(7)), []]
    bucket_counts = [len(d) for d in data]

    permutation, inverse_permutation = data_io.get_permutations(bucket_counts)
    assert len(permutation) == len(inverse_permutation) == len(bucket_counts) == len(data)

    for d, p, pi in zip(data, permutation, inverse_permutation):
        p = p.asnumpy().astype(np.int)
        pi = pi.asnumpy().astype(np.int)
        p_set = set(p)
        pi_set = set(pi)
        assert len(p_set) == len(p)
        assert len(pi_set) == len(pi)
        assert p_set - pi_set == set()
        if d:
            d = np.array(d)
            assert (d[p][pi] == d).all()
        else:
            assert len(p_set) == 1


def test_parallel_data_set_permute():
    batch_size = 5
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))
    dataset = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5)).fill_up(
        bucket_batch_sizes, 'replicate')

    permutations, inverse_permutations = data_io.get_permutations(dataset.get_bucket_counts())

    assert len(permutations) == len(inverse_permutations) == len(dataset)
    dataset_restored = dataset.permute(permutations).permute(inverse_permutations)
    assert len(dataset) == len(dataset_restored)
    for buck_idx in range(len(dataset)):
        num_samples = dataset.source[buck_idx].shape[0]
        if num_samples:
            assert (dataset.source[buck_idx] == dataset_restored.source[buck_idx]).asnumpy().all()
            assert (dataset.target[buck_idx] == dataset_restored.target[buck_idx]).asnumpy().all()
            assert (dataset.label[buck_idx] == dataset_restored.label[buck_idx]).asnumpy().all()
        else:
            assert not dataset_restored.source[buck_idx]
            assert not dataset_restored.target[buck_idx]
            assert not dataset_restored.label[buck_idx]


def test_get_batch_indices():
    max_bucket_size = 50
    batch_size = 10
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))
    dataset = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets=buckets,
                                                                 min_count=1,
                                                                 max_count=max_bucket_size))

    indices = data_io.get_batch_indices(dataset, bucket_batch_sizes=bucket_batch_sizes)

    # check for valid indices
    for buck_idx, start_pos in indices:
        assert 0 <= buck_idx < len(dataset)
        assert 0 <= start_pos < len(dataset.source[buck_idx]) - batch_size + 1

    # check that all indices are used for a filled-up dataset
    dataset = dataset.fill_up(bucket_batch_sizes, fill_up='replicate')
    indices = data_io.get_batch_indices(dataset, bucket_batch_sizes=bucket_batch_sizes)
    all_bucket_indices = set(list(range(len(dataset))))
    computed_bucket_indices = set([i for i, j in indices])

    assert not all_bucket_indices - computed_bucket_indices


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


@pytest.mark.parametrize("sources, target, expected_num_sents, expected_mean, expected_std",
                         [([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]],
                           [[1, 1, 1], [2, 2, 2], [3, 3, 3]], 3, 1.0, 0.0),
                          ([[[1, 1], [2, 2], [3, 3]]],
                           [[1, 1, 1], [2, 2, 2], [3, 3, 3]], 3, 1.5, 0.0),
                          ([[[1, 1, 1], [2, 2], [3, 3, 3, 3, 3, 3, 3]]],
                           [[1, 1, 1], [2], [3, 3, 3]], 2, 0.75, 0.25)])
def test_calculate_length_statistics(sources, target, expected_num_sents, expected_mean, expected_std):
    length_statistics = data_io.calculate_length_statistics(sources, target, 5, 5)
    assert len(sources[0]) == len(target)
    assert length_statistics.num_sents == expected_num_sents
    assert np.isclose(length_statistics.length_ratio_mean, expected_mean)
    assert np.isclose(length_statistics.length_ratio_std, expected_std)


@pytest.mark.parametrize("sources, target",
                         [
                             ([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                               [[1, 1, 1], [2, 2], [3, 3, 3]]],
                              [[1, 1, 1], [2, 2, 2], [3, 3, 3]])
                         ])
def test_non_parallel_calculate_length_statistics(sources, target):
    with pytest.raises(SockeyeError):
        data_io.calculate_length_statistics(sources, target, 5, 5)


def test_get_training_data_iters():
    train_line_count = 100
    train_max_length = 30
    dev_line_count = 20
    dev_max_length = 30
    expected_mean = 1.0
    expected_std = 0.0
    test_line_count = 20
    test_line_count_empty = 0
    test_max_length = 30
    batch_size = 5
    with tmp_digits_dataset("tmp_corpus",
                            train_line_count, train_max_length - C.SPACE_FOR_XOS,
                            dev_line_count, dev_max_length - C.SPACE_FOR_XOS,
                            test_line_count, test_line_count_empty,
                            test_max_length - C.SPACE_FOR_XOS) as data:
        # tmp common vocab
        vcb = vocab.build_from_paths([data['source'], data['target']])

        train_iter, val_iter, config_data, data_info = data_io.get_training_data_iters(
            sources=[data['source']],
            target=data['target'],
            validation_sources=[
                data['validation_source']],
            validation_target=data[
                'validation_target'],
            source_vocabs=[vcb],
            target_vocab=vcb,
            source_vocab_paths=[None],
            target_vocab_path=None,
            shared_vocab=True,
            batch_size=batch_size,
            batch_by_words=False,
            batch_num_devices=1,
            fill_up="replicate",
            max_seq_len_source=train_max_length,
            max_seq_len_target=train_max_length,
            bucketing=True,
            bucket_width=10)
        assert isinstance(train_iter, data_io.ParallelSampleIter)
        assert isinstance(val_iter, data_io.ParallelSampleIter)
        assert isinstance(config_data, data_io.DataConfig)
        assert data_info.sources == [data['source']]
        assert data_info.target == data['target']
        assert data_info.source_vocabs == [None]
        assert data_info.target_vocab is None
        assert config_data.data_statistics.max_observed_len_source == train_max_length
        assert config_data.data_statistics.max_observed_len_target == train_max_length
        assert np.isclose(config_data.data_statistics.length_ratio_mean, expected_mean)
        assert np.isclose(config_data.data_statistics.length_ratio_std, expected_std)

        assert train_iter.batch_size == batch_size
        assert val_iter.batch_size == batch_size
        assert train_iter.default_bucket_key == (train_max_length, train_max_length)
        assert val_iter.default_bucket_key == (dev_max_length, dev_max_length)
        assert train_iter.dtype == 'float32'

        # test some batches
        bos_id = vcb[C.BOS_SYMBOL]
        eos_id = vcb[C.EOS_SYMBOL]
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
                assert source.shape[0] == target.shape[0] == label.shape[0] == batch_size
                # target first symbol should be BOS
                # each source sequence contains one EOS symbol
                assert np.sum(source == eos_id) == batch_size
                assert np.array_equal(target[:, 0], expected_first_target_symbols)
                # label first symbol should be 2nd target symbol
                assert np.array_equal(label[:, 0], target[:, 1])
                # each label sequence contains one EOS symbol
                assert np.sum(label == eos_id) == batch_size
            train_iter.reset()


def _data_batches_equal(db1, db2):
    # We just compare the data, should probably be enough
    equal = True
    for data1, data2 in zip(db1.data, db2.data):
        equal = equal and np.allclose(data1.asnumpy(), data2.asnumpy())
    return equal


def test_parallel_sample_iter():
    batch_size = 2
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    # The first bucket is going to be empty:
    bucket_counts = [0] + [None] * (len(buckets) - 1)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))

    dataset = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5,
                                                                 bucket_counts=bucket_counts))
    it = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)

    with TemporaryDirectory() as work_dir:
        # Test 1
        it.next()
        expected_batch = it.next()

        fname = os.path.join(work_dir, "saved_iter")
        it.save_state(fname)

        it_loaded = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)

        # Test 2
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)

        it_loaded = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
        it_loaded.reset()
        it_loaded.load_state(fname)

        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)

        # Test 3
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)
        it_loaded = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
        it_loaded.reset()
        it_loaded.load_state(fname)

        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)

        while it.iter_next():
            it.next()
            it_loaded.next()
        assert not it_loaded.iter_next()


def test_sharded_parallel_sample_iter():
    batch_size = 2
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    # The first bucket is going to be empty:
    bucket_counts = [0] + [None] * (len(buckets) - 1)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))

    dataset1 = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5,
                                                                  bucket_counts=bucket_counts))
    dataset2 = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5,
                                                                  bucket_counts=bucket_counts))

    with TemporaryDirectory() as work_dir:
        shard1_fname = os.path.join(work_dir, 'shard1')
        shard2_fname = os.path.join(work_dir, 'shard2')
        dataset1.save(shard1_fname)
        dataset2.save(shard2_fname)
        shard_fnames = [shard1_fname, shard2_fname]

        it = data_io.ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes, 'replicate')

        # Test 1
        it.next()
        expected_batch = it.next()

        fname = os.path.join(work_dir, "saved_iter")
        it.save_state(fname)

        it_loaded = data_io.ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes,
                                                      'replicate')
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)

        # Test 2
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)

        it_loaded = data_io.ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes,
                                                      'replicate')
        it_loaded.reset()
        it_loaded.load_state(fname)

        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)

        # Test 3
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)
        it_loaded = data_io.ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes,
                                                      'replicate')
        it_loaded.reset()
        it_loaded.load_state(fname)

        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)

        while it.iter_next():
            it.next()
            it_loaded.next()
        assert not it_loaded.iter_next()


def test_sharded_parallel_sample_iter_num_batches():
    num_shards = 2
    batch_size = 2
    num_batches_per_bucket = 10
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    bucket_counts = [batch_size * num_batches_per_bucket for _ in buckets]
    num_batches_per_shard = num_batches_per_bucket * len(buckets)
    num_batches = num_shards * num_batches_per_shard
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))

    dataset1 = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5,
                                                                  bucket_counts=bucket_counts))
    dataset2 = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5,
                                                                  bucket_counts=bucket_counts))
    with TemporaryDirectory() as work_dir:
        shard1_fname = os.path.join(work_dir, 'shard1')
        shard2_fname = os.path.join(work_dir, 'shard2')
        dataset1.save(shard1_fname)
        dataset2.save(shard2_fname)
        shard_fnames = [shard1_fname, shard2_fname]

        it = data_io.ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes,
                                               'replicate')

        num_batches_seen = 0
        while it.iter_next():
            it.next()
            num_batches_seen += 1
        assert num_batches_seen == num_batches


def test_sharded_and_parallel_iter_same_num_batches():
    """ Tests that a sharded data iterator with just a single shard produces as many shards as an iterator directly
    using the same dataset. """
    batch_size = 2
    num_batches_per_bucket = 10
    buckets = data_io.define_parallel_buckets(100, 100, 10, 1.0)
    bucket_counts = [batch_size * num_batches_per_bucket for _ in buckets]
    num_batches = num_batches_per_bucket * len(buckets)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                           batch_size,
                                                           batch_by_words=False,
                                                           batch_num_devices=1,
                                                           data_target_average_len=[None] * len(buckets))

    dataset = data_io.ParallelDataSet(*_get_random_bucketed_data(buckets, min_count=0, max_count=5,
                                                                 bucket_counts=bucket_counts))

    with TemporaryDirectory() as work_dir:
        shard_fname = os.path.join(work_dir, 'shard1')
        dataset.save(shard_fname)
        shard_fnames = [shard_fname]

        it_sharded = data_io.ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes,
                                                       'replicate')

        it_parallel = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)

        num_batches_seen = 0
        while it_parallel.iter_next():
            assert it_sharded.iter_next()
            it_parallel.next()
            it_sharded.next()
            num_batches_seen += 1
        assert num_batches_seen == num_batches

        print("Resetting...")
        it_sharded.reset()
        it_parallel.reset()

        num_batches_seen = 0
        while it_parallel.iter_next():
            assert it_sharded.iter_next()
            it_parallel.next()
            it_sharded.next()

            num_batches_seen += 1

        assert num_batches_seen == num_batches

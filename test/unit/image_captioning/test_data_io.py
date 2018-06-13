# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from tempfile import TemporaryDirectory

import mxnet as mx
import numpy as np
import pytest

import sockeye.constants as C
import sockeye.data_io
import sockeye.image_captioning.data_io as data_io
from sockeye import vocab
from sockeye.utils import seedRNGs
from test.common_image_captioning import generate_img_or_feat, tmp_img_captioning_dataset, _FEATURE_SHAPE, _CNN_INPUT_IMAGE_SHAPE

seedRNGs(12)


@pytest.mark.parametrize("source_list, target_sentences, num_samples_per_bucket, expected_source_0, expected_target_0, expected_label_0",
                         [(['1', '2', '3', '4', '100'],
                          [[1, 2, 3], [1, 6, 7], [7, 3], [3, 4, 5, 6], [3, 4]],
                          [2, 2, 1],
                           ['3', '100'], [[ 7., 3.], [ 3., 4.]], [[3., 10.], [4., 10.]])])
def test_raw_list_text_dset_loader(source_list, target_sentences, num_samples_per_bucket,
                                   expected_source_0, expected_target_0, expected_label_0):
    # Test Init object
    buckets = sockeye.data_io.define_parallel_buckets(4, 4, 1, 1.0)
    dset_loader = data_io.RawListTextDatasetLoader(buckets=buckets,
                                       eos_id=10, pad_id=C.PAD_ID)

    assert isinstance(dset_loader, data_io.RawListTextDatasetLoader)
    assert len(dset_loader.buckets)==3

    # Test Load data
    pop_dset_loader = dset_loader.load(source_list, target_sentences, num_samples_per_bucket)

    assert isinstance(pop_dset_loader, sockeye.data_io.ParallelDataSet)
    assert len(pop_dset_loader.source)==3
    assert len(pop_dset_loader.target)==3
    assert len(pop_dset_loader.label)==3
    np.testing.assert_equal(pop_dset_loader.source[0], expected_source_0)
    np.testing.assert_almost_equal(pop_dset_loader.target[0].asnumpy(), expected_target_0)
    np.testing.assert_almost_equal(pop_dset_loader.label[0].asnumpy(), expected_label_0)


@pytest.mark.parametrize("source_list, target_sentences, num_samples_per_bucket",
                         [(['a', 'b', 'c', 'd', 'e'],
                          [[1, 2, 3], [1, 6, 7], [7, 3], [3, 4, 5, 6], [3, 4]],
                           [2, 2, 1])])
def test_image_text_sample_iter(source_list, target_sentences, num_samples_per_bucket):
    batch_size = 2
    image_size = _CNN_INPUT_IMAGE_SHAPE
    buckets = sockeye.data_io.define_parallel_buckets(4, 4, 1, 1.0)
    bucket_batch_sizes = sockeye.data_io.define_bucket_batch_sizes(buckets,
                                                                   batch_size,
                                                                   batch_by_words=False,
                                                                   batch_num_devices=1,
                                                                   data_target_average_len=[None]*len(buckets))
    dset_loader = data_io.RawListTextDatasetLoader(buckets=buckets, eos_id=-1, pad_id=C.PAD_ID)
    with TemporaryDirectory() as work_dir:
        source_list_img = []
        source_list_npy = []
        for s in source_list:
            source_list_img.append(os.path.join(work_dir, s + ".jpg"))
            source_list_npy.append(os.path.join(work_dir, s + ".npy"))
        # Create random images/features
        for s in source_list_img:
            filename = os.path.join(work_dir, s)
            generate_img_or_feat(filename, use_features=False)
        for s in source_list_npy:
            filename = os.path.join(work_dir, s)
            generate_img_or_feat(filename, use_features=True)

        # Test image iterator
        pop_dset_loader = dset_loader.load(source_list_img, target_sentences, num_samples_per_bucket)
        data_iter = data_io.ImageTextSampleIter(pop_dset_loader,
                                                buckets,
                                                batch_size,
                                                bucket_batch_sizes,
                                                image_size,
                                                use_feature_loader=False,
                                                preload_features=False)
        data = data_iter.next()
        assert isinstance(data, mx.io.DataBatch)
        np.testing.assert_equal(data.data[0].asnumpy().shape[1:], image_size)

        # Test iterator feature loader + preload all to memory
        pop_dset_loader = dset_loader.load(source_list_npy, target_sentences, num_samples_per_bucket)
        data_iter = data_io.ImageTextSampleIter(pop_dset_loader,
                                                buckets,
                                                batch_size,
                                                bucket_batch_sizes,
                                                _FEATURE_SHAPE,
                                                use_feature_loader=True,
                                                preload_features=True)
        data = data_iter.next()
        assert isinstance(data, mx.io.DataBatch)
        np.testing.assert_equal(data.data[0].asnumpy().shape[1:], _FEATURE_SHAPE)


def test_get_training_feature_text_data_iters():
    # Test features
    source_list = ['1', '2', '3', '4', '100']
    prefix = "tmp_corpus"
    use_feature_loader = True
    preload_features = True
    train_max_length = 30
    dev_max_length = 30
    expected_mean = 1.0
    expected_std = 1.0
    test_max_length = 30
    batch_size = 5
    if use_feature_loader:
        source_image_size = _FEATURE_SHAPE
    else:
        source_image_size = _CNN_INPUT_IMAGE_SHAPE
    with tmp_img_captioning_dataset(source_list,
                                    prefix,
                                    train_max_length,
                                    dev_max_length,
                                    test_max_length,
                                    use_feature_loader) as data:
        # tmp common vocab
        vcb = vocab.build_from_paths([data['target'], data['target']])

        train_iter, val_iter, config_data, data_info = data_io.get_training_image_text_data_iters(source_root=data['work_dir'],
                                                                                                  source=data['source'],
                                                                                                  target=data['target'],
                                                                                                  validation_source_root=data['work_dir'],
                                                                                                  validation_source=data['validation_source'],
                                                                                                  validation_target=data['validation_target'],
                                                                                                  vocab_target=vcb,
                                                                                                  vocab_target_path=None,
                                                                                                  batch_size=batch_size,
                                                                                                  batch_by_words=False,
                                                                                                  batch_num_devices=1,
                                                                                                  source_image_size=source_image_size,
                                                                                                  fill_up="replicate",
                                                                                                  max_seq_len_target=train_max_length,
                                                                                                  bucketing=True,
                                                                                                  bucket_width=10,
                                                                                                  use_feature_loader=use_feature_loader,
                                                                                                  preload_features=preload_features)
        assert isinstance(train_iter, data_io.ParallelSampleIter)
        assert isinstance(val_iter, data_io.ParallelSampleIter)
        assert isinstance(config_data, data_io.DataConfig)
        assert isinstance(data_info.sources[0], data_io.FileListReader)
        assert data_info.target == data['target']
        assert data_info.source_vocabs is None
        assert data_info.target_vocab is None
        assert config_data.data_statistics.max_observed_len_source == 0
        assert config_data.data_statistics.max_observed_len_target == train_max_length - 1
        assert np.isclose(config_data.data_statistics.length_ratio_mean, expected_mean)
        assert np.isclose(config_data.data_statistics.length_ratio_std, expected_std)

        assert train_iter.batch_size == batch_size
        assert val_iter.batch_size == batch_size
        assert train_iter.default_bucket_key == (0, train_max_length)
        assert val_iter.default_bucket_key == (0, dev_max_length)
        assert train_iter.dtype == 'float32'

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
                assert source.shape[0] == target.shape[0] == label.shape[0] == batch_size
                # target first symbol should be BOS
                assert np.array_equal(target[:, 0], expected_first_target_symbols)
                # label first symbol should be 2nd target symbol
                assert np.array_equal(label[:, 0], target[:, 1])
                # each label sequence contains one EOS symbol
                assert np.sum(label == vcb[C.EOS_SYMBOL]) == batch_size
            train_iter.reset()


def test_get_training_image_text_data_iters():
    # Test images
    source_list = ['1', '2', '3', '4', '100']
    prefix = "tmp_corpus"
    use_feature_loader = False
    preload_features = False
    train_max_length = 30
    dev_max_length = 30
    expected_mean = 1.0
    expected_std = 1.0
    test_max_length = 30
    batch_size = 5
    if use_feature_loader:
        source_image_size = _FEATURE_SHAPE
    else:
        source_image_size = _CNN_INPUT_IMAGE_SHAPE
    with tmp_img_captioning_dataset(source_list,
                                    prefix,
                                    train_max_length,
                                    dev_max_length,
                                    test_max_length,
                                    use_feature_loader) as data:
        # tmp common vocab
        vcb = vocab.build_from_paths([data['target'], data['target']])

        train_iter, val_iter, config_data, data_info = data_io.get_training_image_text_data_iters(source_root=data['work_dir'],
                                                                                                  source=data['source'],
                                                                                                  target=data['target'],
                                                                                                  validation_source_root=data['work_dir'],
                                                                                                  validation_source=data['validation_source'],
                                                                                                  validation_target=data['validation_target'],
                                                                                                  vocab_target=vcb,
                                                                                                  vocab_target_path=None,
                                                                                                  batch_size=batch_size,
                                                                                                  batch_by_words=False,
                                                                                                  batch_num_devices=1,
                                                                                                  source_image_size=source_image_size,
                                                                                                  fill_up="replicate",
                                                                                                  max_seq_len_target=train_max_length,
                                                                                                  bucketing=False,
                                                                                                  bucket_width=10,
                                                                                                  use_feature_loader=use_feature_loader,
                                                                                                  preload_features=preload_features)
        assert isinstance(train_iter, data_io.ParallelSampleIter)
        assert isinstance(val_iter, data_io.ParallelSampleIter)
        assert isinstance(config_data, data_io.DataConfig)
        assert isinstance(data_info.sources[0], data_io.FileListReader)
        assert data_info.target == data['target']
        assert data_info.source_vocabs is None
        assert data_info.target_vocab is None
        assert config_data.data_statistics.max_observed_len_source == 0
        assert config_data.data_statistics.max_observed_len_target == train_max_length - 1
        assert np.isclose(config_data.data_statistics.length_ratio_mean, expected_mean)
        assert np.isclose(config_data.data_statistics.length_ratio_std, expected_std)

        assert train_iter.batch_size == batch_size
        assert val_iter.batch_size == batch_size
        assert train_iter.default_bucket_key == (0, train_max_length)
        assert val_iter.default_bucket_key == (0, dev_max_length)
        assert train_iter.dtype == 'float32'

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
                assert source.shape[0] == target.shape[0] == label.shape[0] == batch_size
                # target first symbol should be BOS
                assert np.array_equal(target[:, 0], expected_first_target_symbols)
                # label first symbol should be 2nd target symbol
                assert np.array_equal(label[:, 0], target[:, 1])
                # each label sequence contains one EOS symbol
                assert np.sum(label == vcb[C.EOS_SYMBOL]) == batch_size
            train_iter.reset()
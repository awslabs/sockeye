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

"""
Implements data iterators and I/O related functions for image-to-sequence
models.
"""
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mxnet as mx
import numpy as np

from .utils import load_features, load_feature, load_preprocess_images
from .. import constants as C
from .. import vocab
from ..data_io import ParallelDataSet, RawParallelDatasetLoader, \
    BucketBatchSize, FileListReader, SequenceReader, DataConfig, DataInfo, \
    ParallelSampleIter
from ..data_io import get_target_bucket, get_data_statistics, \
    define_empty_source_parallel_buckets, define_bucket_batch_sizes

logger = logging.getLogger(__name__)


class RawListTextDatasetLoader:
    """
    Loads a data set of variable-length parallel list of string and target sequences into buckets of NDArrays.
    The list of strings are not converted to NDArrays, because we assume that the dataset does not fit in memory.
    We assume that the used data iterator knows how to load the data from disk to memory every time a batch is consumed.
    Note: it does not support multiple source, like `sockeye.data_io.RawParallelDatasetLoader`.

    :param buckets: Bucket list.
    :param eos_id: End-of-sentence id.
    :param pad_id: Padding id.
    :param eos_id: Unknown id.
    :param dtype: Data type.
    """

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 eos_id: int,
                 pad_id: int,
                 dtype: str = 'float32') -> None:
        self.buckets = buckets
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.dtype = dtype

    def load(self,
             source_list: Iterable[List[str]],
             target_sentences: Iterable[List[Any]],
             num_samples_per_bucket: List[int]) -> 'ParallelDataSet':
        """
        Creates a parallel dataset base on source list of strings and target sentences.
        Returns a `sockeye.data_io.ParallelDataSet`.

        :param source_list: Source list of strings (e.g., filenames).
        :param target_sentences: Target sentences used to do bucketing.
        :param num_samples_per_bucket: Number of samples per bucket.
        :return: Returns a parallel dataset `sockeye.data_io.ParallelDataSet`.
        """
        assert len(num_samples_per_bucket) == len(self.buckets)

        data_source = [np.full((num_samples, ), self.pad_id, dtype=object)
                       for num_samples in num_samples_per_bucket]
        # data_source is a List[numpy.array[str]] which semantic is bucket, index, str
        # Its loading to memory is deferred to the iterator, since the full data
        # is supposed to not fit in memory.
        data_target = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_label = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                      for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]

        bucket_sample_index = [0 for buck in self.buckets]

        # track amount of padding introduced through bucketing
        num_tokens_target = 0
        num_pad_target = 0

        # Bucket sentences as padded np arrays
        for source, target in zip(source_list, target_sentences):
            target_len = len(target)
            buck_index, buck = get_target_bucket(self.buckets, target_len)
            if buck is None:
                continue  # skip this sentence pair

            num_tokens_target += buck[1]
            num_pad_target += buck[1] - target_len

            sample_index = bucket_sample_index[buck_index]
            data_source[buck_index][sample_index] = source
            data_target[buck_index][sample_index, :target_len] = target
            # NOTE(fhieber): while this is wasteful w.r.t memory, we need to explicitly create the label sequence
            # with the EOS symbol here sentence-wise and not per-batch due to variable sequence length within a batch.
            # Once MXNet allows item assignments given a list of indices (probably MXNet 1.0): e.g a[[0,1,5,2]] = x,
            # we can try again to compute the label sequence on the fly in next().
            data_label[buck_index][sample_index, :target_len] = target[1:] + [self.eos_id]

            bucket_sample_index[buck_index] += 1

        for i in range(len(data_source)):
            data_target[i] = mx.nd.array(data_target[i], dtype=self.dtype)
            data_label[i] = mx.nd.array(data_label[i], dtype=self.dtype)

        if num_tokens_target > 0:
            logger.info("Created bucketed parallel data set. Introduced padding: target=%.1f%%)",
                        num_pad_target / num_tokens_target * 100)

        return ParallelDataSet(data_source, data_target, data_label)


def get_validation_image_text_data_iter(data_loader: RawParallelDatasetLoader,
                             validation_source_root:str,
                             validation_source: str,
                             validation_target: str,
                             buckets: List[Tuple[int, int]],
                             bucket_batch_sizes: List[BucketBatchSize],
                             source_image_size: tuple,
                             vocab_target: vocab.Vocab,
                             max_seq_len_target: int,
                             batch_size: int,
                             fill_up: str,
                             use_feature_loader: bool = False,
                             preload_features: bool = False) -> 'ParallelSampleIter':
    """
    Returns a ParallelSampleIter for the validation data.
    """
    logger.info("=================================")
    logger.info("Creating validation data iterator")
    logger.info("=================================")

    validation_source_images = [FileListReader(validation_source, validation_source_root)]
    validation_target_sentences = SequenceReader(validation_target, vocab_target, add_bos=True, limit=None)

    validation_data_statistics = get_data_statistics(source_readers=None,
                                          target_reader=validation_target_sentences,
                                          buckets=buckets,
                                          length_ratio_mean=1.0,
                                          length_ratio_std=1.0,
                                          source_vocabs=[None],
                                          target_vocab=vocab_target)
    validation_data_statistics.log(bucket_batch_sizes)

    validation_data = data_loader.load(validation_source_images[0],
                                       validation_target_sentences,
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes,
                                                                                                fill_up)
    return ImageTextSampleIter(data=validation_data,
                               buckets=buckets,
                               batch_size=batch_size,
                               bucket_batch_sizes=bucket_batch_sizes,
                               image_size=source_image_size,
                               use_feature_loader=use_feature_loader,
                               preload_features=preload_features)


def get_training_image_text_data_iters(source_root: str,
                            source: str, target: str,
                            validation_source_root: str,
                            validation_source: str, validation_target: str,
                            vocab_target: vocab.Vocab,
                            vocab_target_path: Optional[str],
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            source_image_size: tuple,
                            fill_up: str,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int,
                            use_feature_loader: bool = False,
                            preload_features: bool = False) -> Tuple['ParallelSampleIter',
                                                                           'ParallelSampleIter',
                                                                           'DataConfig', 'DataInfo']:
    """
    Returns data iterators for training and validation data.

    :param source_root: Path to source images since the file in source contains relative paths.
    :param source: Path to source training data.
    :param target: Path to target training data.
    :param validation_source_root: Path to validation source images since the file in validation_source contains relative paths.
    :param validation_source: Path to source validation data.
    :param validation_target: Path to target validation data.
    :param vocab_target: Target vocabulary.
    :param vocab_target_path: Path to target vocabulary.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param source_image_size: size to resize the image to (for iterator)
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :param use_feature_loader: If True, features are loaded instead of images.
    :param preload_features: If use_feature_loader si True, this enables load all the feature to memory
    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")

    # define buckets
    buckets = define_empty_source_parallel_buckets(max_seq_len_target, bucket_width) if bucketing else [
        (0, max_seq_len_target)]

    source_images = [FileListReader(source, source_root)]
    target_sentences = SequenceReader(target, vocab_target, add_bos=True)

    # 2. pass: Get data statistics only on target (source not considered)
    data_statistics = get_data_statistics(source_readers=None,
                                          target_reader=target_sentences,
                                          buckets=buckets,
                                          length_ratio_mean=1.0,
                                          length_ratio_std=1.0,
                                          source_vocabs=[None],
                                          target_vocab=vocab_target)

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    data_loader = RawListTextDatasetLoader(buckets=buckets,
                                           eos_id=vocab_target[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    training_data = data_loader.load(source_images[0], target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    data_info = DataInfo(sources=source_images,
                         target=target,
                         source_vocabs=None,
                         target_vocab=vocab_target_path,
                         shared_vocab=False,
                         num_shards=1)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=0,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(source_images))

    # Add useful stuff to config_data
    config_data.source_root = source_root
    config_data.validation_source_root = validation_source_root
    config_data.use_feature_loader = use_feature_loader

    train_iter = ImageTextSampleIter(data=training_data,
                                     buckets=buckets,
                                     batch_size=batch_size,
                                     bucket_batch_sizes=bucket_batch_sizes,
                                     image_size=source_image_size,
                                     use_feature_loader=use_feature_loader,
                                     preload_features=preload_features)

    validation_iter = get_validation_image_text_data_iter(data_loader=data_loader,
                                               validation_source_root=validation_source_root,
                                               validation_source=validation_source,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_image_size=source_image_size,
                                               vocab_target=vocab_target,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size,
                                               fill_up=fill_up,
                                               use_feature_loader=use_feature_loader,
                                               preload_features=preload_features)

    return train_iter, validation_iter, config_data, data_info


class ImageTextSampleIter(ParallelSampleIter):
    """
    Data iterator on a bucketed ParallelDataSet which loads images in the source on the fly.
    It also resizes and preprocesses the images. Shuffles data at every reset and
    supports saving and loading the iterator state.
    """

    def __init__(self,
                 data: ParallelDataSet,
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 image_size: tuple,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 dtype='float32',
                 source_only=False,
                 use_feature_loader:bool = False,
                 preload_features: bool = False) -> None:
        super().__init__(data, buckets, batch_size, bucket_batch_sizes,
                         source_data_name, target_data_name, label_name, dtype)

        self.with_text = not source_only
        self.image_size = tuple(image_size)

        # Override provide_data to make sure to use images
        self.provide_data = [
            mx.io.DataDesc(name=self.source_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, ) + self.image_size,  # "NCHW"
                           layout=C.BATCH_MAJOR_IMAGE)
        ]
        if self.with_text:
            self.provide_data += [
                mx.io.DataDesc(name=self.target_data_name,
                               shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                               layout=C.BATCH_MAJOR)
            ]
        self.use_feature_loader = use_feature_loader
        self.preload_features =preload_features
        if self.use_feature_loader:
            self.data_loader = load_features
            # Load already everything to memory
            if self.preload_features:
                logger.info("Loading all the features to memory (this might take a while, be patient)...")
                start = time.time()
                self.loaded_source = {}
                for bucket in self.data.source:
                    for k in bucket:
                        if k not in self.loaded_source:  # avoid to load twice
                            self.loaded_source[k] = load_feature(k, self.image_size)
                logger.info("Feature loaded in {} seconds.".format(time.time()-start))
        else:
            self.data_loader = load_preprocess_images

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1

        batch_size = self.bucket_batch_sizes[i].batch_size
        source = self.data.source[i][j:j + batch_size]
        target = self.data.target[i][j:j + batch_size]
        if self.preload_features:
            loaded_source = []
            for k in source:
                loaded_source.append(self.loaded_source[k])
            loaded_source = mx.nd.array(loaded_source)
        else:
            loaded_source = mx.nd.array(self.data_loader(source, self.image_size))

        label = [self.data.label[i][j:j + batch_size]]

        provide_data = [
            mx.io.DataDesc(name=self.source_data_name, shape=loaded_source.shape, layout=C.BATCH_MAJOR_IMAGE),
        ]
        if self.with_text:
            provide_data += [
                mx.io.DataDesc(name=self.target_data_name, shape=target.shape, layout=C.BATCH_MAJOR)
        ]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        data = [loaded_source]
        if self.with_text:
            data += [target]
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    @staticmethod
    def visualize_batch(batch: mx.io.DataBatch,
                        reverse_vocab: Dict[int, str],
                        source_only: bool =False) -> None:

        try:  # Try to import matplotlib
            import matplotlib  # pylint: disable=import-error
        except ImportError as e:
            raise RuntimeError("Please install matplotlib.")
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        N = M = 4
        fig, axs = plt.subplots(N, M, figsize=(20, 10))
        # Remove axes
        for i in range(N):
            for j in range(M):
                axs[i, j].axis("off")
        for i, img in enumerate(batch.data[0]):
            # (channel, height, width) -> (height, width, channel)
            img_ = np.swapaxes(img.asnumpy(), 0, 2)
            img_ = np.swapaxes(img_, 0, 1)
            axs[i//N%M, i%N].imshow(np.uint8(img_))
            axs[i//N%M, i%N].axis("off")
            if not source_only:
                sentence = ""
                sentence_ids = batch.data[1][i].asnumpy()
                carry_on = jj = 0
                for j,v in enumerate(sentence_ids):
                    if reverse_vocab[v] not in C.VOCAB_SYMBOLS:  # Ignore for visualization
                        sentence += reverse_vocab[v]
                        carry_on += len(reverse_vocab[v])
                        if jj<len(sentence_ids):
                            if carry_on>=15:
                                sentence += "\n"
                                carry_on = 0
                            else:
                                sentence += " "
                        jj += 1
                axs[i//N%M, i%N].text(0, 8, sentence, fontsize=10,
                                      bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2})
        plt.show()

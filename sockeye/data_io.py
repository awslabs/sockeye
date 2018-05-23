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

"""
Implements data iterators and I/O related functions for sequence-to-sequence models.
"""
import bisect
import logging
import math
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any, cast, Dict, Iterator, Iterable, List, Optional, Sequence, Sized, Tuple

import mxnet as mx
import numpy as np

from . import config
from . import constants as C
from . import vocab
from .utils import check_condition, smart_open, get_tokens, OnlineMeanAndVariance

logger = logging.getLogger(__name__)


def define_buckets(max_seq_len: int, step=10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step until the final bucket fits max_seq_len.
    We then limit that bucket to max_seq_len (difference between semi-final and final bucket may be less than step).

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.
    :return: List of bucket sizes.
    """
    buckets = [bucket_len for bucket_len in range(step, max_seq_len + step, step)]
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            length_ratio: float = 1.0) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (max_seq_len_source, max_seq_len_target).  The longer side of the data uses
    steps of bucket_width while the shorter side uses steps scaled down by the average target/source length ratio.  If
    one side reaches its max_seq_len before the other, width of extra buckets on that side is fixed to that max_seq_len.

    :param max_seq_len_source: Maximum source bucket size.
    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    :param length_ratio: Length ratio of data (target/source).
    """
    source_step_size = bucket_width
    target_step_size = bucket_width
    if length_ratio >= 1.0:
        # target side is longer -> scale source
        source_step_size = max(1, int(round(bucket_width / length_ratio)))
    else:
        # source side is longer, -> scale target
        target_step_size = max(1, int(round(bucket_width * length_ratio)))
    source_buckets = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # Extra buckets
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    # minimum bucket size is 2 (as we add BOS symbol to target side)
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))
    # deduplicate for return
    buckets = list(OrderedDict.fromkeys(parallel_buckets))
    buckets.sort()
    return buckets


def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    """
    Given sequence length and a list of buckets, return corresponding bucket.

    :param seq_len: Sequence length.
    :param buckets: List of buckets.
    :return: Chosen bucket.
    """
    bucket_idx = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]


class BucketBatchSize:
    """
    :param bucket: The corresponding bucket.
    :param batch_size: Number of sequences in each batch.
    :param average_words_per_batch: Approximate number of non-padding tokens in each batch.
    """

    def __init__(self, bucket: Tuple[int, int], batch_size: int, average_words_per_batch: float) -> None:
        self.bucket = bucket
        self.batch_size = batch_size
        self.average_words_per_batch = average_words_per_batch


def define_bucket_batch_sizes(buckets: List[Tuple[int, int]],
                              batch_size: int,
                              batch_by_words: bool,
                              batch_num_devices: int,
                              data_target_average_len: List[Optional[float]]) -> List[BucketBatchSize]:
    """
    Computes bucket-specific batch sizes (sentences, average_words).

    If sentence-based batching: number of sentences is the same for each batch, determines the
    number of words. Hence all batch sizes for each bucket are equal.

    If word-based batching: number of sentences for each batch is set to the multiple of number
    of devices that produces the number of words closest to the target batch size.  Average
    target sentence length (non-padding symbols) is used for word number calculations.

    :param buckets: Bucket list.
    :param batch_size: Batch size.
    :param batch_by_words: Batch by words.
    :param batch_num_devices: Number of devices.
    :param data_target_average_len: Optional average target length for each bucket.
    """
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[BucketBatchSize]
    largest_total_num_words = 0
    for buck_idx, bucket in enumerate(buckets):
        # Target/label length with padding
        padded_seq_len = bucket[1]
        # Average target/label length excluding padding
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len
        average_seq_len = data_target_average_len[buck_idx]

        # Word-based: num words determines num sentences
        # Sentence-based: num sentences determines num words
        if batch_by_words:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            # Multiple of number of devices (int) closest to target number of words, assuming each sentence is of
            # average length
            batch_size_seq = batch_num_devices * max(1, round((batch_size / average_seq_len) / batch_num_devices))
            batch_size_word = batch_size_seq * average_seq_len
        else:
            batch_size_seq = batch_size
            batch_size_word = batch_size_seq * average_seq_len
        bucket_batch_sizes.append(BucketBatchSize(bucket, batch_size_seq, batch_size_word))
        # Track largest number of target word samples in a batch
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * padded_seq_len)

    # Final step: guarantee that largest bucket by sequence length also has largest total batch size.
    # When batching by sentences, this will already be the case.
    if batch_by_words:
        padded_seq_len = max(*buckets[-1])
        average_seq_len = data_target_average_len[-1]
        while bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_num_words:
            bucket_batch_sizes[-1] = BucketBatchSize(
                bucket_batch_sizes[-1].bucket,
                bucket_batch_sizes[-1].batch_size + batch_num_devices,
                bucket_batch_sizes[-1].average_words_per_batch + batch_num_devices * average_seq_len)
    return bucket_batch_sizes


def calculate_length_statistics(source_iterables: Sequence[Iterable[Any]],
                                target_iterable: Iterable[Any],
                                max_seq_len_source: int,
                                max_seq_len_target: int) -> 'LengthStatistics':
    """
    Returns mean and standard deviation of target-to-source length ratios of parallel corpus.

    :param source_iterables: Source sequence readers.
    :param target_iterable: Target sequence reader.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :return: The number of sentences as well as the mean and standard deviation of target to source length ratios.
    """
    mean_and_variance = OnlineMeanAndVariance()

    for sources, target in parallel_iter(source_iterables, target_iterable):
        source_len = len(sources[0])
        target_len = len(target)
        if source_len > max_seq_len_source or target_len > max_seq_len_target:
            continue

        length_ratio = target_len / source_len
        mean_and_variance.update(length_ratio)

    num_sents = mean_and_variance.count
    mean = mean_and_variance.mean
    std = math.sqrt(mean_and_variance.variance)
    return LengthStatistics(num_sents, mean, std)


def analyze_sequence_lengths(sources: List[str],
                             target: str,
                             vocab_sources: List[vocab.Vocab],
                             vocab_target: vocab.Vocab,
                             max_seq_len_source: int,
                             max_seq_len_target: int) -> 'LengthStatistics':
    train_sources_sentences, train_target_sentences = create_sequence_readers(sources, target, vocab_sources,
                                                                              vocab_target)

    length_statistics = calculate_length_statistics(train_sources_sentences, train_target_sentences,
                                                    max_seq_len_source,
                                                    max_seq_len_target)

    logger.info("%d sequences of maximum length (%d, %d) in '%s' and '%s'.",
                length_statistics.num_sents, max_seq_len_source, max_seq_len_target, sources[0], target)
    logger.info("Mean training target/source length ratio: %.2f (+-%.2f)",
                length_statistics.length_ratio_mean,
                length_statistics.length_ratio_std)
    return length_statistics


def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)


class DataStatisticsAccumulator:

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 vocab_source: Dict[str, int],
                 vocab_target: Dict[str, int],
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.unk_id_source = vocab_source[C.UNK_SYMBOL]
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_source = len(vocab_source)
        self.size_vocab_target = len(vocab_target)
        self.num_sents = 0
        self.num_discarded = 0
        self.num_tokens_source = 0
        self.num_tokens_target = 0
        self.num_unks_source = 0
        self.num_unks_target = 0
        self.max_observed_len_source = 0
        self.max_observed_len_target = 0
        self._mean_len_target_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence_pair(self,
                      source: List[int],
                      target: List[int],
                      bucket_idx: Optional[int]):
        if bucket_idx is None:
            self.num_discarded += 1
            return

        source_len = len(source)
        target_len = len(target)

        self._mean_len_target_per_bucket[bucket_idx].update(target_len)

        self.num_sents += 1
        self.num_tokens_source += source_len
        self.num_tokens_target += target_len
        self.max_observed_len_source = max(source_len, self.max_observed_len_source)
        self.max_observed_len_target = max(target_len, self.max_observed_len_target)

        self.num_unks_source += source.count(self.unk_id_source)
        self.num_unks_target += target.count(self.unk_id_target)

    @property
    def mean_len_target_per_bucket(self) -> List[Optional[float]]:
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None
                for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def statistics(self):
        num_sents_per_bucket = [mean_and_variance.count for mean_and_variance in self._mean_len_target_per_bucket]
        return DataStatistics(num_sents=self.num_sents,
                              num_discarded=self.num_discarded,
                              num_tokens_source=self.num_tokens_source,
                              num_tokens_target=self.num_tokens_target,
                              num_unks_source=self.num_unks_source,
                              num_unks_target=self.num_unks_target,
                              max_observed_len_source=self.max_observed_len_source,
                              max_observed_len_target=self.max_observed_len_target,
                              size_vocab_source=self.size_vocab_source,
                              size_vocab_target=self.size_vocab_target,
                              length_ratio_mean=self.length_ratio_mean,
                              length_ratio_std=self.length_ratio_std,
                              buckets=self.buckets,
                              num_sents_per_bucket=num_sents_per_bucket,
                              mean_len_target_per_bucket=self.mean_len_target_per_bucket)


def shard_data(source_fnames: List[str],
               target_fname: str,
               source_vocabs: List[vocab.Vocab],
               target_vocab: vocab.Vocab,
               num_shards: int,
               buckets: List[Tuple[int, int]],
               length_ratio_mean: float,
               length_ratio_std: float,
               output_prefix: str) -> Tuple[List[Tuple[List[str], str, 'DataStatistics']], 'DataStatistics']:
    """
    Assign int-coded source/target sentence pairs to shards at random.

    :param source_fnames: The path to the source text (and optional token-parallel factor files).
    :param target_fname: The file name of the target file.
    :param source_vocabs: Source vocabulary (and optional source factor vocabularies).
    :param target_vocab: Target vocabulary.
    :param num_shards: The total number of shards.
    :param buckets: Bucket list.
    :param length_ratio_mean: Mean length ratio.
    :param length_ratio_std: Standard deviation of length ratios.
    :param output_prefix: The prefix under which the shard files will be created.
    :return: Tuple of source (and source factor) file names, target file names and statistics for each shard,
             as well as global statistics.
    """
    os.makedirs(output_prefix, exist_ok=True)
    sources_shard_fnames = [[os.path.join(output_prefix, C.SHARD_SOURCE % i) + ".%d" % f for i in range(num_shards)]
                            for f in range(len(source_fnames))]
    target_shard_fnames = [os.path.join(output_prefix, C.SHARD_TARGET % i)
                           for i in range(num_shards)]  # type: List[str]

    data_stats_accumulator = DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocab,
                                                       length_ratio_mean, length_ratio_std)
    per_shard_stat_accumulators = [DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocab, length_ratio_mean,
                                                             length_ratio_std) for shard_idx in range(num_shards)]

    with ExitStack() as exit_stack:
        sources_shards = [[exit_stack.enter_context(smart_open(f, mode="wt")) for f in sources_shard_fnames[i]] for i in
                          range(len(source_fnames))]
        target_shards = [exit_stack.enter_context(smart_open(f, mode="wt")) for f in target_shard_fnames]

        source_readers, target_reader = create_sequence_readers(source_fnames, target_fname,
                                                                source_vocabs, target_vocab)

        random_shard_iter = iter(lambda: random.randrange(num_shards), None)

        for (sources, target), random_shard_index in zip(parallel_iter(source_readers, target_reader),
                                                         random_shard_iter):
            random_shard_index = cast(int, random_shard_index)
            source_len = len(sources[0])
            target_len = len(target)

            buck_idx, buck = get_parallel_bucket(buckets, source_len, target_len)
            data_stats_accumulator.sequence_pair(sources[0], target, buck_idx)
            per_shard_stat_accumulators[random_shard_index].sequence_pair(sources[0], target, buck_idx)

            if buck is None:
                continue

            for i, line in enumerate(sources):
                sources_shards[i][random_shard_index].write(ids2strids(line) + "\n")
            target_shards[random_shard_index].write(ids2strids(target) + "\n")

    per_shard_stats = [shard_stat_accumulator.statistics for shard_stat_accumulator in per_shard_stat_accumulators]

    sources_shard_fnames_by_shards = zip(*sources_shard_fnames)  # type: List[List[str]]

    return list(
        zip(sources_shard_fnames_by_shards, target_shard_fnames, per_shard_stats)), data_stats_accumulator.statistics


class RawParallelDatasetLoader:
    """
    Loads a data set of variable-length parallel source/target sequences into buckets of NDArrays.

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
             source_iterables: Sequence[Iterable],
             target_iterable: Iterable,
             num_samples_per_bucket: List[int]) -> 'ParallelDataSet':

        assert len(num_samples_per_bucket) == len(self.buckets)
        num_factors = len(source_iterables)

        data_source = [np.full((num_samples, source_len, num_factors), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_label = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                      for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]

        bucket_sample_index = [0 for _ in self.buckets]

        # track amount of padding introduced through bucketing
        num_tokens_source = 0
        num_tokens_target = 0
        num_pad_source = 0
        num_pad_target = 0

        # Bucket sentences as padded np arrays
        for sources, target in parallel_iter(source_iterables, target_iterable):
            source_len = len(sources[0])
            target_len = len(target)
            buck_index, buck = get_parallel_bucket(self.buckets, source_len, target_len)
            if buck is None:
                continue  # skip this sentence pair

            num_tokens_source += buck[0]
            num_tokens_target += buck[1]
            num_pad_source += buck[0] - source_len
            num_pad_target += buck[1] - target_len

            sample_index = bucket_sample_index[buck_index]
            for i, s in enumerate(sources):
                data_source[buck_index][sample_index, 0:source_len, i] = s
            data_target[buck_index][sample_index, :target_len] = target
            # NOTE(fhieber): while this is wasteful w.r.t memory, we need to explicitly create the label sequence
            # with the EOS symbol here sentence-wise and not per-batch due to variable sequence length within a batch.
            # Once MXNet allows item assignments given a list of indices (probably MXNet 1.0): e.g a[[0,1,5,2]] = x,
            # we can try again to compute the label sequence on the fly in next().
            data_label[buck_index][sample_index, :target_len] = target[1:] + [self.eos_id]

            bucket_sample_index[buck_index] += 1

        for i in range(len(data_source)):
            data_source[i] = mx.nd.array(data_source[i], dtype=self.dtype)
            data_target[i] = mx.nd.array(data_target[i], dtype=self.dtype)
            data_label[i] = mx.nd.array(data_label[i], dtype=self.dtype)

        if num_tokens_source > 0 and num_tokens_target > 0:
            logger.info("Created bucketed parallel data set. Introduced padding: source=%.1f%% target=%.1f%%)",
                        num_pad_source / num_tokens_source * 100,
                        num_pad_target / num_tokens_target * 100)

        return ParallelDataSet(data_source, data_target, data_label)


def get_num_shards(num_samples: int, samples_per_shard: int, min_num_shards: int) -> int:
    """
    Returns the number of shards.

    :param num_samples: Number of training data samples.
    :param samples_per_shard: Samples per shard.
    :param min_num_shards: Minimum number of shards.
    :return: Number of shards.
    """
    return max(int(math.ceil(num_samples / samples_per_shard)), min_num_shards)


def prepare_data(source_fnames: List[str],
                 target_fname: str,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 source_vocab_paths: List[Optional[str]],
                 target_vocab_path: Optional[str],
                 shared_vocab: bool,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 bucketing: bool,
                 bucket_width: int,
                 samples_per_shard: int,
                 min_num_shards: int,
                 output_prefix: str,
                 keep_tmp_shard_files: bool = False):
    logger.info("Preparing data.")
    # write vocabularies to data folder
    vocab.save_source_vocabs(source_vocabs, output_prefix)
    vocab.save_target_vocab(target_vocab, output_prefix)

    # Pass 1: get target/source length ratios.
    length_statistics = analyze_sequence_lengths(source_fnames, target_fname, source_vocabs, target_vocab,
                                                 max_seq_len_source, max_seq_len_target)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width,
                                      length_statistics.length_ratio_mean) if bucketing else [
        (max_seq_len_source, max_seq_len_target)]
    logger.info("Buckets: %s", buckets)

    # Pass 2: Randomly assign data to data shards
    # no pre-processing yet, just write the sentences to different files
    num_shards = get_num_shards(length_statistics.num_sents, samples_per_shard, min_num_shards)
    logger.info("%d samples will be split into %d shard(s) (requested samples/shard=%d, min_num_shards=%d)."
                % (length_statistics.num_sents, num_shards, samples_per_shard, min_num_shards))
    shards, data_statistics = shard_data(source_fnames=source_fnames,
                                         target_fname=target_fname,
                                         source_vocabs=source_vocabs,
                                         target_vocab=target_vocab,
                                         num_shards=num_shards,
                                         buckets=buckets,
                                         length_ratio_mean=length_statistics.length_ratio_mean,
                                         length_ratio_std=length_statistics.length_ratio_std,
                                         output_prefix=output_prefix)
    data_statistics.log()

    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    # 3. convert each shard to serialized ndarrays
    for shard_idx, (shard_sources, shard_target, shard_stats) in enumerate(shards):
        sources_sentences = [SequenceReader(s) for s in shard_sources]
        target_sentences = SequenceReader(shard_target)
        dataset = data_loader.load(sources_sentences, target_sentences, shard_stats.num_sents_per_bucket)
        shard_fname = os.path.join(output_prefix, C.SHARD_NAME % shard_idx)
        shard_stats.log()
        logger.info("Writing '%s'", shard_fname)
        dataset.save(shard_fname)

        if not keep_tmp_shard_files:
            for f in shard_sources:
                os.remove(f)
            os.remove(shard_target)

    data_info = DataInfo(sources=[os.path.abspath(fname) for fname in source_fnames],
                         target=os.path.abspath(target_fname),
                         source_vocabs=source_vocab_paths,
                         target_vocab=target_vocab_path,
                         shared_vocab=shared_vocab,
                         num_shards=num_shards)
    data_info_fname = os.path.join(output_prefix, C.DATA_INFO)
    logger.info("Writing data info to '%s'", data_info_fname)
    data_info.save(data_info_fname)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(source_fnames),
                             source_with_eos=True)
    config_data_fname = os.path.join(output_prefix, C.DATA_CONFIG)
    logger.info("Writing data config to '%s'", config_data_fname)
    config_data.save(config_data_fname)

    version_file = os.path.join(output_prefix, C.PREPARED_DATA_VERSION_FILE)

    with open(version_file, "w") as version_out:
        version_out.write(str(C.PREPARED_DATA_VERSION))


def get_data_statistics(source_readers: Sequence[Iterable],
                        target_reader: Iterable,
                        buckets: List[Tuple[int, int]],
                        length_ratio_mean: float,
                        length_ratio_std: float,
                        source_vocabs: List[vocab.Vocab],
                        target_vocab: vocab.Vocab) -> 'DataStatistics':
    data_stats_accumulator = DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocab,
                                                       length_ratio_mean, length_ratio_std)

    for sources, target in parallel_iter(source_readers, target_reader):
        buck_idx, buck = get_parallel_bucket(buckets, len(sources[0]), len(target))
        data_stats_accumulator.sequence_pair(sources[0], target, buck_idx)

    return data_stats_accumulator.statistics


def get_validation_data_iter(data_loader: RawParallelDatasetLoader,
                             validation_sources: List[str],
                             validation_target: str,
                             buckets: List[Tuple[int, int]],
                             bucket_batch_sizes: List[BucketBatchSize],
                             source_vocabs: List[vocab.Vocab],
                             target_vocab: vocab.Vocab,
                             max_seq_len_source: int,
                             max_seq_len_target: int,
                             batch_size: int,
                             fill_up: str) -> 'ParallelSampleIter':
    """
    Returns a ParallelSampleIter for the validation data.
    """
    logger.info("=================================")
    logger.info("Creating validation data iterator")
    logger.info("=================================")
    validation_length_statistics = analyze_sequence_lengths(validation_sources, validation_target,
                                                            source_vocabs, target_vocab,
                                                            max_seq_len_source, max_seq_len_target)
    validation_sources_sentences, validation_target_sentences = create_sequence_readers(validation_sources,
                                                                                        validation_target,
                                                                                        source_vocabs, target_vocab)

    validation_data_statistics = get_data_statistics(validation_sources_sentences,
                                                     validation_target_sentences,
                                                     buckets,
                                                     validation_length_statistics.length_ratio_mean,
                                                     validation_length_statistics.length_ratio_std,
                                                     source_vocabs, target_vocab)

    validation_data_statistics.log(bucket_batch_sizes)

    validation_data = data_loader.load(validation_sources_sentences, validation_target_sentences,
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes,
                                                                                                fill_up)

    return ParallelSampleIter(data=validation_data,
                              buckets=buckets,
                              batch_size=batch_size,
                              bucket_batch_sizes=bucket_batch_sizes,
                              num_factors=len(validation_sources))


def get_prepared_data_iters(prepared_data_dir: str,
                            validation_sources: List[str],
                            validation_target: str,
                            shared_vocab: bool,
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            fill_up: str) -> Tuple['BaseParallelSampleIter',
                                                   'BaseParallelSampleIter',
                                                   'DataConfig', List[vocab.Vocab], vocab.Vocab]:
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")

    version_file = os.path.join(prepared_data_dir, C.PREPARED_DATA_VERSION_FILE)
    with open(version_file) as version_in:
        version = int(version_in.read())
        check_condition(version == C.PREPARED_DATA_VERSION,
                        "The dataset %s was written in an old and incompatible format. Please rerun data "
                        "preparation with a current version of Sockeye." % prepared_data_dir)
    info_file = os.path.join(prepared_data_dir, C.DATA_INFO)
    check_condition(os.path.exists(info_file),
                    "Could not find data info %s. Are you sure %s is a directory created with "
                    "python -m sockeye.prepare_data?" % (info_file, prepared_data_dir))
    data_info = cast(DataInfo, DataInfo.load(info_file))
    config_file = os.path.join(prepared_data_dir, C.DATA_CONFIG)
    check_condition(os.path.exists(config_file),
                    "Could not find data config %s. Are you sure %s is a directory created with "
                    "python -m sockeye.prepare_data?" % (config_file, prepared_data_dir))
    config_data = cast(DataConfig, DataConfig.load(config_file))
    shard_fnames = [os.path.join(prepared_data_dir,
                                 C.SHARD_NAME % shard_idx) for shard_idx in range(data_info.num_shards)]
    for shard_fname in shard_fnames:
        check_condition(os.path.exists(shard_fname), "Shard %s does not exist." % shard_fname)

    check_condition(shared_vocab == data_info.shared_vocab, "Shared config needed (e.g. for weight tying), but "
                                                            "data was prepared without a shared vocab. Use %s when "
                                                            "preparing the data." % C.VOCAB_ARG_SHARED_VOCAB)

    source_vocabs = vocab.load_source_vocabs(prepared_data_dir)
    target_vocab = vocab.load_target_vocab(prepared_data_dir)

    check_condition(len(source_vocabs) == len(data_info.sources),
                    "Wrong number of source vocabularies. Found %d, need %d." % (len(source_vocabs),
                                                                                 len(data_info.sources)))

    buckets = config_data.data_statistics.buckets
    max_seq_len_source = config_data.max_seq_len_source
    max_seq_len_target = config_data.max_seq_len_target

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   config_data.data_statistics.average_len_target_per_bucket)

    config_data.data_statistics.log(bucket_batch_sizes)

    train_iter = ShardedParallelSampleIter(shard_fnames,
                                           buckets,
                                           batch_size,
                                           bucket_batch_sizes,
                                           fill_up,
                                           num_factors=len(data_info.sources))

    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=target_vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size,
                                               fill_up=fill_up)

    return train_iter, validation_iter, config_data, source_vocabs, target_vocab


def get_training_data_iters(sources: List[str],
                            target: str,
                            validation_sources: List[str],
                            validation_target: str,
                            source_vocabs: List[vocab.Vocab],
                            target_vocab: vocab.Vocab,
                            source_vocab_paths: List[Optional[str]],
                            target_vocab_path: Optional[str],
                            shared_vocab: bool,
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            fill_up: str,
                            max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int) -> Tuple['BaseParallelSampleIter',
                                                        'BaseParallelSampleIter',
                                                        'DataConfig', 'DataInfo']:
    """
    Returns data iterators for training and validation data.

    :param sources: Path to source training data (with optional factor data paths).
    :param target: Path to target training data.
    :param validation_sources: Path to source validation data (with optional factor data paths).
    :param validation_target: Path to target validation data.
    :param source_vocabs: Source vocabulary and optional factor vocabularies.
    :param target_vocab: Target vocabulary.
    :param source_vocab_paths: Path to source vocabulary.
    :param target_vocab_path: Path to target vocabulary.
    :param shared_vocab: Whether the vocabularies are shared.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")
    # Pass 1: get target/source length ratios.
    length_statistics = analyze_sequence_lengths(sources, target, source_vocabs, target_vocab,
                                                 max_seq_len_source, max_seq_len_target)
    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width,
                                      length_statistics.length_ratio_mean) if bucketing else [
        (max_seq_len_source, max_seq_len_target)]

    sources_sentences, target_sentences = create_sequence_readers(sources, target, source_vocabs, target_vocab)

    # 2. pass: Get data statistics
    data_statistics = get_data_statistics(sources_sentences, target_sentences, buckets,
                                          length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                          source_vocabs, target_vocab)

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    training_data = data_loader.load(sources_sentences, target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, fill_up)

    data_info = DataInfo(sources=sources,
                         target=target,
                         source_vocabs=source_vocab_paths,
                         target_vocab=target_vocab_path,
                         shared_vocab=shared_vocab,
                         num_shards=1)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(sources),
                             source_with_eos=True)

    train_iter = ParallelSampleIter(data=training_data,
                                    buckets=buckets,
                                    batch_size=batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_factors=len(sources))

    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=target_vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size,
                                               fill_up=fill_up)

    return train_iter, validation_iter, config_data, data_info


class LengthStatistics(config.Config):

    def __init__(self,
                 num_sents: int,
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std


class DataStatistics(config.Config):

    def __init__(self,
                 num_sents: int,
                 num_discarded,
                 num_tokens_source,
                 num_tokens_target,
                 num_unks_source,
                 num_unks_target,
                 max_observed_len_source,
                 max_observed_len_target,
                 size_vocab_source,
                 size_vocab_target,
                 length_ratio_mean,
                 length_ratio_std,
                 buckets: List[Tuple[int, int]],
                 num_sents_per_bucket: List[int],
                 mean_len_target_per_bucket: List[Optional[float]]) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.num_discarded = num_discarded
        self.num_tokens_source = num_tokens_source
        self.num_tokens_target = num_tokens_target
        self.num_unks_source = num_unks_source
        self.num_unks_target = num_unks_target
        self.max_observed_len_source = max_observed_len_source
        self.max_observed_len_target = max_observed_len_target
        self.size_vocab_source = size_vocab_source
        self.size_vocab_target = size_vocab_target
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.buckets = buckets
        self.num_sents_per_bucket = num_sents_per_bucket
        self.average_len_target_per_bucket = mean_len_target_per_bucket

    def log(self, bucket_batch_sizes: Optional[List[BucketBatchSize]] = None):
        logger.info("Tokens: source %d target %d", self.num_tokens_source, self.num_tokens_target)
        if self.num_tokens_source > 0 and self.num_tokens_target > 0:
            logger.info("Vocabulary coverage: source %.0f%% target %.0f%%",
                        (1 - self.num_unks_source / self.num_tokens_source) * 100,
                        (1 - self.num_unks_target / self.num_tokens_target) * 100)
        logger.info("%d sequences across %d buckets", self.num_sents, len(self.num_sents_per_bucket))
        logger.info("%d sequences did not fit into buckets and were discarded", self.num_discarded)
        if bucket_batch_sizes is not None:
            describe_data_and_buckets(self, bucket_batch_sizes)


def describe_data_and_buckets(data_statistics: DataStatistics, bucket_batch_sizes: List[BucketBatchSize]):
    """
    Describes statistics across buckets
    """
    check_condition(len(bucket_batch_sizes) == len(data_statistics.buckets),
                    "Number of bucket batch sizes (%d) does not match number of buckets in statistics (%d)."
                    % (len(bucket_batch_sizes), len(data_statistics.buckets)))
    for bucket_batch_size, num_seq in zip(bucket_batch_sizes, data_statistics.num_sents_per_bucket):
        if num_seq > 0:
            logger.info("Bucket %s: %d samples in %d batches of %d, ~%.1f tokens/batch.",
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.average_words_per_batch)


class DataInfo(config.Config):
    """
    Stores training data information that is not relevant for inference.
    """

    def __init__(self,
                 sources: List[str],
                 target: str,
                 source_vocabs: List[Optional[str]],
                 target_vocab: Optional[str],
                 shared_vocab: bool,
                 num_shards: int) -> None:
        super().__init__()
        self.sources = sources
        self.target = target
        self.source_vocabs = source_vocabs
        self.target_vocab = target_vocab
        self.shared_vocab = shared_vocab
        self.num_shards = num_shards


class DataConfig(config.Config):
    """
    Stores data statistics relevant for inference.
    """

    def __init__(self,
                 data_statistics: DataStatistics,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 num_source_factors: int,
                 source_with_eos: bool = False) -> None:
        super().__init__()
        self.data_statistics = data_statistics
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.num_source_factors = num_source_factors
        self.source_with_eos = source_with_eos


def read_content(path: str, limit: Optional[int] = None) -> Iterator[List[str]]:
    """
    Returns a list of tokens for each line in path up to a limit.

    :param path: Path to files containing sentences.
    :param limit: How many lines to read from path.
    :return: Iterator over lists of words.
    """
    with smart_open(path) as indata:
        for i, line in enumerate(indata):
            if limit is not None and i == limit:
                break
            yield list(get_tokens(line))


def tokens2ids(tokens: Iterable[str], vocab: Dict[str, int]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of tokens and vocab.

    :param tokens: List of string tokens.
    :param vocab: Vocabulary (containing UNK symbol).
    :return: List of word ids.
    """
    return [vocab.get(w, vocab[C.UNK_SYMBOL]) for w in tokens]


def strids2ids(tokens: Iterable[str]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of string ids.


    :param tokens: List of integer tokens.
    :return: List of word ids.
    """
    return list(map(int, tokens))


def ids2strids(ids: Iterable[int]) -> str:
    """
    Returns a string representation of a sequence of integers.

    :param ids: Sequence of integers.
    :return: String sequence
    """
    return " ".join(map(str, ids))


class SequenceReader(Iterable):
    """
    Reads sequence samples from path and (optionally) creates integer id sequences.
    Streams from disk, instead of loading all samples into memory.
    If vocab is None, the sequences in path are assumed to be integers coded as strings.
    Empty sequences are yielded as None.

    :param path: Path to read data from.
    :param vocab: Optional mapping from strings to integer ids.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    """

    def __init__(self,
                 path: str,
                 vocabulary: Optional[vocab.Vocab] = None,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 limit: Optional[int] = None) -> None:
        self.path = path
        self.vocab = vocabulary
        self.bos_id = None
        self.eos_id = None
        if vocabulary is not None:
            assert C.UNK_SYMBOL in vocabulary
            assert vocabulary[C.PAD_SYMBOL] == C.PAD_ID
            assert C.BOS_SYMBOL in vocabulary
            assert C.EOS_SYMBOL in vocabulary
            self.bos_id = vocabulary[C.BOS_SYMBOL]
            self.eos_id = vocabulary[C.EOS_SYMBOL]
        else:
            check_condition(not add_bos and not add_eos, "Adding a BOS or EOS symbol requires a vocabulary")
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.limit = limit

    def __iter__(self):
        for tokens in read_content(self.path, self.limit):
            if self.vocab is not None:
                sequence = tokens2ids(tokens, self.vocab)
            else:
                sequence = strids2ids(tokens)
            if len(sequence) == 0:
                yield None
                continue
            if self.add_bos:
                sequence.insert(0, self.bos_id)
            if self.add_eos:
                sequence.append(self.eos_id)
            yield sequence


def create_sequence_readers(sources: List[str], target: str,
                            vocab_sources: List[vocab.Vocab],
                            vocab_target: vocab.Vocab) -> Tuple[List[SequenceReader], SequenceReader]:
    """
    Create source readers with EOS and target readers with BOS.

    :param sources: The file names of source data and factors.
    :param target: The file name of the target data.
    :param vocab_sources: The source vocabularies.
    :param vocab_target: The target vocabularies.
    :return: The source sequence readers and the target reader.
    """
    source_sequence_readers = [SequenceReader(source, vocab, add_eos=True) for source, vocab in
                               zip(sources, vocab_sources)]
    target_sequence_reader = SequenceReader(target, vocab_target, add_bos=True)
    return source_sequence_readers, target_sequence_reader


def parallel_iter(source_iters: Sequence[Iterable[Optional[Any]]], target_iterable: Iterable[Optional[Any]]):
    """
    Yields parallel source(s), target sequences from iterables.
    Checks for token parallelism in source sequences.
    Skips pairs where element in at least one iterable is None.
    Checks that all iterables have the same number of elements.
    """
    num_skipped = 0
    source_iters = [iter(s) for s in source_iters]
    target_iter = iter(target_iterable)
    for sources, target in zip(zip(*source_iters), target_iter):
        if any((s is None for s in sources)) or target is None:
            num_skipped += 1
            continue
        check_condition(are_token_parallel(sources), "Source sequences are not token-parallel: %s" % (str(sources)))
        yield sources, target

    if num_skipped > 0:
        logger.warning("Parallel reading of sequences skipped %d elements", num_skipped)

    check_condition(
        all(next(cast(Iterator, s), None) is None for s in source_iters) and next(cast(Iterator, target_iter),
                                                                                  None) is None,
        "Different number of lines in source(s) and target iterables.")


def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)


def get_parallel_bucket(buckets: List[Tuple[int, int]],
                        length_source: int,
                        length_target: int) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            return j, (source_bkt, target_bkt)
    return None, None


class ParallelDataSet(Sized):
    """
    Bucketed parallel data set with labels
    """

    def __init__(self,
                 source: List[mx.nd.array],
                 target: List[mx.nd.array],
                 label: List[mx.nd.array]) -> None:
        check_condition(len(source) == len(target) == len(label),
                        "Number of buckets for source/target/label do not match: %d/%d/%d." % (len(source),
                                                                                               len(target),
                                                                                               len(label)))
        self.source = source
        self.target = target
        self.label = label

    def __len__(self) -> int:
        return len(self.source)

    def get_bucket_counts(self):
        return [len(self.source[buck_idx]) for buck_idx in range(len(self))]

    def save(self, fname: str):
        """
        Saves the dataset to a binary .npy file.
        """
        mx.nd.save(fname, self.source + self.target + self.label)

    @staticmethod
    def load(fname: str) -> 'ParallelDataSet':
        """
        Loads a dataset from a binary .npy file.
        """
        data = mx.nd.load(fname)
        n = len(data) // 3
        source = data[:n]
        target = data[n:2 * n]
        label = data[2 * n:]
        assert len(source) == len(target) == len(label)
        return ParallelDataSet(source, target, label)

    def fill_up(self,
                bucket_batch_sizes: List[BucketBatchSize],
                fill_up: str,
                seed: int = 42) -> 'ParallelDataSet':
        """
        Returns a new dataset with buckets filled up using the specified fill-up strategy.

        :param bucket_batch_sizes: Bucket batch sizes.
        :param fill_up: Fill-up strategy.
        :param seed: The random seed used for sampling sentences to fill up.
        :return: New dataset with buckets filled up to the next multiple of batch size
        """
        source = list(self.source)
        target = list(self.target)
        label = list(self.label)

        rs = np.random.RandomState(seed)

        for bucket_idx in range(len(self)):
            bucket = bucket_batch_sizes[bucket_idx].bucket
            bucket_batch_size = bucket_batch_sizes[bucket_idx].batch_size
            bucket_source = self.source[bucket_idx]
            bucket_target = self.target[bucket_idx]
            bucket_label = self.label[bucket_idx]
            num_samples = bucket_source.shape[0]

            if num_samples % bucket_batch_size != 0:
                if fill_up == 'replicate':
                    rest = bucket_batch_size - num_samples % bucket_batch_size
                    logger.info("Replicating %d random samples from %d samples in bucket %s "
                                "to size it to multiple of %d",
                                rest, num_samples, bucket, bucket_batch_size)
                    random_indices = mx.nd.array(rs.randint(num_samples, size=rest))
                    source[bucket_idx] = mx.nd.concat(bucket_source, bucket_source.take(random_indices), dim=0)
                    target[bucket_idx] = mx.nd.concat(bucket_target, bucket_target.take(random_indices), dim=0)
                    label[bucket_idx] = mx.nd.concat(bucket_label, bucket_label.take(random_indices), dim=0)
                else:
                    raise NotImplementedError('Unknown fill-up strategy')

        return ParallelDataSet(source, target, label)

    def permute(self, permutations: List[mx.nd.NDArray]) -> 'ParallelDataSet':
        assert len(self) == len(permutations)
        source = []
        target = []
        label = []
        for buck_idx in range(len(self)):
            num_samples = self.source[buck_idx].shape[0]
            if num_samples:  # not empty bucket
                permutation = permutations[buck_idx]
                source.append(self.source[buck_idx].take(permutation))
                target.append(self.target[buck_idx].take(permutation))
                label.append(self.label[buck_idx].take(permutation))
            else:
                source.append(self.source[buck_idx])
                target.append(self.target[buck_idx])
                label.append(self.label[buck_idx])

        return ParallelDataSet(source, target, label)


def get_permutations(bucket_counts: List[int]) -> Tuple[List[mx.nd.NDArray], List[mx.nd.NDArray]]:
    """
    Returns the indices of a random permutation for each bucket and the corresponding inverse permutations that can
    restore the original order of the data if applied to the permuted data.

    :param bucket_counts: The number of elements per bucket.
    :return: For each bucket a permutation and inverse permutation is returned.
    """
    data_permutations = []  # type: List[mx.nd.NDArray]
    inverse_data_permutations = []  # type: List[mx.nd.NDArray]
    for num_samples in bucket_counts:
        if num_samples == 0:
            num_samples = 1
        # new random order:
        data_permutation = np.random.permutation(num_samples)
        inverse_data_permutation = np.empty(num_samples, np.int32)
        inverse_data_permutation[data_permutation] = np.arange(num_samples)
        inverse_data_permutation = mx.nd.array(inverse_data_permutation)
        data_permutation = mx.nd.array(data_permutation)

        data_permutations.append(data_permutation)
        inverse_data_permutations.append(inverse_data_permutation)
    return data_permutations, inverse_data_permutations


def get_batch_indices(data: ParallelDataSet,
                      bucket_batch_sizes: List[BucketBatchSize]) -> List[Tuple[int, int]]:
    """
    Returns a list of index tuples that index into the bucket and the start index inside a bucket given
    the batch size for a bucket. These indices are valid for the given dataset.

    :param data: Data to create indices for.
    :param bucket_batch_sizes: Bucket batch sizes.
    :return: List of 2d indices.
    """
    # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
    idxs = []  # type: List[Tuple[int, int]]
    for buck_idx, buck in enumerate(data.source):
        bucket = bucket_batch_sizes[buck_idx].bucket
        batch_size = bucket_batch_sizes[buck_idx].batch_size
        num_samples = data.source[buck_idx].shape[0]
        rest = num_samples % batch_size
        if rest > 0:
            logger.info("Ignoring %d samples from bucket %s with %d samples due to incomplete batch",
                        rest, bucket, num_samples)
        idxs.extend([(buck_idx, j) for j in range(0, num_samples - batch_size + 1, batch_size)])
    return idxs


class MetaBaseParallelSampleIter(ABC):
    pass


class BaseParallelSampleIter(mx.io.DataIter):
    """
    Base parallel sample iterator.
    """
    __metaclass__ = MetaBaseParallelSampleIter

    def __init__(self,
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 source_data_name,
                 target_data_name,
                 label_name,
                 num_factors: int = 1,
                 dtype='float32') -> None:
        super().__init__(batch_size=batch_size)

        self.buckets = list(buckets)
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.bucket_batch_sizes = bucket_batch_sizes
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.label_name = label_name
        self.num_factors = num_factors
        self.dtype = dtype

        # "Staging area" that needs to fit any size batch we're using by total number of elements.
        # When computing per-bucket batch sizes, we guarantee that the default bucket will have the
        # largest total batch size.
        # Note: this guarantees memory sharing for input data and is generally a good heuristic for
        # other parts of the model, but it is possible that some architectures will have intermediate
        # operations that produce shapes larger than the default bucket size.  In these cases, MXNet
        # will silently allocate additional memory.
        self.provide_data = [
            mx.io.DataDesc(name=self.source_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[0],
                                  self.num_factors),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=self.target_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]
        self.provide_label = [
            mx.io.DataDesc(name=self.label_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]

        self.data_names = [self.source_data_name, self.target_data_name]
        self.label_names = [self.label_name]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def iter_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> mx.io.DataBatch:
        pass

    @abstractmethod
    def save_state(self, fname: str):
        pass

    @abstractmethod
    def load_state(self, fname: str):
        pass


class ShardedParallelSampleIter(BaseParallelSampleIter):
    """
    Goes through the data one shard at a time. The memory consumption is limited by the memory consumption of the
    largest shard. The order in which shards are traversed is changed with each reset.
    """

    def __init__(self,
                 shards_fnames: List[str],
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 fill_up: str,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 num_factors: int = 1,
                 dtype='float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         source_data_name=source_data_name, target_data_name=target_data_name,
                         label_name=label_name, num_factors=num_factors, dtype=dtype)
        assert len(shards_fnames) > 0
        self.shards_fnames = list(shards_fnames)
        self.shard_index = -1
        self.fill_up = fill_up

        self.reset()

    def _load_shard(self):
        shard_fname = self.shards_fnames[self.shard_index]
        logger.info("Loading shard %s.", shard_fname)
        dataset = ParallelDataSet.load(self.shards_fnames[self.shard_index]).fill_up(self.bucket_batch_sizes,
                                                                                     self.fill_up,
                                                                                     seed=self.shard_index)
        self.shard_iter = ParallelSampleIter(data=dataset,
                                             buckets=self.buckets,
                                             batch_size=self.batch_size,
                                             bucket_batch_sizes=self.bucket_batch_sizes,
                                             source_data_name=self.source_data_name,
                                             target_data_name=self.target_data_name,
                                             num_factors=self.num_factors)

    def reset(self):
        if len(self.shards_fnames) > 1:
            logger.info("Shuffling the shards.")
            # Making sure to not repeat a shard:
            if self.shard_index < 0:
                current_shard_fname = ""
            else:
                current_shard_fname = self.shards_fnames[self.shard_index]
            remaining_shards = [shard for shard in self.shards_fnames if shard != current_shard_fname]
            next_shard_fname = random.choice(remaining_shards)
            remaining_shards = [shard for shard in self.shards_fnames if shard != next_shard_fname]
            random.shuffle(remaining_shards)

            self.shards_fnames = [next_shard_fname] + remaining_shards

            self.shard_index = 0
            self._load_shard()
        else:
            if self.shard_index < 0:
                self.shard_index = 0
                self._load_shard()
            # We can just reset the shard_iter as we only have a single shard
            self.shard_iter.reset()

    def iter_next(self) -> bool:
        next_shard_index = self.shard_index + 1
        return self.shard_iter.iter_next() or next_shard_index < len(self.shards_fnames)

    def next(self) -> mx.io.DataBatch:
        if not self.shard_iter.iter_next():
            if self.shard_index < len(self.shards_fnames) - 1:
                self.shard_index += 1
                self._load_shard()
            else:
                raise StopIteration
        return self.shard_iter.next()

    def save_state(self, fname: str):
        with open(fname, "wb") as fp:
            pickle.dump(self.shards_fnames, fp)
            pickle.dump(self.shard_index, fp)
        self.shard_iter.save_state(fname + ".sharditer")

    def load_state(self, fname: str):
        with open(fname, "rb") as fp:
            self.shards_fnames = pickle.load(fp)
            self.shard_index = pickle.load(fp)
        self._load_shard()
        self.shard_iter.load_state(fname + ".sharditer")


class ParallelSampleIter(BaseParallelSampleIter):
    """
    Data iterator on a bucketed ParallelDataSet. Shuffles data at every reset and supports saving and loading the
    iterator state.
    """

    def __init__(self,
                 data: ParallelDataSet,
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 num_factors: int = 1,
                 dtype='float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         source_data_name=source_data_name, target_data_name=target_data_name,
                         label_name=label_name, num_factors=num_factors, dtype=dtype)

        # create independent lists to be shuffled
        self.data = ParallelDataSet(list(data.source), list(data.target), list(data.label))

        # create index tuples (buck_idx, batch_start_pos) into buckets. These will be shuffled.
        self.batch_indices = get_batch_indices(self.data, bucket_batch_sizes)
        self.curr_batch_index = 0

        self.inverse_data_permutations = [mx.nd.arange(0, max(1, self.data.source[i].shape[0]))
                                          for i in range(len(self.data))]
        self.data_permutations = [mx.nd.arange(0, max(1, self.data.source[i].shape[0]))
                                  for i in range(len(self.data))]

        self.reset()

    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_batch_index = 0
        # shuffle batch start indices
        random.shuffle(self.batch_indices)

        # restore
        self.data = self.data.permute(self.inverse_data_permutations)

        self.data_permutations, self.inverse_data_permutations = get_permutations(self.data.get_bucket_counts())

        self.data = self.data.permute(self.data_permutations)

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_batch_index != len(self.batch_indices)

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
        data = [source, target]
        label = [self.data.label[i][j:j + batch_size]]

        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        # TODO: num pad examples is not set here if fillup strategy would be padding
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    def save_state(self, fname: str):
        """
        Saves the current state of iterator to a file, so that iteration can be
        continued. Note that the data is not saved, i.e. the iterator must be
        initialized with the same parameters as in the first call.

        :param fname: File name to save the information to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self.batch_indices, fp)
            pickle.dump(self.curr_batch_index, fp)
            np.save(fp, [a.asnumpy() for a in self.inverse_data_permutations])
            np.save(fp, [a.asnumpy() for a in self.data_permutations])

    def load_state(self, fname: str):
        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """

        # restore order
        self.data = self.data.permute(self.inverse_data_permutations)

        with open(fname, "rb") as fp:
            self.batch_indices = pickle.load(fp)
            self.curr_batch_index = pickle.load(fp)
            inverse_data_permutations = np.load(fp)
            data_permutations = np.load(fp)

        # Right after loading the iterator state, next() should be called
        self.curr_batch_index -= 1

        # load previous permutations
        self.inverse_data_permutations = []
        self.data_permutations = []

        for bucket in range(len(self.data)):
            inverse_permutation = mx.nd.array(inverse_data_permutations[bucket])
            self.inverse_data_permutations.append(inverse_permutation)

            permutation = mx.nd.array(data_permutations[bucket])
            self.data_permutations.append(permutation)

        self.data = self.data.permute(self.data_permutations)

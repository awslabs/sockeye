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
from typing import Any, cast, Dict, Iterator, Iterable, List, Optional, Sequence, Sized, Tuple, Set

import mxnet as mx
import numpy as np

import multiprocessing

from . import config
from . import constants as C
from . import horovod_mpi
from . import vocab
from .utils import check_condition, smart_open, get_tokens, OnlineMeanAndVariance

logger = logging.getLogger(__name__)


def define_buckets(max_seq_len: int, step: int = 10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step until the final bucket fits max_seq_len.
    We then limit that bucket to max_seq_len (difference between semi-final and final bucket may be less than step).

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.

    :return: List of bucket sizes.
    """
    buckets = list(range(step, max_seq_len + step, step))
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            bucket_scaling: bool = True,
                            length_ratio: float = 1.0) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (max_seq_len_source, max_seq_len_target).  The longer side of the data uses
    steps of bucket_width while the shorter side uses steps scaled down by the average target/source length ratio.  If
    one side reaches its max_seq_len before the other, width of extra buckets on that side is fixed to that max_seq_len.

    :param max_seq_len_source: Maximum source bucket size.
    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    :param bucket_scaling: Scale bucket steps based on length ratio.
    :param length_ratio: Length ratio of data (target/source).
    """
    source_step_size = bucket_width
    target_step_size = bucket_width
    if bucket_scaling:
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


def define_empty_source_parallel_buckets(max_seq_len_target: int,
                                         bucket_width: int = 10) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (None, max_seq_len_target). The source
    is empty since it is supposed to not contain data that can be bucketized.
    The target is used as reference to create the buckets.

    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    """
    target_step_size = max(1, bucket_width)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # source buckets are always 0 since there is no text
    source_buckets = [0 for b in target_buckets]
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
    :param average_target_words_per_batch: Approximate number of target non-padding tokens in each batch.
    """

    def __init__(self, bucket: Tuple[int, int], batch_size: int, average_target_words_per_batch: float) -> None:
        self.bucket = bucket
        self.batch_size = batch_size
        self.average_target_words_per_batch = average_target_words_per_batch


def define_bucket_batch_sizes(buckets: List[Tuple[int, int]],
                              batch_size: int,
                              batch_type: str,
                              batch_num_devices: int,
                              data_target_average_len: List[Optional[float]],
                              batch_sentences_multiple_of: int = 1) -> List[BucketBatchSize]:
    """
    Compute bucket-specific batch sizes (sentences, average_target_words).

    If sentence batching: number of sentences is the same for each batch.

    If word batching: number of sentences for each batch is the multiple of
    number of devices that produces the number of words closest to the target
    batch size. Number of sentences is finally rounded to the nearest multiple
    of batch_sentences_multiple_of * batch_num_devices. Average target sentence
    length (non-padding symbols) is used for word number calculations.

    If max-word batching: number of sentences for each batch is set to the
    multiple of batch_sentences_multiple_of * batch_num_devices that is closest
    to batch_size without exceeding the value.

    :param buckets: Bucket list.
    :param batch_size: Batch size.
    :param batch_type: Type of batching.
    :param batch_num_devices: Number of devices.
    :param data_target_average_len: Optional average target length for each
        bucket.
    :param batch_sentences_multiple_of: Guarantee the number of sentences in
        each bucket's batch to a multiple of this value.
    """
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[BucketBatchSize]
    largest_total_num_words = 0
    # Ensure the correct multiple for each batch per device.
    min_batch_step = batch_sentences_multiple_of * batch_num_devices

    for buck_idx, bucket in enumerate(buckets):
        # Target/label length with padding
        padded_seq_len = bucket[1]
        # Average target/label length excluding padding
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len
        average_seq_len = data_target_average_len[buck_idx]

        # Batch size for each bucket is measured in sentences:
        # - word batching: convert average word-based size to number of
        #       sequences
        # - max-word batching: convert max word-based size to number of
        #       sequences
        # - sentence batching: use batch size directly
        if batch_type == C.BATCH_TYPE_WORD:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            # Multiple of minimum batch step closest to target number of words,
            # assuming each sentence is of average length
            batch_size_seq = min_batch_step * max(1, round((batch_size / average_seq_len) / min_batch_step))
        elif batch_type == C.BATCH_TYPE_MAX_WORD:
            check_condition(padded_seq_len <= batch_size,
                            'Word batch size must cover sequence lengths for all buckets: (%d > %d)'
                            % (padded_seq_len, batch_size))
            # Max number of sequences without exceeding batch size
            batch_size_seq = batch_size // padded_seq_len
            # Round down to closest multiple
            batch_size_seq = (batch_size_seq // min_batch_step) * min_batch_step
        elif batch_type == C.BATCH_TYPE_SENTENCE:
            batch_size_seq = batch_size
        else:
            raise ValueError('Unknown batch type: %s' % batch_type)
        # Number of words here is an average of non-padding tokens
        batch_size_word = batch_size_seq * average_seq_len

        bucket_batch_sizes.append(BucketBatchSize(bucket, batch_size_seq, batch_size_word))
        # Track largest number of source or target word samples in a batch
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * max(*bucket))

    # TODO: This is a legacy step from the bucketing module version of Sockeye.
    #       It is no longer necessary but is preserved to keep training behavior
    #       consistent.  Determine whether this can be safely removed.
    # Final step for average word-based batching: guarantee that largest bucket
    # by sequence length also has a shape that covers any (batch_size,
    # len_source) and (batch_size, len_target).
    if batch_type == C.BATCH_TYPE_WORD:
        padded_seq_len = max(*buckets[-1])
        average_seq_len = data_target_average_len[-1]
        while bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_num_words:
            bucket_batch_sizes[-1] = BucketBatchSize(
                bucket_batch_sizes[-1].bucket,
                bucket_batch_sizes[-1].batch_size + min_batch_step,
                bucket_batch_sizes[-1].average_target_words_per_batch + min_batch_step * average_seq_len)

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

    return LengthStatistics(mean_and_variance.count, mean_and_variance.mean, mean_and_variance.std)


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


def are_none(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences are None.
    """
    if not sequences:
        return True
    return all(s is None for s in sequences)


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
                 vocab_source: Optional[Dict[str, int]],
                 vocab_target: Dict[str, int],
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        if vocab_source is not None:
            self.unk_id_source = vocab_source[C.UNK_SYMBOL]
            self.size_vocab_source = len(vocab_source)
        else:
            self.unk_id_source = None
            self.size_vocab_source = 0
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
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
        self._length_ratio_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence_pair(self,
                      source: List[int],
                      target: List[int],
                      bucket_idx: Optional[int]):
        if bucket_idx is None:
            self.num_discarded += 1
            return

        source_len = len(source)
        target_len = len(target)
        length_ratio = target_len / (source_len if source_len else 1.)

        self._mean_len_target_per_bucket[bucket_idx].update(target_len)
        self._length_ratio_per_bucket[bucket_idx].update(length_ratio)

        self.num_sents += 1
        self.num_tokens_source += source_len
        self.num_tokens_target += target_len
        self.max_observed_len_source = max(source_len, self.max_observed_len_source)
        self.max_observed_len_target = max(target_len, self.max_observed_len_target)

        if self.unk_id_source is not None:
            self.num_unks_source += source.count(self.unk_id_source)
        self.num_unks_target += target.count(self.unk_id_target)

    @property
    def mean_len_target_per_bucket(self) -> List[Optional[float]]:
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None
                for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def length_ratio_stats_per_bucket(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(mean_and_variance.mean, mean_and_variance.std) if mean_and_variance.count > 0 else (None, None)
                for mean_and_variance in self._length_ratio_per_bucket]

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
                              mean_len_target_per_bucket=self.mean_len_target_per_bucket,
                              length_ratio_stats_per_bucket=self.length_ratio_stats_per_bucket)


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
    :param skip_blanks: Whether to skip blank lines.
    :param dtype: Data type.
    """

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 eos_id: int,
                 pad_id: int,
                 skip_blanks: bool = True,
                 dtype: str = 'float32') -> None:
        self.buckets = buckets
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.skip_blanks = skip_blanks
        self.dtype = dtype

    def load(self,
             source_iterables: Sequence[Iterable],
             target_iterable: Iterable,
             num_samples_per_bucket: List[int]) -> 'ParallelDataSet':

        assert len(num_samples_per_bucket) == len(self.buckets)
        num_factors = len(source_iterables)

        data_source = [np.full((num_samples, source_len, num_factors), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target = [np.full((num_samples, target_len + 1), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]

        bucket_sample_index = [0 for _ in self.buckets]

        # track amount of padding introduced through bucketing
        num_tokens_source = 0
        num_tokens_target = 0
        num_pad_source = 0
        num_pad_target = 0

        # Bucket sentences as padded np arrays
        for sentno, (sources, target) in enumerate(parallel_iter(source_iterables, target_iterable, skip_blanks=self.skip_blanks), 1):
            sources = [[] if stream is None else stream for stream in sources]
            if target is None:
                target = []
            source_len = len(sources[0])
            target_len = len(target)
            buck_index, buck = get_parallel_bucket(self.buckets, source_len, target_len)
            if buck is None:
                if self.skip_blanks:
                    continue  # skip this sentence pair
                else:
                    buck_index = len(self.buckets)
                    buck = self.buckets[buck_index]

            num_tokens_source += buck[0]
            num_tokens_target += buck[1]
            num_pad_source += buck[0] - source_len
            num_pad_target += buck[1] - target_len

            sample_index = bucket_sample_index[buck_index]
            for i, s in enumerate(sources):
                data_source[buck_index][sample_index, 0:source_len, i] = s
            data_target[buck_index][sample_index, :target_len + 1] = target + [self.eos_id]

            bucket_sample_index[buck_index] += 1

        for i in range(len(data_source)):
            # TODO(fhieber): Consider using pinned memory: mx.cpu_pinned() here
            data_source[i] = mx.nd.from_numpy(data_source[i], zero_copy=True)
            data_target[i] = mx.nd.from_numpy(data_target[i], zero_copy=True)

        if num_tokens_source > 0 and num_tokens_target > 0:
            logger.info("Created bucketed parallel data set. Introduced padding: source=%.1f%% target=%.1f%%)",
                        num_pad_source / num_tokens_source * 100,
                        num_pad_target / num_tokens_target * 100)

        return ParallelDataSet(data_source, data_target)


def get_num_shards(num_samples: int, samples_per_shard: int, min_num_shards: int) -> int:
    """
    Returns the number of shards.

    :param num_samples: Number of training data samples.
    :param samples_per_shard: Samples per shard.
    :param min_num_shards: Minimum number of shards.
    :return: Number of shards.
    """
    return max(int(math.ceil(num_samples / samples_per_shard)), min_num_shards)


def save_shard(shard_idx: int, data_loader: RawParallelDatasetLoader,
               shard_sources: List[str], shard_target: str,
               shard_stats: 'DataStatistics', output_prefix: str, keep_tmp_shard_files: bool):
    """
    Load shard source and target data files into NDArrays and save to disk.
    Optionally it can delete the source/target files.

    :param shard_idx: The index of the shard.
    :param data_loader: A loader for loading parallel data from sources and target.
    :param shard_sources: A list of sources file names.
    :param shard_target: A target file name.
    :param shard_stats: The statistics for the sources/target data.
    :param output_prefix: The prefix of the output file name.
    :param keep_tmp_shard_files: Keep the sources/target files when it is True otherwise delete them.
    """
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
                 bucket_scaling: bool = True,
                 keep_tmp_shard_files: bool = False,
                 max_processes: int = 1):
    logger.info("Preparing data.")
    # write vocabularies to data folder
    vocab.save_source_vocabs(source_vocabs, output_prefix)
    vocab.save_target_vocab(target_vocab, output_prefix)

    # Pass 1: get target/source length ratios.
    length_statistics = analyze_sequence_lengths(source_fnames, target_fname, source_vocabs, target_vocab,
                                                 max_seq_len_source, max_seq_len_target)

    check_condition(length_statistics.num_sents > 0,
                    "No training sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling,
                                      length_statistics.length_ratio_mean) if bucketing else [(max_seq_len_source,
                                                                                               max_seq_len_target)]
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
                                           eos_id=C.EOS_ID,
                                           pad_id=C.PAD_ID)

    # 3. convert each shard to serialized ndarrays
    if max_processes == 1:
        logger.info("Processing shards sequentially.")
        # Process shards sequantially woithout using multiprocessing
        for shard_idx, (shard_sources, shard_target, shard_stats) in enumerate(shards):
            save_shard(shard_idx, data_loader, shard_sources, shard_target,
                       shard_stats, output_prefix, keep_tmp_shard_files)
    else:
        logger.info("Processing shards using %s processes.", max_processes)
        # Process shards in parallel using max_processes process
        results = []
        pool = multiprocessing.pool.Pool(processes=max_processes)
        for shard_idx, (shard_sources, shard_target, shard_stats) in enumerate(shards):
            args = (shard_idx, data_loader, shard_sources, shard_target,
                    shard_stats, output_prefix, keep_tmp_shard_files)
            result = pool.apply_async(save_shard, args=args)
            results.append(result)
        pool.close()
        pool.join()

        for result in results:
            if not result.successful():
                logger.error("Process ended in error.")
                raise RuntimeError("Shard processing failed.")


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
                             num_source_factors=len(source_fnames))
    config_data_fname = os.path.join(output_prefix, C.DATA_CONFIG)
    logger.info("Writing data config to '%s'", config_data_fname)
    config_data.save(config_data_fname)

    version_file = os.path.join(output_prefix, C.PREPARED_DATA_VERSION_FILE)

    with open(version_file, "w") as version_out:
        version_out.write(str(C.PREPARED_DATA_VERSION))


def get_data_statistics(source_readers: Optional[Sequence[Iterable]],
                        target_reader: Iterable,
                        buckets: List[Tuple[int, int]],
                        length_ratio_mean: float,
                        length_ratio_std: float,
                        source_vocabs: Optional[List[vocab.Vocab]],
                        target_vocab: vocab.Vocab) -> 'DataStatistics':
    data_stats_accumulator = DataStatisticsAccumulator(buckets,
                                                       source_vocabs[0] if source_vocabs is not None else None,
                                                       target_vocab,
                                                       length_ratio_mean,
                                                       length_ratio_std)

    if source_readers is not None:
        for sources, target in parallel_iter(source_readers, target_reader):
            buck_idx, buck = get_parallel_bucket(buckets, len(sources[0]), len(target))
            data_stats_accumulator.sequence_pair(sources[0], target, buck_idx)
    else:  # Allow stats for target only data
        for target in target_reader:
            buck_idx, buck = get_target_bucket(buckets, len(target))
            data_stats_accumulator.sequence_pair([], target, buck_idx)

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
                             batch_size: int) -> 'ParallelSampleIter':
    """
    Returns a ParallelSampleIter for the validation data.
    """
    logger.info("=================================")
    logger.info("Creating validation data iterator")
    logger.info("=================================")
    validation_length_statistics = analyze_sequence_lengths(validation_sources, validation_target,
                                                            source_vocabs, target_vocab,
                                                            max_seq_len_source, max_seq_len_target)

    check_condition(validation_length_statistics.num_sents > 0,
                    "No validation sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

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
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)

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
                            batch_type: str,
                            batch_num_devices: int,
                            batch_sentences_multiple_of: int = 1,
                            permute: bool = True) -> Tuple['BaseParallelSampleIter',
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

    check_condition(shared_vocab == data_info.shared_vocab, "Shared vocabulary settings need to match these "
                                                            "of the prepared data (e.g. for weight tying). "
                                                            "Specify or omit %s consistently when training "
                                                            "and preparing the data." % C.VOCAB_ARG_SHARED_VOCAB)

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
                                                   batch_type,
                                                   batch_num_devices,
                                                   config_data.data_statistics.average_len_target_per_bucket,
                                                   batch_sentences_multiple_of)

    config_data.data_statistics.log(bucket_batch_sizes)

    train_iter = ShardedParallelSampleIter(shard_fnames,
                                           buckets,
                                           batch_size,
                                           bucket_batch_sizes,
                                           num_factors=len(data_info.sources),
                                           permute=permute)

    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=C.EOS_ID,
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
                                               batch_size=batch_size)

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
                            batch_type: str,
                            batch_num_devices: int,
                            max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int,
                            bucket_scaling: bool = True,
                            allow_empty: bool = False,
                            batch_sentences_multiple_of: int = 1) -> Tuple['BaseParallelSampleIter',
                                                                                Optional['BaseParallelSampleIter'],
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
    :param batch_type: Method for sizing batches.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :param bucket_scaling: Scale bucket steps based on source/target length ratio.
    :param allow_empty: Unless True if no sentences are below or equal to the maximum length an exception is raised.
    :param batch_sentences_multiple_of: Round the number of sentences in each
        bucket's batch to a multiple of this value (word-based batching only).

    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")
    # Pass 1: get target/source length ratios.
    length_statistics = analyze_sequence_lengths(sources, target, source_vocabs, target_vocab,
                                                 max_seq_len_source, max_seq_len_target)

    if not allow_empty:
        check_condition(length_statistics.num_sents > 0,
                        "No training sequences found with length smaller or equal than the maximum sequence length."
                        "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling,
                                      length_statistics.length_ratio_mean) if bucketing else [(max_seq_len_source,
                                                                                               max_seq_len_target)]

    sources_sentences, target_sentences = create_sequence_readers(sources, target, source_vocabs, target_vocab)

    # Pass 2: Get data statistics and determine the number of data points for each bucket.
    data_statistics = get_data_statistics(sources_sentences, target_sentences, buckets,
                                          length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                          source_vocabs, target_vocab)

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_type,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket,
                                                   batch_sentences_multiple_of)

    data_statistics.log(bucket_batch_sizes)

    # Pass 3: Load the data into memory and return the iterator.
    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=C.EOS_ID,
                                           pad_id=C.PAD_ID)

    training_data = data_loader.load(sources_sentences, target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)

    data_info = DataInfo(sources=sources,
                         target=target,
                         source_vocabs=source_vocab_paths,
                         target_vocab=target_vocab_path,
                         shared_vocab=shared_vocab,
                         num_shards=1)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(sources))

    train_iter = ParallelSampleIter(data=training_data,
                                    buckets=buckets,
                                    batch_size=batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_factors=len(sources),
                                    permute=True)

    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=target_vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size)

    return train_iter, validation_iter, config_data, data_info


def get_scoring_data_iters(sources: List[str],
                           target: str,
                           source_vocabs: List[vocab.Vocab],
                           target_vocab: vocab.Vocab,
                           batch_size: int,
                           max_seq_len_source: int,
                           max_seq_len_target: int) -> 'BaseParallelSampleIter':
    """
    Returns a data iterator for scoring. The iterator loads data on demand,
    batch by batch, and does not skip any lines. Lines that are too long
    are truncated.

    :param sources: Path to source training data (with optional factor data paths).
    :param target: Path to target training data.
    :param source_vocabs: Source vocabulary and optional factor vocabularies.
    :param target_vocab: Target vocabulary.
    :param batch_size: Batch size.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :return: The scoring data iterator.
    """
    logger.info("==============================")
    logger.info("Creating scoring data iterator")
    logger.info("==============================")

    # One bucket to hold them all,
    bucket = (max_seq_len_source, max_seq_len_target)

    # ...One loader to raise them,
    data_loader = RawParallelDatasetLoader(buckets=[bucket],
                                           eos_id=C.EOS_ID,
                                           pad_id=C.PAD_ID,
                                           skip_blanks=False)

    # ...one iterator to traverse them all,
    scoring_iter = BatchedRawParallelSampleIter(data_loader=data_loader,
                                                sources=sources,
                                                target=target,
                                                source_vocabs=source_vocabs,
                                                target_vocab=target_vocab,
                                                bucket=bucket,
                                                batch_size=batch_size,
                                                max_lens=(max_seq_len_source, max_seq_len_target),
                                                num_factors=len(sources))

    # and with the model appraise them.
    return scoring_iter


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
                 mean_len_target_per_bucket: List[Optional[float]],
                 length_ratio_stats_per_bucket: Optional[List[Tuple[Optional[float], Optional[float]]]] = None) -> None:
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
        self.length_ratio_stats_per_bucket = length_ratio_stats_per_bucket
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
    for bucket_batch_size, num_seq, (lr_mean, lr_std) in zip(bucket_batch_sizes,
                                                             data_statistics.num_sents_per_bucket,
                                                             data_statistics.length_ratio_stats_per_bucket):
        if num_seq > 0:
            logger.info("Bucket %s: %d samples in %d batches of %d, ~%.1f target tokens/batch, "
                        "trg/src length ratio: %.2f (+-%.2f)",
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.average_target_words_per_batch,
                        lr_mean, lr_std)


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
                 num_source_factors: int) -> None:
        super().__init__()
        self.data_statistics = data_statistics
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.num_source_factors = num_source_factors


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
    return C.TOKEN_SEPARATOR.join(map(str, ids))


def ids2tokens(token_ids: Iterable[int],
               vocab_inv: Dict[int, str],
               exclude_set: Set[int]) -> Iterator[str]:
    """
    Transforms a list of token IDs into a list of words, excluding any IDs in `exclude_set`.

    :param token_ids: The list of token IDs.
    :param vocab_inv: The inverse vocabulary.
    :param exclude_set: The list of token IDs to exclude.
    :return: The list of words.
    """
    tokens = (vocab_inv[token] for token in token_ids)
    return (tok for token_id, tok in zip(token_ids, tokens) if token_id not in exclude_set)


class SequenceReader:
    """
    Reads sequence samples from path and (optionally) creates integer id sequences.
    Streams from disk, instead of loading all samples into memory.
    If vocab is None, the sequences in path are assumed to be integers coded as strings.
    Empty sequences are yielded as None.

    :param path: Path to read data from.
    :param vocabulary: Optional mapping from strings to integer ids.
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
            assert vocab.is_valid_vocab(vocabulary)
            self.bos_id = C.BOS_ID
            self.eos_id = C.EOS_ID
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


def parallel_iter(source_iterables: Sequence[Iterable[Optional[Any]]],
                  target_iterable: Iterable[Optional[Any]],
                  skip_blanks: bool = True):
    """
    Creates iterators over parallel iteratables by calling iter() on the iterables
    and chaining to parallel_iterate(). The purpose of the separation is to allow
    the caller to save iterator state between calls, if desired.

    :param source_iterables: A list of source iterables.
    :param target_iterable: A target iterable.
    :param skip_blanks: Whether to skip empty target lines.
    :return: Iterators over sources and target.
    """
    source_iterators = [iter(s) for s in source_iterables]
    target_iterator = iter(target_iterable)
    return parallel_iterate(source_iterators, target_iterator, skip_blanks)


def parallel_iterate(source_iterators: Sequence[Iterator[Optional[Any]]],
                     target_iterator: Iterator[Optional[Any]],
                     skip_blanks: bool = True):
    """
    Yields parallel source(s), target sequences from iterables.
    Checks for token parallelism in source sequences.
    Skips pairs where element in at least one iterable is None.
    Checks that all iterables have the same number of elements.
    Can optionally continue from an already-begun iterator.

    :param source_iterators: A list of source iterators.
    :param target_iterator: A target iterator.
    :param skip_blanks: Whether to skip empty target lines.
    :return: Iterators over sources and target.
    """
    num_skipped = 0
    while True:
        try:
            sources = [next(source_iter) for source_iter in source_iterators]
            target = next(target_iterator)
        except StopIteration:
            break
        if skip_blanks and (any((s is None for s in sources)) or target is None):
            num_skipped += 1
            continue
        check_condition(are_none(sources) or are_token_parallel(sources), "Source sequences are not token-parallel: %s" % (str(sources)))
        yield sources, target

    if num_skipped > 0:
        logger.warning("Parallel reading of sequences skipped %d elements", num_skipped)

    check_condition(
        all(next(cast(Iterator, s), None) is None for s in source_iterators) and next(cast(Iterator, target_iterator),
                                                                                      None) is None,
        "Different number of lines in source(s) and target iterables.")


class FileListReader:
    """
    Reads sequence samples from path provided in a file.

    :param fname: File name containing a list of relative paths.
    :param path: Path to read data from, which is prefixed to the relative paths of fname.
    """

    def __init__(self,
                 fname: str,
                 path: str) -> None:
        self.fname = fname
        self.path = path
        self.fd = smart_open(fname)
        self.count = 0

    def __next__(self):
        fname = self.fd.readline().strip("\n")

        if fname is None:
            self.fd.close()
            raise StopIteration

        self.count += 1
        return os.path.join(self.path, fname)


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
    Algorithm assumes buckets are sorted from shortest to longest.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets, in sorted order, shortest to longest.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            return j, (source_bkt, target_bkt)
    return None, None


def get_target_bucket(buckets: List[Tuple[int, int]],
                      length_target: int) -> Optional[Tuple[int, Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket = None, None  # type: Tuple[int, Tuple[int, int]]
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if target_bkt >= length_target:
            bucket = j, (source_bkt, target_bkt)
            break
    return bucket


class ParallelDataSet:
    """
    Bucketed parallel data set
    """

    def __init__(self,
                 source: List[mx.nd.array],
                 target: List[mx.nd.array]) -> None:
        check_condition(len(source) == len(target),
                        "Number of buckets for source/target do not match: %d/%d." % (len(source), len(target)))
        self.source = source
        self.target = target

    def __len__(self) -> int:
        return len(self.source)

    def get_bucket_counts(self):
        return [len(self.source[buck_idx]) for buck_idx in range(len(self))]

    def save(self, fname: str):
        """
        Saves the dataset to a binary .npy file.
        """
        mx.nd.save(fname, self.source + self.target)

    @staticmethod
    def load(fname: str) -> 'ParallelDataSet':
        """
        Loads a dataset from a binary .npy file.  When running Horovod, the data
        is sliced and each worker loads a different slice based on its rank.
        """
        data = mx.nd.load(fname)
        n = len(data) // 2
        source = data[:n]
        target = data[n:2 * n]
        if horovod_mpi.using_horovod() and horovod_mpi.hvd.size() > 1:
            split_index = horovod_mpi.hvd.rank()
            total_splits = horovod_mpi.hvd.size()
            i = split_index / total_splits
            j = (split_index + 1) / total_splits
            # Load this worker's slice of each bucket.  If the bucket is empty,
            # there is no need to slice and attempting to do so will raise an
            # error.
            source = [s[math.floor(i * s.shape[0]):math.floor(j * s.shape[0])]
                      if s.shape[0] > 0
                      else s for s in source]
            target = [t[math.floor(i * t.shape[0]):math.floor(j * t.shape[0])]
                      if t.shape[0] > 0
                      else t for t in target]
        assert len(source) == len(target)
        return ParallelDataSet(source, target)

    def fill_up(self,
                bucket_batch_sizes: List[BucketBatchSize],
                seed: int = 42) -> 'ParallelDataSet':
        """
        Returns a new dataset with buckets filled up.

        :param bucket_batch_sizes: Bucket batch sizes.
        :param seed: The random seed used for sampling sentences to fill up.
        :return: New dataset with buckets filled up to the next multiple of batch size
        """
        source = list(self.source)
        target = list(self.target)

        rs = np.random.RandomState(seed)

        for bucket_idx in range(len(self)):
            bucket_batch_size = bucket_batch_sizes[bucket_idx].batch_size
            bucket_source = self.source[bucket_idx]
            bucket_target = self.target[bucket_idx]
            num_samples = bucket_source.shape[0]

            # Determine the target number of samples (current value or minimally
            # higher value that meets the batch size requirement).
            target_num_samples = num_samples
            if num_samples % bucket_batch_size != 0:
                target_num_samples = num_samples + (bucket_batch_size - (num_samples % bucket_batch_size))

            if horovod_mpi.using_horovod():
                # Workers load different slices of the data.  When the total
                # number of samples is not evenly divisible by the number of
                # workers, each worker may have +/- 1 sample.  Use the largest
                # target number of samples across all workers to keep the number
                # of batches in sync and guarantee that all samples are used.
                target_num_samples = max(horovod_mpi.MPI.COMM_WORLD.allgather(target_num_samples))

            # Fill up the last batch by randomly sampling from the extant items.
            rest = target_num_samples - num_samples
            if rest > 0:
                desired_indices_np = rs.randint(num_samples, size=rest)
                desired_indices = mx.nd.from_numpy(desired_indices_np, zero_copy=True)
                source[bucket_idx] = mx.nd.concat(bucket_source, bucket_source.take(desired_indices), dim=0)
                target[bucket_idx] = mx.nd.concat(bucket_target, bucket_target.take(desired_indices), dim=0)

        return ParallelDataSet(source, target)

    def permute(self, permutations: List[mx.nd.NDArray]) -> 'ParallelDataSet':
        """
        Permutes the data within each bucket. The permutation is received as an argument,
        allowing the data to be unpermuted (i.e., restored) later on.

        :param permutations: For each bucket, a permutation of the data within that bucket.
        :return: A new, permuted ParallelDataSet.
        """
        assert len(self) == len(permutations)
        source = []
        target = []
        for buck_idx in range(len(self)):
            num_samples = self.source[buck_idx].shape[0]
            if num_samples:  # not empty bucket
                permutation = permutations[buck_idx]
                if isinstance(self.source[buck_idx], np.ndarray):
                    source.append(self.source[buck_idx].take(np.int64(permutation.asnumpy())))
                else:
                    source.append(self.source[buck_idx].take(permutation))
                target.append(self.target[buck_idx].take(permutation))
            else:
                source.append(self.source[buck_idx])
                target.append(self.target[buck_idx])

        return ParallelDataSet(source, target)


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
        inverse_data_permutation = mx.nd.from_numpy(inverse_data_permutation, zero_copy=True)
        data_permutation = mx.nd.from_numpy(data_permutation, zero_copy=True)

        data_permutations.append(data_permutation)
        inverse_data_permutations.append(inverse_data_permutation)
    return data_permutations, inverse_data_permutations


def get_batch_indices(data: ParallelDataSet,
                      bucket_batch_sizes: List[BucketBatchSize]) -> List[Tuple[int, int]]:
    """
    Returns a list of index tuples that index into the bucket and the start index inside a bucket given
    the batch size for a bucket. These indices are valid for the given dataset.

    Put another way, this returns the starting points for all batches within the dataset, across all buckets.

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

    :param buckets: The list of buckets.
    :param bucket_batch_sizes: A list, parallel to `buckets`, containing the number of samples in each bucket.
    :param num_factors: The number of source factors.
    :param permute: Randomly shuffle the parallel data.
    :param dtype: The MXNet data type.
    """
    __metaclass__ = MetaBaseParallelSampleIter

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 batch_size: int,
                 bucket_batch_sizes: List[BucketBatchSize],
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype='float32') -> None:
        super().__init__(batch_size=batch_size)

        self.buckets = list(buckets)
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.bucket_batch_sizes = bucket_batch_sizes
        self.num_factors = num_factors
        self.permute = permute
        self.dtype = dtype

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def iter_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> 'Batch':
        pass

    @abstractmethod
    def save_state(self, fname: str):
        pass

    @abstractmethod
    def load_state(self, fname: str):
        pass


class BatchedRawParallelSampleIter(BaseParallelSampleIter):
    """
    Goes through the raw data, loading only one batch at a time into memory.
    Used by the scorer. Iterates through the data in order, and therefore does
    not support bucketing.
    """

    def __init__(self,
                 data_loader: RawParallelDatasetLoader,
                 sources: List[str],
                 target: str,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 bucket: Tuple[int, int],
                 batch_size: int,
                 max_lens: Tuple[int, int],
                 num_factors: int = 1,
                 dtype='float32') -> None:
        super().__init__(buckets=[bucket],
                         batch_size=batch_size,
                         bucket_batch_sizes=[BucketBatchSize(bucket, batch_size, None)],
                         num_factors=num_factors,
                         permute=False,
                         dtype=dtype)
        self.data_loader = data_loader
        self.sources_sentences, self.target_sentences = create_sequence_readers(sources, target,
                                                                                source_vocabs, target_vocab)
        self.sources_iters = [iter(s) for s in self.sources_sentences]
        self.target_iter = iter(self.target_sentences)
        self.max_len_source, self.max_len_target = max_lens
        self.next_batch = None  # type: Optional[Batch]
        self.sentno = 1

    def reset(self):
        raise Exception('Not supported!')

    def iter_next(self) -> bool:
        """
        True if the iterator can return another batch.
        """

        # Read batch_size lines from the source stream
        sources_sentences = [[] for x in self.sources_sentences]  # type: List[List[str]]
        target_sentences = []  # type: List[str]
        num_read = 0
        for num_read, (sources, target) in enumerate(
                parallel_iterate(self.sources_iters, self.target_iter, skip_blanks=False), 1):
            source_len = 0 if sources[0] is None else len(sources[0])
            target_len = 0 if target is None else len(target)
            if source_len > self.max_len_source:
                logger.info("Trimming source sentence {} ({} -> {})".format(self.sentno + num_read,
                                                                            source_len,
                                                                            self.max_len_source))
                sources = [source[0: self.max_len_source] for source in sources]
            if target_len > self.max_len_target:
                logger.info("Trimming target sentence {} ({} -> {})".format(self.sentno + num_read,
                                                                            target_len,
                                                                            self.max_len_target))
                target = target[0: self.max_len_target]

            for i, source in enumerate(sources):
                sources_sentences[i].append(source)
            target_sentences.append(target)
            if num_read == self.batch_size:
                break

        self.sentno += num_read

        if num_read == 0:
            self.next_batch = None
            return False

        dataset = self.data_loader.load(sources_sentences, target_sentences, [num_read])

        source = dataset.source[0]
        target, label = create_target_and_shifted_label_sequences(dataset.target[0])
        self.next_batch = create_batch_from_parallel_sample(source, target, label)
        return True

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch.
        """
        if self.iter_next():
            return self.next_batch
        raise StopIteration

    def save_state(self, fname: str):
        raise NotImplementedError('Not supported!')

    def load_state(self, fname: str):
        raise NotImplementedError('Not supported!')


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
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype = 'float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         num_factors=num_factors, permute=permute, dtype=dtype)
        assert len(shards_fnames) > 0
        self.shards_fnames = list(shards_fnames)
        self.shard_index = -1

        self.reset()

    def _load_shard(self):
        shard_fname = self.shards_fnames[self.shard_index]
        logger.info("Loading shard %s.", shard_fname)
        dataset = ParallelDataSet.load(self.shards_fnames[self.shard_index]).fill_up(self.bucket_batch_sizes,
                                                                                     seed=self.shard_index)
        self.shard_iter = ParallelSampleIter(data=dataset,
                                             buckets=self.buckets,
                                             batch_size=self.batch_size,
                                             bucket_batch_sizes=self.bucket_batch_sizes,
                                             num_factors=self.num_factors,
                                             permute=self.permute)

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

            if horovod_mpi.using_horovod():
                # Synchronize shard order across workers
                self.shards_fnames = horovod_mpi.MPI.COMM_WORLD.bcast(self.shards_fnames, root=0)

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

    def next(self) -> 'Batch':
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
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype='float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         num_factors=num_factors, permute=permute, dtype=dtype)

        # create independent lists to be shuffled
        self.data = ParallelDataSet(list(data.source), list(data.target))

        # create index tuples (buck_idx, batch_start_pos) into buckets.
        # This is the list of all batches across all buckets in the dataset. These will be shuffled.
        self.batch_indices = get_batch_indices(self.data, bucket_batch_sizes)
        self.curr_batch_index = 0

        # Produces a permutation of the batches within each bucket, along with the permutation that inverts it.
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
        if self.permute:
            # Primary worker or not using Horovod: shuffle batch start indices.
            if not horovod_mpi.using_horovod() or horovod_mpi.hvd.rank() == 0:
                random.shuffle(self.batch_indices)
            if horovod_mpi.using_horovod():
                # Synchronize order across workers.  This guarantees that each
                # worker processes a batch from the same bucket at each step.
                self.batch_indices = horovod_mpi.MPI.COMM_WORLD.bcast(self.batch_indices, root=0)

            # restore the data permutation
            self.data = self.data.permute(self.inverse_data_permutations)

            # permute the data within each batch
            self.data_permutations, self.inverse_data_permutations = get_permutations(self.data.get_bucket_counts())
            self.data = self.data.permute(self.data_permutations)

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_batch_index != len(self.batch_indices)

    def next(self) -> 'Batch':
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1

        batch_size = self.bucket_batch_sizes[i].batch_size
        source = self.data.source[i][j:j + batch_size]
        target, label = create_target_and_shifted_label_sequences(self.data.target[i][j:j + batch_size])
        return create_batch_from_parallel_sample(source, target, label)

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
            np.save(fp, [a.asnumpy() for a in self.inverse_data_permutations], allow_pickle=True)
            np.save(fp, [a.asnumpy() for a in self.data_permutations], allow_pickle=True)

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
            inverse_data_permutations = np.load(fp, allow_pickle=True)
            data_permutations = np.load(fp, allow_pickle=True)

        # Right after loading the iterator state, next() should be called
        self.curr_batch_index -= 1

        # load previous permutations
        self.inverse_data_permutations = []
        self.data_permutations = []

        for bucket in range(len(self.data)):
            inverse_permutation = mx.nd.from_numpy(inverse_data_permutations[bucket], zero_copy=True)
            self.inverse_data_permutations.append(inverse_permutation)

            permutation = mx.nd.from_numpy(data_permutations[bucket], zero_copy=True)
            self.data_permutations.append(permutation)

        self.data = self.data.permute(self.data_permutations)


class Batch:

    __slots__ = ['source', 'source_length', 'target', 'target_length', 'labels', 'samples', 'tokens']

    def __init__(self, source, source_length, target, target_length, labels, samples, tokens):
        self.source = source
        self.source_length = source_length
        self.target = target
        self.target_length = target_length
        self.labels = labels
        self.samples = samples
        self.tokens = tokens

    def split_and_load(self, ctx: List[mx.context.Context]) -> 'Batch':
        source = mx.gluon.utils.split_and_load(self.source, ctx, batch_axis=0)
        source_length = mx.gluon.utils.split_and_load(self.source_length, ctx, batch_axis=0)
        target = mx.gluon.utils.split_and_load(self.target, ctx, batch_axis=0)
        target_length = mx.gluon.utils.split_and_load(self.target_length, ctx, batch_axis=0)
        labels = {name: mx.gluon.utils.split_and_load(label, ctx, batch_axis=0) for name, label in self.labels.items()}
        return Batch(source, source_length, target, target_length, labels, self.samples, self.tokens)

    def shards(self) -> Iterable[Tuple[Tuple, Dict[str, mx.nd.NDArray]]]:
        assert isinstance(self.source, list), "Must call split_and_load() first"
        for i, inputs in enumerate(zip(self.source, self.source_length, self.target, self.target_length)):
            # model inputs, labels
            yield inputs, {name: label[i] for name, label in self.labels.items()}


def create_target_and_shifted_label_sequences(target_and_label: mx.nd.NDArray) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
    """
    Returns the target and label sequence from a joint array of varying-length sequences including both <bos> and <eos>.
    Both ndarrays returned have input size of second dimension - 1.
    """
    target = target_and_label[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
    target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
    label = target_and_label[:, 1:]  # label skips <bos>
    return target, label


def create_batch_from_parallel_sample(source: mx.nd.NDArray, target: mx.nd.NDArray, label: mx.nd.NDArray) -> Batch:
    """
    Creates a Batch instance from parallel data.
    """
    source_words = mx.nd.squeeze(mx.nd.slice(source, begin=(None, None, 0), end=(None, None, 1)), axis=2)
    source_length = mx.nd.sum(source_words != C.PAD_ID, axis=1)
    target_length = mx.nd.sum(target != C.PAD_ID, axis=1)
    length_ratio = source_length / target_length

    samples = source.shape[0]
    tokens = source.shape[1] * samples

    labels = {C.TARGET_LABEL_NAME: label, C.LENRATIO_LABEL_NAME: length_ratio}

    return Batch(source, source_length, target, target_length, labels, samples, tokens)

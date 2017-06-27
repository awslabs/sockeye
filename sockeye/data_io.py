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

"""
Implements data iterators and I/O related functions for sequence-to-sequence models.
"""
import bisect
import gzip
import logging
import pickle
import random
from typing import Dict, Iterator, Iterable, List, NamedTuple, Optional, Tuple

import mxnet as mx
import numpy as np

import sockeye.constants as C

logger = logging.getLogger(__name__)


def define_buckets(max_seq_len: int, step=10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step making sure that max_seq_len is covered by a bucket.
    This entails that generation of the next-largest bucket that includes max_seq_len.

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.
    :return: List of bucket sizes.
    """
    return [bucket_len for bucket_len in range(step, max_seq_len + step, step)]


def define_parallel_buckets(max_seq_len: int, bucket_width=10, length_ratio=1.0) -> List[Tuple[int, int]]:
    """
    Returns (src,trg) buckets in steps of bucket_width. Minimum bucket size for both source and target is 2.
    If length_ratio >=1, then we make sure that a target sentence of max_seq_len will be covered by a bucket.
    Otherwise, we make sure that a source sentence of max_seq_len will be covered by a bucket.

    :param max_seq_len: Maximum bucket size.
    :param bucket_width: Width of buckets.
    :param length_ratio: Length ratio between source and target data.
    """
    if length_ratio >= 1.0:
        # target side is longer, hence defines number of buckets
        target_buckets = define_buckets(max_seq_len, step=bucket_width)
        source_step_size = max(1, int(bucket_width / length_ratio))
        source_buckets = define_buckets(len(target_buckets) * source_step_size, step=source_step_size)
    else:
        # source side is longer, hence defines number of buckets
        source_buckets = define_buckets(max_seq_len, step=bucket_width)
        target_step_size = max(1, int(bucket_width * length_ratio))
        target_buckets = define_buckets(len(source_buckets) * target_step_size, step=target_step_size)

    # minimum bucket size is 2 (as we add BOS symbol to target side)
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]

    return list(zip(source_buckets, target_buckets))


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


def read_parallel_corpus(data_source: str,
                         data_target: str,
                         vocab_source: Dict[str, int],
                         vocab_target: Dict[str, int]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Loads source and target data, making sure they have the same length.

    :param data_source: Path to source training data.
    :param data_target: Path to target training data.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :return: Tuple of (source sentences, target sentences).
    """
    source_sentences = read_sentences(data_source, vocab_source, add_bos=False)
    target_sentences = read_sentences(data_target, vocab_target, add_bos=True)
    assert len(source_sentences) == len(
        target_sentences), "Number of source sentences does not match number of target sentences"
    return source_sentences, target_sentences


def get_training_data_iters(source: str, target: str,
                            validation_source: str, validation_target: str,
                            vocab_source: Dict[str, int], vocab_target: Dict[str, int],
                            batch_size: int,
                            fill_up: str,
                            max_seq_len: int,
                            bucketing: bool,
                            bucket_width: int) -> Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter']:
    """
    Returns data iterators for training and validation data.

    :param source: Path to source training data.
    :param target: Path to target training data.
    :param validation_source: Path to source validation data.
    :param validation_target: Path to target validation data.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param batch_size: Batch size.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len: Maximum sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: Tuple of (training data iterator, validation data iterator).
    """
    logger.info("Creating train data iterator")
    train_source_sentences, train_target_sentences = read_parallel_corpus(source,
                                                                          target,
                                                                          vocab_source,
                                                                          vocab_target)
    length_ratio = sum(len(t) / float(len(s)) for t, s in zip(train_source_sentences, train_target_sentences)) / len(
        train_target_sentences)
    logger.info("Average training target/source length ratio: %.2f", length_ratio)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len, bucket_width, length_ratio) if bucketing else [
        (max_seq_len, max_seq_len)]

    train_iter = ParallelBucketSentenceIter(train_source_sentences,
                                            train_target_sentences,
                                            buckets,
                                            batch_size,
                                            vocab_target[C.EOS_SYMBOL],
                                            C.PAD_ID,
                                            vocab_target[C.UNK_SYMBOL],
                                            fill_up=fill_up)

    logger.info("Creating validation data iterator")
    val_source_sentences, val_target_sentences = read_parallel_corpus(validation_source,
                                                                      validation_target,
                                                                      vocab_source,
                                                                      vocab_target)
    val_iter = ParallelBucketSentenceIter(val_source_sentences,
                                          val_target_sentences,
                                          buckets,
                                          batch_size,
                                          vocab_target[C.EOS_SYMBOL],
                                          C.PAD_ID,
                                          vocab_target[C.UNK_SYMBOL],
                                          fill_up=fill_up)
    return train_iter, val_iter


DataInfo = NamedTuple('DataInfo', [
    ('source', str),
    ('target', str),
    ('validation_source', str),
    ('validation_target', str),
    ('vocab_source', str),
    ('vocab_target', str),
])
"""
Tuple to collect data information for training.

:param source: Path to training source.
:param target: Path to training target.
:param validation_source: Path to validation source.
:param validation_target: Path to validation target.
:param vocab_source: Path to source vocabulary.
:param vocab_target: Path to target vocabulary.
"""


def smart_open(filename: str, mode="rt", ftype="auto", errors='replace'):
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open
    :param errors: Encoding error handling during reading. Defaults to 'replace'
    :return: File descriptor
    """
    if ftype == 'gzip' or ftype == 'gz' or (ftype == 'auto' and filename.endswith(".gz")):
        return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    else:
        return open(filename, mode=mode, encoding='utf-8', errors=errors)


def read_content(path: str, limit=None) -> Iterator[List[str]]:
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


def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token


def tokens2ids(tokens: Iterable[str], vocab: Dict[str, int]) -> List[int]:
    """
    Returns sequence of ids given a sequence of tokens and vocab.
    
    :param tokens: List of tokens.
    :param vocab: Vocabulary (containing UNK symbol).
    :return: List of word ids.
    """
    return [vocab.get(w, vocab[C.UNK_SYMBOL]) for w in tokens]


def read_sentences(path: str, vocab: Dict[str, int], add_bos=False, limit=None) -> List[List[int]]:
    """
    Reads sentences from path and creates word id sentences.

    :param path: Path to read data from.
    :param vocab: Vocabulary mapping.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    :return: List of integer sequences.
    """
    assert C.UNK_SYMBOL in vocab
    assert C.UNK_SYMBOL in vocab
    assert vocab[C.PAD_SYMBOL] == C.PAD_ID
    assert C.BOS_SYMBOL in vocab
    assert C.EOS_SYMBOL in vocab
    sentences = []
    for sentence_tokens in read_content(path, limit):
        sentence = tokens2ids(sentence_tokens, vocab)
        assert len(sentence) > 0, "Empty sentence in file %s" % path
        if add_bos:
            sentence.insert(0, vocab[C.BOS_SYMBOL])
        sentences.append(sentence)
    logger.info("%d sentences loaded from '%s'", len(sentences), path)
    return sentences


def get_default_bucket_key(buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)


def get_parallel_bucket(buckets: List[Tuple[int, int]],
                        length_source: int,
                        length_target: int) -> Optional[Tuple[int, Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket = None, None
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            bucket = j, (source_bkt, target_bkt)
            break
    return bucket


# TODO: consider more memory-efficient data reading (load from disk on demand)
# TODO: consider using HDF5 format for language data
class ParallelBucketSentenceIter(mx.io.DataIter):
    """
    A Bucket sentence iterator for parallel data. Randomly shuffles the data after every call to reset().
    Data is stored in NDArrays for each epoch for fast indexing during iteration.

    :param source_sentences: List of source sentences (integer-coded).
    :param target_sentences: List of target sentences (integer-coded).
    :param buckets: List of buckets.
    :param batch_size: Batch_size of generated data batches.
           Incomplete batches are discarded if fill_up == None, or filled up according to the fill_up strategy.
    :param fill_up: If not None, fill up bucket data to a multiple of batch_size to avoid discarding incomplete batches.
           for each bucket. If set to 'replicate', sample examples from the bucket and use them to fill up.
    :param eos_id: Word id for end-of-sentence.
    :param pad_id: Word id for padding symbols.
    :param unk_id: Word id for unknown symbols.
    :param dtype: Data type of generated NDArrays.
    """

    def __init__(self,
                 source_sentences: List[List[int]],
                 target_sentences: List[List[int]],
                 buckets: List[Tuple[int, int]],
                 batch_size: int,
                 eos_id: int,
                 pad_id: int,
                 unk_id: int,
                 fill_up: Optional[str] = None,
                 source_data_name=C.SOURCE_NAME,
                 source_data_length_name=C.SOURCE_LENGTH_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 dtype='float32'):
        super(ParallelBucketSentenceIter, self).__init__()

        self.buckets = list(buckets)
        self.buckets.sort()
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.batch_size = batch_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.dtype = dtype
        self.source_data_name = source_data_name
        self.source_data_length_name = source_data_length_name
        self.target_data_name = target_data_name
        self.label_name = label_name
        self.fill_up = fill_up

        # TODO: consider avoiding explicitly creating length and label arrays to save host memory
        self.data_source = [[] for _ in self.buckets]
        self.data_length = [[] for _ in self.buckets]
        self.data_target = [[] for _ in self.buckets]
        self.data_label = [[] for _ in self.buckets]

        # assign sentence pairs to buckets
        self._assign_to_buckets(source_sentences, target_sentences)

        # convert to single numpy array for each bucket
        self._convert_to_array()

        self.provide_data = [
            mx.io.DataDesc(name=source_data_name, shape=(batch_size, self.default_bucket_key[0]), layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=source_data_length_name, shape=(batch_size,), layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=target_data_name, shape=(batch_size, self.default_bucket_key[1]), layout=C.BATCH_MAJOR)]
        self.provide_label = [
            mx.io.DataDesc(name=label_name, shape=(self.batch_size, self.default_bucket_key[1]), layout=C.BATCH_MAJOR)]

        self.data_names = [self.source_data_name, self.source_data_length_name, self.target_data_name]
        self.label_names = [self.label_name]

        # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
        self.idx = []
        for i, buck in enumerate(self.data_source):
            rest = len(buck) % batch_size
            if rest > 0:
                logger.info("Discarding %d samples from bucket %s due to incomplete batch", rest, self.buckets[i])
            idxs = [(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)]
            self.idx.extend(idxs)
        self.curr_idx = 0

        # holds NDArrays
        self.indices = []  # This will define how the data arrays will be organized
        self.nd_source = []
        self.nd_length = []
        self.nd_target = []
        self.nd_label = []

        self.reset()

    def _assign_to_buckets(self, source_sentences, target_sentences):
        ndiscard = 0
        tokens_source = 0
        tokens_target = 0
        num_of_unks_source = 0
        num_of_unks_target = 0
        for source, target in zip(source_sentences, target_sentences):
            tokens_source += len(source)
            tokens_target += len(target)
            num_of_unks_source += source.count(self.unk_id)
            num_of_unks_target += target.count(self.unk_id)

            buck_idx, buck = get_parallel_bucket(self.buckets, len(source), len(target))
            if buck is None:
                ndiscard += 1
                continue

            buff_source = np.full((buck[0],), self.pad_id, dtype=self.dtype)
            buff_target = np.full((buck[1],), self.pad_id, dtype=self.dtype)
            buff_label = np.full((buck[1],), self.pad_id, dtype=self.dtype)
            buff_source[:len(source)] = source
            buff_target[:len(target)] = target
            buff_label[:len(target)] = target[1:] + [self.eos_id]
            self.data_source[buck_idx].append(buff_source)
            self.data_length[buck_idx].append(len(source))
            self.data_target[buck_idx].append(buff_target)
            self.data_label[buck_idx].append(buff_label)

        logger.info("Source words: %d", tokens_source)
        logger.info("Target words: %d", tokens_target)
        logger.info("Vocab coverage source: %.0f%%", (1 - num_of_unks_source / tokens_source) * 100)
        logger.info("Vocab coverage target: %.0f%%", (1 - num_of_unks_target / tokens_target) * 100)
        logger.info('Total: {0} samples in {1} buckets'.format(len(self.data_source), len(self.buckets)))
        nsamples = 0
        for bkt, buck in zip(self.buckets, self.data_length):
            logger.info("bucket of {0} : {1} samples".format(bkt, len(buck)))
            nsamples += len(buck)
        assert nsamples > 0, "0 data points available in the data iterator. " \
                             "%d data points have been discarded because they didn't fit into any bucket. Consider " \
                             "increasing the --max-seq-len to fit your data." % ndiscard
        logger.info("%d sentence pairs out of buckets", ndiscard)
        logger.info("fill up mode: %s", self.fill_up)
        logger.info("")

    def _convert_to_array(self):
        for i in range(len(self.data_source)):
            self.data_source[i] = np.asarray(self.data_source[i], dtype=self.dtype)
            self.data_length[i] = np.asarray(self.data_length[i], dtype=self.dtype)
            self.data_target[i] = np.asarray(self.data_target[i], dtype=self.dtype)
            self.data_label[i] = np.asarray(self.data_label[i], dtype=self.dtype)

            n = len(self.data_source[i])
            if n % self.batch_size != 0:
                buck_shape = self.buckets[i]
                rest = self.batch_size - n % self.batch_size
                if self.fill_up == 'pad':
                    raise NotImplementedError
                elif self.fill_up == 'replicate':
                    logger.info(
                        "Replicating %d random examples from bucket %s to size it to multiple of batch size %d", rest,
                        buck_shape, self.batch_size)
                    random_indices = np.random.randint(self.data_source[i].shape[0], size=rest)

                    self.data_source[i] = np.concatenate((self.data_source[i], self.data_source[i][random_indices, :]),
                                                         axis=0)
                    self.data_length[i] = np.concatenate((self.data_length[i], self.data_length[i][random_indices]),
                                                         axis=0)
                    self.data_target[i] = np.concatenate((self.data_target[i], self.data_target[i][random_indices, :]),
                                                         axis=0)
                    self.data_label[i] = np.concatenate((self.data_label[i], self.data_label[i][random_indices, :]),
                                                        axis=0)

    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_idx = 0
        # shuffle indices
        random.shuffle(self.idx)

        self.nd_source = []
        self.nd_length = []
        self.nd_target = []
        self.nd_label = []
        self.indices = []
        for i in range(len(self.data_source)):
            # shuffle indices within each bucket
            self.indices.append(np.random.permutation(len(self.data_source[i])))
            self._append_ndarrays(i, self.indices[-1])

    def _append_ndarrays(self, bucket: int, shuffled_indices: np.array):
        """
        Appends the actual data, selected by the given indices, to the NDArrays
        of the appropriate bucket. Use when reshuffling the data.

        :param bucket: Current bucket.
        :param shuffled_indices: Indices indicating which data to select.
        """
        self.nd_source.append(mx.nd.array(self.data_source[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_length.append(mx.nd.array(self.data_length[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_target.append(mx.nd.array(self.data_target[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))
        self.nd_label.append(mx.nd.array(self.data_label[bucket].take(shuffled_indices, axis=0), dtype=self.dtype))

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_idx != len(self.idx)

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        source = self.nd_source[i][j:j + self.batch_size]
        length = self.nd_length[i][j:j + self.batch_size]
        target = self.nd_target[i][j:j + self.batch_size]
        data = [source, length, target]
        label = [self.nd_label[i][j:j + self.batch_size]]

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
            pickle.dump(self.idx, fp)
            pickle.dump(self.curr_idx, fp)
            np.save(fp, self.indices)

    def load_state(self, fname: str):
        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """
        with open(fname, "rb") as fp:
            self.idx = pickle.load(fp)
            self.curr_idx = pickle.load(fp)
            self.indices = np.load(fp)

        # Because of how checkpointing is done (pre-fetching the next batch in
        # each iteration), curr_idx should be always >= 1
        assert self.curr_idx >= 1
        # Right after loading the iterator state, next() should be called
        self.curr_idx -= 1

        self.nd_source = []
        self.nd_length = []
        self.nd_target = []
        self.nd_label = []
        for i in range(len(self.data_source)):
            self._append_ndarrays(i, self.indices[i])

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

import tempfile

import numpy as np
import pytest

import sockeye.data_io
from test.common import generate_random_sentence


def create_parallel_sentence_iter(source_sentences, target_sentences, max_len, batch_size, batch_by_words):
    buckets = sockeye.data_io.define_parallel_buckets(max_len, max_len, 10)
    batch_num_devices = 1
    eos = 0
    pad = 1
    unk = 2
    bucket_iterator = sockeye.data_io.ParallelBucketSentenceIter(source_sentences,
                                                                 target_sentences,
                                                                 buckets,
                                                                 batch_size,
                                                                 batch_by_words,
                                                                 batch_num_devices,
                                                                 eos, pad, unk)
    return bucket_iterator


def data_batches_equal(db1, db2):
    # We just compare the data, should probably be enough
    equal = True
    for data1, data2 in zip(db1.data, db2.data):
        equal = equal and np.allclose(data1.asnumpy(), data2.asnumpy())
    return equal


@pytest.mark.parametrize("batch_size, batch_by_words", [
    (50, False),
    (123, True),
])
def test_parallel_sentence_iter(batch_size, batch_by_words):
    # Create random sentences
    vocab_size = 100
    max_len = 100
    source_sentences = []
    target_sentences = []
    for _ in range(1000):
        source_sentences.append(generate_random_sentence(vocab_size, max_len))
        target_sentences.append(generate_random_sentence(vocab_size, max_len))

    ori_iterator = create_parallel_sentence_iter(source_sentences, target_sentences, max_len, batch_size, batch_by_words)
    ori_iterator.reset()  # Random order
    # Simulate some iterations
    ori_iterator.next()
    ori_iterator.next()
    ori_iterator.next()
    ori_iterator.next()
    expected_output = ori_iterator.next()
    # expected_output because the user is expected to call next() after loading

    # Save the state to disk
    tmp_file = tempfile.NamedTemporaryFile()
    ori_iterator.save_state(tmp_file.name)

    # Load the state in a new iterator
    load_iterator = create_parallel_sentence_iter(source_sentences, target_sentences, max_len, batch_size, batch_by_words)
    load_iterator.reset()  # Random order
    load_iterator.load_state(tmp_file.name)

    # Compare the outputs
    loaded_output = load_iterator.next()
    assert data_batches_equal(expected_output, loaded_output)

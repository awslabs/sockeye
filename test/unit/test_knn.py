# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import torch as pt
from unittest.mock import patch, mock_open, Mock

from sockeye.generate_decoder_states import NumpyMemmapStorage, DecoderStateGenerator
from sockeye.knn import KNNConfig, FaissIndexBuilder
from sockeye.vocab import build_vocab

# Only run certain tests in this file if faiss is installed
try:
    import faiss  # pylint: disable=E0401
    faiss_installed = True
except:
    faiss_installed = False


def test_numpy_memmap_storage():

    # test open
    with patch('numpy.memmap') as mock_np_memmap:
        store = NumpyMemmapStorage("foo", 64, np.float16)
        store.open(64, 8)
        mock_np_memmap.assert_called_once_with("foo", dtype=np.float16, mode='w+', shape=(64, 64))

    mock_storage = np.zeros((64, 64), dtype=np.float16)
    # test add
    store.mmap.__setitem__ = Mock()
    store.mmap.__setitem__.side_effect = mock_storage.__setitem__
    ones_block_16 = np.ones((16, 64), dtype=np.float16)
    ones_block_32 = np.ones((32, 64), dtype=np.float16)
    zeros_block_16 = np.zeros((16, 64), dtype=np.float16)
    zeros_block_32 = np.zeros((32, 64), dtype=np.float16)
    # add for 0-15
    store.add(ones_block_16)
    store.mmap.__setitem__.assert_called()
    assert (mock_storage[:16, :] == ones_block_16).all()
    assert (mock_storage[16:32, :] == zeros_block_16).all()
    # add for 16-31
    store.add(ones_block_16)
    assert (mock_storage[:32, :] == ones_block_32).all()
    assert (mock_storage[32:, :] == zeros_block_32).all()
    # add with size overflow -- this should trigger a warning without doing anything
    store.add(np.ones((64, 64), dtype=np.float16))
    assert (mock_storage[:32, :] == ones_block_32).all()
    assert (mock_storage[32:, :] == zeros_block_32).all()


def test_decoder_state_generator():
    # utils.is_gzip_file expects byte-like object, hence supplying a byte string
    data = b'One Ring to rule them all, One Ring to find them'

    model = Mock()
    vocabs = [build_vocab([data])]

    generator = DecoderStateGenerator(model, vocabs, vocabs, 'foo', 1, 1,
                                      'float16', 'int32', pt.device('cpu'))

    # test init_store_file
    with patch('sockeye.generate_decoder_states.NumpyMemmapStorage.open') as mock_memmap_open:
        generator.init_store_file(64)
        mock_memmap_open.assert_called_with(64, 1)

    generator.dimension = 64

    # test generate_states_and_store
    with patch('builtins.open', new=mock_open(read_data=data)) as _file, \
            patch('torch.jit.trace_module') as mock_jit_trace, \
            patch('sockeye.generate_decoder_states.NumpyMemmapStorage.add') as mock_memmap_add:
        data_paths = ['foo/bar']
        generator.generate_states_and_store(data_paths, data_paths, 1)

        _file.assert_called_with('foo/bar', mode='rt', encoding='utf-8', errors='replace')
        mock_jit_trace.assert_called_once()
        assert generator.traced_model is not None
        generator.traced_model.get_decoder_states.assert_called_once()
        mock_memmap_add.assert_called()

    # test save_config
    with patch('sockeye.config.open', new=mock_open()) as _file:
        generator.save_config()
        _file.assert_called_once_with("foo.conf.yaml", 'w')


@pytest.mark.skipif(not faiss_installed, reason='Faiss is not installed')
def test_faiss_index_builder():
    num_data_points = 16
    num_dimensions = 16

    config = KNNConfig(num_data_points, num_dimensions, 'float32', 'int32', "Flat", -1)
    builder = FaissIndexBuilder(config)
    index = builder.init_faiss_index()

    # build data
    states = np.outer(np.arange(num_data_points, dtype=np.float32), np.ones(num_dimensions, dtype=np.float32))

    # offset should be < 0.5
    def query_tests(offset):
        # check by querying into the index
        for i in range(1, num_data_points - 1):
            query = np.expand_dims(states[i], axis=0) + offset
            dists, idxs = index.search(query, 3)  # 3 because we expect to see itself and the two neighboring ones

            # check for idxs
            assert idxs[0][0] == i

            # check for dists
            # note that faiss won't do sqrt for the L2 distances
            # (reference: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_l2)
            gld_dists = np.array([[np.power(offset, 2) * num_dimensions,
                                   np.power(1 - offset, 2) * num_dimensions, np.power(1 + offset, 2) * num_dimensions]])
            assert np.all(np.isclose(dists, gld_dists))

    # test add_items
    builder.add_items(index, states)
    query_tests(0.1)
    index.reset()

    # test block_add_items
    builder.block_add_items(index, states, block_size=3)  # add items with block size of 3
    query_tests(0.1)

    # test build_train_sample
    train_sample = builder.build_train_sample(states, 8)
    train_sample.shape == 8, num_dimensions
    states.dtype == train_sample.dtype

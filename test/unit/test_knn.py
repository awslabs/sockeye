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
import os
import pytest
import tempfile
import torch as pt

import sockeye
import sockeye.constants as C
from sockeye.generate_decoder_states import NumpyMemmapStorage, DecoderStateGenerator
from sockeye.knn import KNNConfig, FaissIndexBuilder, get_config_path, get_state_store_path, get_word_store_path
from sockeye.vocab import build_vocab

# Only run certain tests in this file if faiss is installed
try:
    import faiss  # pylint: disable=E0401
    faiss_installed = True
except:
    faiss_installed = False


def test_numpy_memmap_storage():
    # test open
    with tempfile.TemporaryDirectory() as memmap_dir:
        memmap_file = os.path.join(memmap_dir, "foo")
        store = NumpyMemmapStorage(memmap_file, 64, np.float16)
        store.open(64, 8)

        # test add
        ones_block_16 = np.ones((16, 64), dtype=np.float16)
        ones_block_32 = np.ones((32, 64), dtype=np.float16)
        zeros_block_16 = np.zeros((16, 64), dtype=np.float16)
        zeros_block_32 = np.zeros((32, 64), dtype=np.float16)
        # add for 0-15
        store.add(ones_block_16)
        assert (store.mmap[:16, :] == ones_block_16).all()
        assert (store.mmap[16:32, :] == zeros_block_16).all()
        assert (store.mmap[32:, :] == zeros_block_32).all()
        # add for 16-31
        store.add(ones_block_16)
        assert (store.mmap[:32, :] == ones_block_32).all()
        assert (store.mmap[32:, :] == zeros_block_32).all()
        # add with size overflow -- this should trigger a warning without doing anything
        store.add(np.ones((64, 64), dtype=np.float16))
        assert (store.mmap[:32, :] == ones_block_32).all()
        assert (store.mmap[32:, :] == zeros_block_32).all()


def test_decoder_state_generator():
    data = 'One Ring to rule them all, One Ring to find them'
    max_seq_len_source = 30
    max_seq_len_target = 30

    vocabs = [build_vocab([data])]
    config_embed = sockeye.encoder.EmbeddingConfig(vocab_size=len(vocabs[0]), num_embed=16, dropout=0.0)
    config_encoder = sockeye.encoder.EncoderConfig(model_size=16, attention_heads=2,
                                                   feed_forward_num_hidden=16, depth_key_value=16,
                                                   act_type='relu', num_layers=2, dropout_attention=0.0,
                                                   dropout_act=0.0, dropout_prepost=0.0,
                                                   positional_embedding_type='fixed', preprocess_sequence='n',
                                                   postprocess_sequence='n', max_seq_len_source=max_seq_len_source,
                                                   max_seq_len_target=max_seq_len_target)
    config_data = sockeye.data_io.DataConfig(data_statistics=None,
                                             max_seq_len_source=max_seq_len_source,
                                             max_seq_len_target=max_seq_len_target,
                                             num_source_factors=0, num_target_factors=0)
    config = sockeye.model.ModelConfig(config_data=config_data,
                                       vocab_source_size=len(vocabs[0]), vocab_target_size=len(vocabs[0]),
                                       config_embed_source=config_embed, config_embed_target=config_embed,
                                       config_encoder=config_encoder, config_decoder=config_encoder)

    with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as data_dir:
        params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)

        # create and save float32 model
        model = sockeye.model.SockeyeModel(config=config)
        assert model.dtype == pt.float32
        for param in model.parameters():
            assert param.dtype == pt.float32
        model.save_config(model_dir)
        model.save_version(model_dir)
        model.save_parameters(params_fname)
        model.eval()

        # add dummy sentence to data_path
        data_path = os.path.join(data_dir, "data.txt")
        data_file = open(data_path, 'w')
        data_file.write(data + '\n')
        data_file.close()

        # create generator from mock model and vocab
        generator = DecoderStateGenerator(model, vocabs, vocabs, data_dir, max_seq_len_source, max_seq_len_target,
                                          'float32', 'int32', pt.device('cpu'))
        max_seq_len_target = min(max_seq_len_target + C.SPACE_FOR_XOS, max_seq_len_target)
        generator.num_states = DecoderStateGenerator.probe_token_count(data_path, max_seq_len_target)

        # test init_store_file
        generator.init_store_file(generator.num_states)
        generator.dimension = 16

        # test generate_states_and_store
        data_paths = [data_path]
        generator.generate_states_and_store(data_paths, data_paths, 1)

        # check if state and word store files are there
        assert os.path.isfile(get_state_store_path(data_dir))
        assert os.path.isfile(get_word_store_path(data_dir))

        # test save_config
        generator.save_config()

        # check if the config content makes sense
        config = KNNConfig.load(get_config_path(data_dir))
        assert config.index_size == DecoderStateGenerator.probe_token_count(data_path, max_seq_len_target)
        assert config.dimension == 16
        assert config.state_data_type == 'float32'
        assert config.word_data_type == 'int32'
        assert config.index_type == ''
        assert config.train_data_size == -1


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
    assert train_sample.shape == (8, num_dimensions)
    assert states.dtype == train_sample.dtype

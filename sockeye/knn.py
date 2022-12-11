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

from dataclasses import dataclass
import logging
import numpy as np
import os
import shutil
from typing import Optional

from . import arguments
from sockeye import config, utils, constants as C
from sockeye.log import setup_main_logger

try:
    import faiss
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class KNNConfig(config.Config):
    """
    KNNConfig defines knn-specific configurations, including the information about the data store
    as well as the index itself.

    :param index_size: Size of the index and the data store.
    :param dimension: Number of dimensions of the decoder states.
    :param state_data_type: Data type of the decoder states (keys).
    :param word_data_type: Data type of the stored word indexes (values).
    :param index_type: faiss index signature, see https://github.com/facebookresearch/faiss/wiki/The-index-factory
    :param train_data_size: Size of the training data used to train the index (if it needs to be trained).
    """
    index_size: int
    dimension: int
    state_data_type: str
    word_data_type: str
    index_type: str
    train_data_size: int


class FaissIndexBuilder:
    """
    Builds a faiss index from a data store containing stored keys (i.e., decoder hidden states for k-NN-based MT).

    :param config: a KNNConfig object containing the index configuration information
    :param use_gpu: build index on a gpu
    :param device_id: device id if building index on gpu
    """

    def __init__(self, config: KNNConfig, use_gpu: bool = False, device_id: int = 0):
        utils.check_import_faiss()  # faiss will definitely be used for this class, so check here
        self.config = config
        self.use_gpu = use_gpu
        self.device_id = device_id

    def init_faiss_index(self, train_sample: Optional[np.memmap] = None):
        """Initialize the Faiss index to be built and conduct training if needed."""
        index = faiss.index_factory(self.config.dimension, self.config.index_type)
        # pylint is disabled for members that only exists in faiss-gpu
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # pylint: disable=no-member
            co = faiss.GpuClonerOptions()  # pylint: disable=no-member
            index = faiss.index_cpu_to_gpu(res, self.device_id, index, co)  # pylint: disable=no-member

        if not index.is_trained and train_sample is not None:
            index.train(train_sample.astype(np.float32))  # unfortunately, faiss index only supports float32
        elif not index.is_trained:
            logger.error("Index needs training but no training sample is passed.")

        return index

    def add_items(self, index, keys: np.ndarray):
        """Add items to the index (must call `init_faiss_index` first)."""
        item_count, key_dim = keys.shape
        assert key_dim == self.config.dimension

        index.add(keys.astype(np.float32))  # unfortunately, faiss index only supports float32

    def block_add_items(self, index, keys: np.ndarray, block_size: int = C.DEFAULT_DATA_STORE_BLOCK_SIZE):
        """Add items to the index in blocks -- used for a large number of items (must call `init_faiss_index` first)."""
        item_count, key_dim = keys.shape
        assert key_dim == self.config.dimension

        n_blocks = item_count // block_size
        for i in range(n_blocks):
            logger.debug(f"adding block no.{i}")
            start = block_size * i
            end = block_size * (i + 1)
            index.add(keys[start:end].astype(np.float32))  # unfortunately, faiss index only supports float32

        if block_size * n_blocks < item_count:
            start = block_size * n_blocks
            index.add(keys[start:item_count].astype(np.float32))  # unfortunately, faiss index only supports float32

    @staticmethod
    def build_train_sample(keys: np.ndarray, sample_size: int):
        """Randomly sample `sample_size` keys as training sample."""
        item_count, _ = keys.shape
        assert 0 < sample_size <= item_count

        if sample_size < item_count:
            train_sample_idx = np.random.choice(np.arange(item_count), size=[sample_size], replace=False)
            train_sample = keys[train_sample_idx]
        else:
            train_sample = keys

        return train_sample

    def build_faiss_index(self, keys: np.ndarray, train_sample: Optional[np.memmap] = None):
        """
        Top-level function of the class to build faiss index for a set of keys, optionally with samples for training.
        """
        item_count, _ = keys.shape
        if train_sample is None and self.config.train_data_size > 0:
            train_sample = FaissIndexBuilder.build_train_sample(keys, self.config.train_data_size)

        index = self.init_faiss_index(train_sample)
        self.block_add_items(index, keys)

        return index


def get_state_store_path(dir):
    """Get the path to the state store file given a kNN export directory."""
    return os.path.join(dir, C.KNN_STATE_DATA_STORE_NAME)


def get_word_store_path(dir):
    """Get the path to the word store file given a kNN export directory."""
    return os.path.join(dir, C.KNN_WORD_DATA_STORE_NAME)


def get_config_path(dir):
    """Get the path to the kNN config file given a kNN export directory."""
    return os.path.join(dir, C.KNN_CONFIG_NAME)


def build_knn_index_package(args):
    """
    Top-level function that builds a kNN index package (kNN index and config file) from an existing state and word
    store.
    """
    state_store_filename = get_state_store_path(args.input_dir)
    word_store_filename = get_word_store_path(args.input_dir)
    config_filename = get_config_path(args.input_dir)
    utils.check_condition(os.path.exists(state_store_filename), f"Input file {state_store_filename} not found!")
    utils.check_condition(os.path.exists(word_store_filename), f"Input file {word_store_filename} not found!")
    utils.check_condition(os.path.exists(config_filename), f"Config file {config_filename} not found!")
    utils.check_import_faiss()

    setup_main_logger(file_logging=False,
                      console=not args.quiet,
                      level=args.loglevel)  # pylint: disable=no-member
    utils.log_basic_info(args)

    config = KNNConfig.load(config_filename)
    if args.index_type is not None:
        config.index_type = args.index_type
    if args.train_data_size is not None:
        config.train_data_size = args.train_data_size

    keys = np.memmap(state_store_filename, dtype=config.state_data_type,
                     mode='r', shape=(config.index_size, config.dimension))
    builder = FaissIndexBuilder(config, not args.use_cpu, args.device_id)
    train_sample = None
    if args.train_data_input_file is not None:
        train_sample = np.memmap(args.train_data_input_file, dtype=config.state_data_type,
                                 mode='r', shape=(config.index_size, config.dimension))
    index = builder.build_faiss_index(keys, train_sample)

    if not args.use_cpu:
        index_cpu = faiss.index_gpu_to_cpu(index)  # pylint: disable=no-member
    else:
        index_cpu = index

    if not args.output_dir:
        args.output_dir = args.input_dir
    elif args.output_dir != args.input_dir:
        shutil.copy(word_store_filename, os.path.join(args.output_dir, C.KNN_WORD_DATA_STORE_NAME))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    faiss.write_index(index_cpu, os.path.join(args.output_dir, C.KNN_INDEX_NAME))
    config.save(os.path.join(args.output_dir, C.KNN_CONFIG_NAME))


def main():
    params = arguments.ConfigArgumentParser(description='CLI to build knn index.')
    arguments.add_build_knn_index_args(params)
    arguments.add_logging_args(params)
    arguments.add_device_args(params)
    args = params.parse_args()

    build_knn_index_package(args)


if __name__ == "__main__":
    main()

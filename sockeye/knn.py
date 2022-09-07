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

import argparse
from abc import abstractmethod
from dataclasses import dataclass
import logging
import math
import numpy as np
import os
import random
import shutil
from typing import Dict, Iterable, List, Optional, Tuple, Callable

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
    KNNConfig defines knn-specific configurations, including the information about the data dump
    as well as the index itself.

    :param index_size: Size of the index and the data dump.
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

    def __init__(self, config: KNNConfig, use_gpu: bool = False, device_id: int = 0):
        utils.init_faiss()  # faiss will definitely be used for this class, so check here
        self.config = config
        self.use_gpu = use_gpu
        self.device_id = device_id

    def init_faiss_index(self, train_sample: Optional[np.memmap] = None):
        index = faiss.index_factory(self.config.dimension, self.config.index_type)
        if self.use_gpu is True:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            index = faiss.index_cpu_to_gpu(res, self.device_id, index, co)

        if not index.is_trained and train_sample is not None:
            index.train(train_sample.astype(np.float32))  # unfortunately, faiss index only supports float32
        elif not index.is_trained:
            logger.error("Index needs training but no training sample is passed.")

        return index

    def add_items(self, index, keys: np.array):
        item_count, key_dim = keys.shape
        assert key_dim == self.config.dimension

        index.add(keys.astype(np.float32))  # unfortunately, faiss index only supports float32

    def block_add_items(self, index, keys: np.array, block_size: int = 1024*1024):
        item_count, key_dim = keys.shape
        assert key_dim == self.config.dimension

        n_blocks = item_count // block_size
        for i in range(n_blocks):
            logger.info("adding block no.{0}".format(i))
            start = block_size * i
            end = block_size * (i + 1)
            index.add(keys[start:end].astype(np.float32))  # unfortunately, faiss index only supports float32

        if block_size * n_blocks < item_count:
            start = block_size * n_blocks
            index.add(keys[start:item_count].astype(np.float32))  # unfortunately, faiss index only supports float32

    @staticmethod
    def build_train_sample(keys: np.array, sample_size: int):
        item_count, _ = keys.shape
        assert 0 < sample_size <= item_count

        if sample_size < item_count:
            train_sample_idx = np.random.choice(np.arange(item_count), size=[sample_size], replace=False)
            train_sample = keys[train_sample_idx]
        else:
            train_sample = keys

        return train_sample

    def build_faiss_index(self, keys: np.array, train_sample: Optional[np.memmap] = None):
        if train_sample is None and self.config.train_data_size > 0:
            train_sample = FaissIndexBuilder.build_train_sample(keys, self.config.train_data_size)
        
        index = self.init_faiss_index(train_sample)
        self.block_add_items(index, keys)

        return index


def get_state_dump_filename(prefix: str) -> str:
    return prefix + ".states.npy"

def get_word_dump_filename(prefix: str) -> str:
    return prefix + ".words.npy"

def main():
    params = arguments.ConfigArgumentParser(description='CLI to build knn index.')
    arguments.add_build_knn_index_args(params)
    arguments.add_logging_args(params)
    arguments.add_device_args(params)
    args = params.parse_args()

    state_dump_filename = get_state_dump_filename(args.input_dump_prefix)
    word_dump_filename = get_word_dump_filename(args.input_dump_prefix)
    utils.check_condition(os.path.exists(state_dump_filename), f"Input file {state_dump_filename} not found!")
    utils.check_condition(os.path.exists(word_dump_filename), f"Input file {word_dump_filename} not found!")
    utils.check_condition(os.path.exists(args.config_file), f"Config file {args.config_file} not found!")
    utils.init_faiss()

    setup_main_logger(file_logging=False,
                      console=not args.quiet,
                      level=args.loglevel)  # pylint: disable=no-member
    utils.log_basic_info(args)

    config = KNNConfig.load(args.config_file)
    if args.index_type is not None:
        config.index_type = args.index_type
    if args.train_data_size is not None:
        config.train_data_size = args.train_data_size

    keys = np.memmap(state_dump_filename, dtype=config.state_data_type, mode='r', shape=(config.index_size, config.dimension))
    builder = FaissIndexBuilder(config, not args.use_cpu, args.device_id)
    train_sample = None
    if args.train_data_input_file is not None:
        train_sample = np.memmap(args.train_data_input_file, dtype=config.state_data_type, mode='r', shape=(config.index_size, config.dimension))
    index = builder.build_faiss_index(keys, train_sample)

    if not args.use_cpu:
        index_cpu = faiss.index_gpu_to_cpu(index)
    else:
        index_cpu = index

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        faiss.write_index(index_cpu, os.path.join(args.output_dir, "key_index"))
        config.save(os.path.join(args.output_dir, "config.yaml"))
        shutil.copy(word_dump_filename, os.path.join(args.output_dir, "vals.npy"))


if __name__ == "__main__":
    main()

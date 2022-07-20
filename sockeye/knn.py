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
import argparse
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import faiss
import logging
import numpy as np
import os
from sockeye import config, utils, constants as C
from sockeye.log import setup_main_logger
from typing import Dict, Iterable, List, Optional, Tuple, Callable

#from torch import long
#from itertools import chain, islice

logger = logging.getLogger(__name__)

@dataclass
class KNNConfig(config.Config):
    index_size: int
    dimension: int
    data_type: str  # must be primitive type
    index_type: str  # must be primitive type


def get_numpy_dtype(config):
    if config.data_type == "float16":
        return np.float16
    if config.data_type == "float32":
        return np.float32
    if config.data_type == "int16":
        return np.int16
    raise NotImplementedError


def get_faiss_index(config):
    if config.index_type == "IndexFlatL2":
        return faiss.IndexFlatL2(config.dimension)
    raise NotImplementedError


def build_from_path(input_file: str, output_file: str, config: KNNConfig):
    """
    :param input_file: Path to memmap file that stores the keys to be indexed.
    :param output_file: Path to the output index file.
    :param config: The KNNConfig object.
    :param index_type: The index type.
    :return: The faiss index object.
    """
    index_size = config["index_size"]
    dimention = config["dimention"]
    data_type = config["data_type"]
    index = get_faiss_index(config) # initialize the index
    logger.info(f"index.is_trained: {index.is_trained}")
    keys = np.memmap(input_file, dtype=data_type, mode='r', shape=(index_size, dimention)) # load key vectors from the memmap file
    index.add(keys) # add vectors to the index
    logger.info(f"index.ntotal: {index.ntotal}")
    faiss.write_index(index, output_file) # Dump index to output file
    return index


def get_index_file_path(input_file: str) -> str:
    return f"{input_file}.knn.idx"


def build_index(args: argparse.Namespace):
    input_file = args.input_file
    utils.check_condition(os.path.exists(input_file), f"Input file {input_file} not found!")
    config_file = args.config_file
    utils.check_condition(os.path.exists(input_file), f"Config file {config_file} not found!")

    index_file = get_index_file_path(input_file)
    setup_main_logger(file_logging=not args.no_logfile, console=not args.quiet,
                      path="%s.%s" % (index_file, C.LOG_NAME))
    logger.info(f"Build index for: {input_file}")
    knn_config = KNNConfig.load(config_file)
    logger.info(f"Config: {knn_config}")

    build_from_path(input_file, index_file, knn_config)
    logger.info(f"Index file is saved to: {index_file}.")


def load_from_path(index_file: str):
    """
    :param index_file: Path to index file.
    :return: The faiss index object.
    """
    index = faiss.read_index(index_file)
    return index


def search_index(index, query_keys: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param index: The faiss index object.
    :param query_keys: The array of keys to be queried.
    :param k: The number of nearest neighbors to be returned.
    :return: Both Tuple[0] and Tuple[1] has the same shape(len(query_keys), k). 
    Tuple[0] contains distance of each returned neighbor. Tuple[1] contains the index of each returned neighbor.
    """
    return index.search(query_keys, k)


def main():
    from . import arguments
    params = argparse.ArgumentParser(description='CLI to build knn index.')
    arguments.add_build_knn_index_args(params)
    arguments.add_logging_args(params)
    args = params.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()

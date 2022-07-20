# Copyright 2018--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
import os
from typing import Dict, List

import faiss
import numpy as np
import torch as pt

from . import arguments
from . import constants as C
from . import data_io
from . import utils
from .log import setup_main_logger
from .model import SockeyeModel, load_model
from .vocab import Vocab
from .utils import check_condition

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = logging.getLogger(__name__)

# a general blocked mmap stroage interface
# TODO: could be implemented with numpy memmap, h5py, zarr, etc.
# TODO: add (1) load from existing file dump (2) read from existing dump
class BlockMMapStorage:

    def __init__(self,
                 file_name: str,
                 num_dim: int,
                 dtype: np.dtype) -> None:
        self.file_name = file_name
        self.num_dim = num_dim  # dimension of a single entry
        self.dtype = dtype
        self.block_size = -1
        self.mmap = None
        self.tail_idx = 0  # where the next entry should be inserted
        self.size = 0  # size of storage already assigned

    def open(self, initial_size: int, block_size: int) -> None:
        pass

    def add(self, array: np.ndarray) -> None:
        pass


class NumpyMemmapStorage(BlockMMapStorage):
    
    def open(self, initial_size: int, block_size: int) -> None:
        self.mmap = np.memmap(self.file_name, dtype=self.dtype, mode='w+', shape=(initial_size, self.num_dim))
        self.size = initial_size
        self.block_size = block_size

    def add(self, array: np.ndarray) -> None:
        """
        It turns out that numpy memmap actually cannot be re-sized.
        So we have to pre-estimate how many entries we need and put it down as initial_size.
        If we end up adding more entries to the memmap than initially claimed, we'll have to bail out.
        """
        assert self.mmap != None
        num_entries, num_dim = array.shape
        assert num_dim == self.num_dim

        if self.tail_idx + num_entries > self.size:
            # bail out
            logger.warning(
                "Trying to write {0} entries into a numpy memmap that has size {1} and already has {2} entries. Nothing is written."
                .format(num_entries, self.size, self.tail_idx)
            )
        else:
            start = self.tail_idx
            end = self.tail_idx + num_entries
            self.mmap[start:end] = array

            self.tail_idx += num_entries


class StateDumper:

    def __init__(self,
                 model: SockeyeModel,
                 source_vocabs: List[Vocab],
                 target_vocabs: List[Vocab],
                 dump_path: str,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 device: pt.device) -> None:
        self.model = model
        self.source_vocabs = source_vocabs
        self.target_vocabs = target_vocabs
        self.device = device
        self.traced_get_decode_states = None
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target

        self.state_dump_file = None
        self.words_dump_file = None

    @staticmethod
    def probe_token_count(self, target: str) -> None:
        token_count = 0
        with open(target) as f:
            for line in f:
                token_count += len(line.strip().split(' '))  # TODO: +2 for BOS and EOS?

    def init_dump_file(self, initial_size: int) -> None:
        self.state_dump_file = NumpyMemmapStorage(self.dump_path, self.model.config.config_decoder.model_size, np.float16)  # TODO: shouldn't hard-code dtype
        self.words_dump_file = NumpyMemmapStorage(self.dump_path, 1, np.int16)  # TODO: shouldn't hard-code dtype
        self.state_dump_file.open(initial_size, 1)
        self.words_dump_file.open(initial_size, 1)

    def build_states_and_dump(self,
                              sources: List[str],
                              targets: List[str],
                              batch_size: int) -> None:

        assert self.state_dump_file != None, \
               "You should call probe_token_count first to initialize the dump files."

        # get data iter
        data_iter = data_io.get_scoring_data_iters(
            sources=sources,
            targets=targets,
            source_vocabs=self.source_vocabs,
            target_vocabs=self.target_vocabs,
            batch_size=batch_size,
            max_seq_len_source=self.max_seq_len_source,
            max_seq_len_target=self.max_seq_len_target
        )

        for batch_no, batch in enumerate(data_iter, 1):
            # get decoder states
            batch = batch.load(self.device)
            model_inputs = (batch.source, batch.source_length, batch.target, batch.target_length)
            if self.traced_model is None:
                self.traced_get_decode_states = pt.jit.trace(self.model.get_decoder_states, model_inputs, strict=False)
            decoder_states = self.traced_get_decode_states(*model_inputs)  # shape: (batch, sent_len, hidden_dim)

            # flatten batch and sent_len dimensions, remove pads on the target
            pad_mask = [ batch.target != C.PAD_ID ]
            flat_target = batch.target[pad_mask]
            flat_states = decoder_states[pad_mask]

            # dump
            self.state_dump_file.add(flat_states)
            self.words_dump_file.add(flat_target)


def main():
    params = arguments.ConfigArgumentParser(description='Score data with an existing model.')
    arguments.add_score_cli_args(params)
    args = params.parse_args()
    check_condition(args.batch_type == C.BATCH_TYPE_SENTENCE, "Batching by number of words is not supported")

    setup_main_logger(file_logging=False,
                      console=not args.quiet,
                      level=args.loglevel)  # pylint: disable=no-member

    utils.log_basic_info(args)

    dump(args)


def dump(args: argparse.Namespace):
    setup_main_logger(file_logging=False,
                      console=not args.quiet,
                      level=args.loglevel)  # pylint: disable=no-member

    utils.log_basic_info(args)

    use_cpu = args.use_cpu
    if not pt.cuda.is_available():
        logger.info("CUDA not available, using cpu")
        use_cpu = True
    device = pt.device('cpu') if use_cpu else pt.device('cuda', args.device_id)
    logger.info(f"Scoring device: {device}")

    model, source_vocabs, target_vocabs = load_model(args.model, device=device, dtype=args.dtype)
    model.eval()

    max_seq_len_source = model.max_supported_len_source
    max_seq_len_target = model.max_supported_len_target
    if args.max_seq_len is not None:
        max_seq_len_source = min(args.max_seq_len[0] + C.SPACE_FOR_XOS, max_seq_len_source)
        max_seq_len_target = min(args.max_seq_len[1] + C.SPACE_FOR_XOS, max_seq_len_target)

    sources = [args.source] + args.source_factors
    sources = [str(os.path.abspath(source)) for source in sources]
    targets = [args.target] + args.target_factors
    targets = [str(os.path.abspath(target)) for target in targets]

    check_condition(len(targets) == model.num_target_factors,
                    "Number of target inputs/factors provided (%d) does not match number of target factors "
                    "required by the model (%d)" % (len(targets), model.num_target_factors))

    dumper = StateDumper(model, source_vocabs, target_vocabs, args.dump_path, max_seq_len_source, max_seq_len_target, device)
    dumper.init_dump_file(StateDumper.probe_token_count(targets[0]))  # TODO: assuming targets[0] is the text file, the rest are factors
    dumper.build_states_and_dump(sources, targets, args.batch_size)


if __name__ == "__main__":
    main()

# Copyright 2017--2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import logging

from . import arguments
from . import constants as C
from . import data_io
from . import utils
from . import vocab
from .log import setup_main_logger

logger = logging.getLogger(__name__)


def main():
    params = argparse.ArgumentParser(description='Preprocesses and shards monolingual data.')
    arguments.add_prepare_mono_data_cli_args(params)
    args = params.parse_args()
    prepare_data(args)


def prepare_data(args: argparse.Namespace):
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)
    setup_main_logger(console=not args.quiet,
                      file_logging=not args.no_logfile,
                      path=os.path.join(output_folder, C.LOG_NAME))
    utils.log_basic_info(args)
    utils.seed_rngs(args.seed)

    minimum_num_shards = args.min_num_shards
    samples_per_shard = args.num_samples_per_shard
    bucket_width = args.bucket_width

    mono_data = [args.monolingual] + args.monolingual_factors
    mono_data = [str(os.path.abspath(data)) for data in mono_data]

    num_words = args.mono_num_words if args.mono_num_words > 0 else None

    factor_vocab_paths = [args.mono_factor_vocabs[i] if i < len(args.mono_factor_vocabs)
                          else None for i in range(len(args.monolingual_factors))]
    vocab_paths = [args.mono_vocab] + factor_vocab_paths

    word_min_count = args.mono_word_min_count

    max_seq_len, max_seq_len_other = args.max_seq_len
    utils.check_condition(max_seq_len == max_seq_len_other,
                          "The source and target maximum length must match for monolingual data preparation.")

    # The maximum length is the length before we add the BOS/EOS symbols
    max_seq_len = max_seq_len + C.SPACE_FOR_XOS
    logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: %d",
                max_seq_len)

    vocabs = vocab.load_or_create_mono_vocab(data_paths=mono_data, vocab_paths=vocab_paths,
                                             factor_vocab_same_as_main=args.mono_factors_use_mono_vocab,
                                             num_words=num_words, word_min_count=word_min_count,
                                             pad_to_multiple_of=args.mono_pad_vocab_to_multiple_of)

    data_io.prepare_mono_data(fnames=mono_data,
                              vocabs=vocabs,
                              vocab_paths=vocab_paths,
                              lang=args.monolingual_language,
                              max_seq_len=max_seq_len,
                              bucket_width=bucket_width,
                              samples_per_shard=samples_per_shard,
                              min_num_shards=minimum_num_shards,
                              output_prefix=output_folder,
                              max_processes=args.max_processes)


if __name__ == "__main__":
    main()

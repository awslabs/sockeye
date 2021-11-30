# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from . import arguments
from . import constants as C
from . import data_io_pt
from . import utils
from . import vocab
from .log import setup_main_logger

logger = logging.getLogger(__name__)


def main():
    params = arguments.ConfigArgumentParser(description='Preprocesses and shards training data.')
    arguments.add_prepare_data_cli_args(params)
    args = params.parse_args()
    prepare_data(args)


def prepare_data(args: argparse.Namespace):
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)
    setup_main_logger(console=not args.quiet,
                      file_logging=not args.no_logfile,
                      path=os.path.join(output_folder, C.LOG_NAME))
    utils.log_basic_info(args)
    arguments.save_args(args, os.path.join(output_folder, C.ARGS_STATE_NAME))
    utils.seed_rngs(args.seed)

    minimum_num_shards = args.min_num_shards
    samples_per_shard = args.num_samples_per_shard
    bucketing = not args.no_bucketing
    bucket_width = args.bucket_width
    bucket_scaling = args.bucket_scaling

    source_paths = [args.source] + args.source_factors
    source_factor_vocab_paths = [args.source_factor_vocabs[i] if i < len(args.source_factor_vocabs)
                                 else None for i in range(len(args.source_factors))]
    source_vocab_paths = [args.source_vocab] + source_factor_vocab_paths
    target_paths = [args.target] + args.target_factors
    target_factor_vocab_paths = [args.target_factor_vocabs[i] if i < len(args.target_factor_vocabs)
                                 else None for i in range(len(args.target_factors))]
    target_vocab_paths = [args.target_vocab] + target_factor_vocab_paths

    num_words_source, num_words_target = args.num_words
    num_words_source = num_words_source if num_words_source > 0 else None
    num_words_target = num_words_target if num_words_target > 0 else None

    word_min_count_source, word_min_count_target = args.word_min_count
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    # The maximum length is the length before we add the BOS/EOS symbols
    max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
    max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
    logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)",
                max_seq_len_source, max_seq_len_target)

    # Split input into shards and randomly assign data to shards
    with utils.smart_open(source_paths[0], mode='rb') as infile:
        num_sents = sum(1 for _ in infile)
    num_shards = data_io_pt.get_num_shards(num_sents, samples_per_shard, minimum_num_shards)
    logger.info("%d samples will be split into %d shard(s) (requested samples/shard=%d, min_num_shards=%d)."
                % (num_sents, num_shards, samples_per_shard, minimum_num_shards))
    shards, keep_tmp_shard_files = data_io_pt.create_shards(source_fnames=source_paths,
                                                            target_fnames=target_paths,
                                                            num_shards=num_shards,
                                                            output_prefix=output_folder)
    shard_source_paths, shard_target_paths = [paths for paths in zip(*shards)]

    # Process shards in parallel using max_processes process
    with utils.create_pool(args.max_processes) as pool:
        source_vocabs, target_vocabs = vocab.load_or_create_vocabs(
            shard_source_paths=shard_source_paths,
            source_factor_vocab_same_as_source=args.source_factors_use_source_vocab,
            target_factor_vocab_same_as_target=args.target_factors_use_target_vocab,
            shard_target_paths=shard_target_paths,
            source_vocab_paths=source_vocab_paths,
            target_vocab_paths=target_vocab_paths,
            shared_vocab=args.shared_vocab,
            num_words_source=num_words_source,
            word_min_count_source=word_min_count_source,
            num_words_target=num_words_target,
            word_min_count_target=word_min_count_target,
            pad_to_multiple_of=args.pad_vocab_to_multiple_of,
            mapper=pool.map)

        data_io_pt.prepare_data(source_fnames=source_paths,
                                target_fnames=target_paths,
                                source_vocabs=source_vocabs,
                                target_vocabs=target_vocabs,
                                source_vocab_paths=source_vocab_paths,
                                target_vocab_paths=[args.target_vocab],
                                shared_vocab=args.shared_vocab,
                                max_seq_len_source=max_seq_len_source,
                                max_seq_len_target=max_seq_len_target,
                                bucketing=bucketing,
                                bucket_width=bucket_width,
                                num_shards=num_shards,
                                output_prefix=output_folder,
                                bucket_scaling=bucket_scaling,
                                pool=pool,
                                shards=shards,
                                keep_tmp_shard_files=keep_tmp_shard_files)


if __name__ == "__main__":
    main()

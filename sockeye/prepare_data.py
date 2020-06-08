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
    params = argparse.ArgumentParser(description='Preprocesses and shards training data.')
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

    source_vocabs, target_vocab = vocab.load_or_create_vocabs(
        source_paths=source_paths,
        factor_vocab_same_as_source=args.source_factors_use_source_vocab,
        target_path=args.target,
        source_vocab_paths=source_vocab_paths,
        target_vocab_path=args.target_vocab,
        shared_vocab=args.shared_vocab,
        num_words_source=num_words_source,
        word_min_count_source=word_min_count_source,
        num_words_target=num_words_target,
        word_min_count_target=word_min_count_target,
        pad_to_multiple_of=args.pad_vocab_to_multiple_of)

    data_io.prepare_data(source_fnames=source_paths,
                         target_fname=args.target,
                         source_vocabs=source_vocabs,
                         target_vocab=target_vocab,
                         source_vocab_paths=source_vocab_paths,
                         target_vocab_path=args.target_vocab,
                         shared_vocab=args.shared_vocab,
                         max_seq_len_source=max_seq_len_source,
                         max_seq_len_target=max_seq_len_target,
                         bucketing=bucketing,
                         bucket_width=bucket_width,
                         samples_per_shard=samples_per_shard,
                         min_num_shards=minimum_num_shards,
                         output_prefix=output_folder,
                         bucket_scaling=bucket_scaling,
                         max_processes=args.max_processes)


if __name__ == "__main__":
    main()

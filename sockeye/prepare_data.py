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

import argparse
import os

from . import arguments
from . import constants as C
from . import data_io
from . import utils
from . import vocab
from .log import setup_main_logger

logger = setup_main_logger(__name__, file_logging=False, console=True)


def main():
    params = argparse.ArgumentParser(description='Preprocesses and shards training data.')
    arguments.add_prepare_data_cli_args(params)
    args = params.parse_args()

    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)
    global logger
    logger = setup_main_logger(__name__, file_logging=True, path=os.path.join(output_folder, C.LOG_NAME))

    utils.seedRNGs(args.seed)

    minimum_num_shards = args.min_num_shards
    samples_per_shard = args.num_samples_per_shard
    bucketing = not args.no_bucketing
    bucket_width = args.bucket_width

    shared_vocab = args.shared_vocab
    vocab_source_path = args.source_vocab
    vocab_target_path = args.target_vocab
    num_words_source, num_words_target = args.num_words
    word_min_count_source, word_min_count_target = args.word_min_count
    max_len_source, max_len_target = args.max_seq_len

    vocab_source, vocab_target = vocab.load_or_create_vocabs(source=args.source,
                                                             target=args.target,
                                                             source_vocab_path=args.source_vocab,
                                                             target_vocab_path=args.target_vocab,
                                                             shared_vocab=args.shared_vocab,
                                                             num_words_source=num_words_source,
                                                             word_min_count_source=word_min_count_source,
                                                             num_words_target=num_words_target,
                                                             word_min_count_target=word_min_count_target)

    data_io.prepare_data(args.source, args.target,
                         vocab_source, vocab_target,
                         vocab_source_path, vocab_target_path,
                         shared_vocab,
                         max_len_source,
                         max_len_target,
                         bucketing=bucketing,
                         bucket_width=bucket_width,
                         samples_per_shard=samples_per_shard,
                         min_num_shards=minimum_num_shards,
                         output_prefix=output_folder)


if __name__ == "__main__":
    main()


# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
Simple Training CLI.
"""
import argparse
import os
import sys
from contextlib import ExitStack
from typing import Any, cast, Optional, Dict, List, Tuple

import mxnet as mx

from . import arguments
from . import constants as C
from . import data_io
from . import inference
from . import model
from . import scoring
from . import train
from . import utils
from .log import setup_main_logger
from .output_handler import get_output_handler, OutputHandler
from .utils import check_condition, log_basic_info

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = setup_main_logger(__name__, file_logging=False, console=True)


def main():
    params = arguments.ConfigArgumentParser(description='Score data with an existing model.')
    arguments.add_score_cli_args(params)
    args = params.parse_args()
    score(args)


def score(args: argparse.Namespace):
    global logger
    logger = setup_main_logger(__name__, file_logging=False)

    utils.log_basic_info(args)

    with ExitStack() as exit_stack:
        context = utils.determine_context(device_ids=args.device_ids,
                                          use_cpu=args.use_cpu,
                                          disable_device_locking=args.disable_device_locking,
                                          lock_dir=args.lock_dir,
                                          exit_stack=exit_stack)
        if args.batch_type == C.BATCH_TYPE_SENTENCE:
            check_condition(args.batch_size % len(context) == 0, "When using multiple devices the batch size must be "
                                                                 "divisible by the number of devices. Choose a batch "
                                                                 "size that is a multiple of %d." % len(context))
        logger.info("Scoring Device(s): %s", ", ".join(str(c) for c in context))

        model_config = model.SockeyeModel.load_config(os.path.join(args.model, C.CONFIG_NAME))
        max_seq_len_source = model_config.config_data.max_seq_len_source
        max_seq_len_target = model_config.config_data.max_seq_len_target

        score_iter, _, config_data, source_vocabs, target_vocab, data_info = train.create_data_iters_and_vocabs(
            args=args,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            shared_vocab=args.shared_vocab,
            resume_training=True,
            output_folder=args.model,
            bucketing=False,
            fill_up='zeros',
            no_permute=True)

        scoring_model = scoring.ScoringModel(config=model_config,
                                             model_dir=args.model,
                                             context=context,
                                             provide_data=score_iter.provide_data,
                                             provide_label=score_iter.provide_label,
                                             default_bucket_key=score_iter.default_bucket_key,
                                             score_type=args.score_type,
                                             bucketing=False,
                                             length_penalty=inference.LengthPenalty(alpha=args.length_penalty_alpha,
                                                                                    beta=args.length_penalty_beta))

        scorer = scoring.Scorer(scoring_model, source_vocabs, target_vocab)

        scorer.score(score_iter=score_iter,
                     score_type=args.score_type,
                     output_handler=get_output_handler(args.output_type))


if __name__ == "__main__":
    main()

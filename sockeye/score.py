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
import shutil
import sys
import tempfile
from contextlib import ExitStack
from typing import Any, cast, Optional, Dict, List, Tuple

import mxnet as mx

from . import arguments
from . import checkpoint_decoder
from . import constants as C
from . import convolution
from . import coverage
from . import data_io
from . import decoder
from . import encoder
from . import inference
from . import initializer
from . import loss
from . import lr_scheduler
from . import model
from . import rnn
from . import rnn_attention
from . import scoring
from . import train
from . import training
from . import transformer
from . import utils
from . import vocab
from .config import Config
from .log import setup_main_logger
from .optimizers import OptimizerConfig
from .utils import check_condition, log_basic_info

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = setup_main_logger(__name__, file_logging=False, console=True)


def check_arg_compatibility(args: argparse.Namespace):
    pass


def create_scoring_model(config: model.ModelConfig,
                         model_dir: str,
                         context: List[mx.Context],
                         score_iter: data_io.BaseParallelSampleIter,
                         bucketing: bool = False) -> scoring.ScoringModel:
    """
    Create a scoring model and load the parameters from disk if needed.

    :param config: The configuration for the model.
    :param context: The context(s) to run on.
    :param output_dir: Output folder.
    :param train_iter: The training data iterator.
    :param args: Arguments as returned by argparse.
s    :return: The training model.
    """
    scoring_model = scoring.ScoringModel(config=config,
                                         model_dir=model_dir,
                                         context=context,
                                         provide_data=score_iter.provide_data,
                                         default_bucket_key=score_iter.default_bucket_key,
                                         bucketing=bucketing)

    return scoring_model

def main():
    params = arguments.ConfigArgumentParser(description='Score data with an existing model.')
    arguments.add_score_cli_args(params)
    args = params.parse_args()
    score(args)


def score(args: argparse.Namespace):
    check_arg_compatibility(args)

    global logger
    logger = setup_main_logger(__name__, file_logging=False)

    utils.log_basic_info(args)

    max_seq_len_source, max_seq_len_target = args.max_seq_len
    # The maximum length is the length before we add the BOS/EOS symbols
    max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
    max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
    logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)",
                max_seq_len_source, max_seq_len_target)

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

        score_iter, _, config_data, source_vocabs, target_vocab, data_info = train.create_data_iters_and_vocabs(
            args=args,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            shared_vocab=args.shared_vocab,
            resume_training=True,
            output_folder=args.model,
            fill_up='zeros',
            no_permute=True)

        max_seq_len_source = config_data.max_seq_len_source
        max_seq_len_target = config_data.max_seq_len_target

        model_config = model.SockeyeModel.load_config(os.path.join(args.model, C.CONFIG_NAME))

        scoring_model = create_scoring_model(config=model_config,
                                             model_dir=args.model,
                                             context=context,
                                             bucketing=not args.no_bucketing,
                                             score_iter=score_iter)

        scorer = scoring.Scorer(scoring_model, source_vocabs, target_vocab,
                                length_penalty=inference.LengthPenalty(alpha=args.length_penalty_alpha,
                                                                       beta=args.length_penalty_beta))

        scorer.score(score_iter=score_iter)

if __name__ == "__main__":
    main()

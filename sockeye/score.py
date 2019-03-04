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
Scoring CLI.
"""
import argparse
import logging
import os
from contextlib import ExitStack
from typing import Optional, List, Tuple

from . import arguments
from . import constants as C
from . import data_io
from . import inference
from . import model
from . import scoring
from . import utils
from . import vocab
from .log import setup_main_logger
from .output_handler import get_output_handler
from .utils import check_condition

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = logging.getLogger(__name__)


def main():
    params = arguments.ConfigArgumentParser(description='Score data with an existing model.')
    arguments.add_score_cli_args(params)
    args = params.parse_args()
    setup_main_logger(file_logging=False, console=True, level=args.loglevel)  # pylint: disable=no-member
    score(args)


def get_data_iters_and_vocabs(args: argparse.Namespace,
                              model_folder: Optional[str]) -> Tuple['data_io.BaseParallelSampleIter',
                                                                    List[vocab.Vocab], vocab.Vocab, model.ModelConfig]:
    """
    Loads the data iterators and vocabularies.

    :param args: Arguments as returned by argparse.
    :param model_folder: Output folder.
    :return: The scoring data iterator as well as the source and target vocabularies.
    """

    model_config = model.SockeyeModel.load_config(os.path.join(args.model, C.CONFIG_NAME))

    if args.max_seq_len is None:
        max_seq_len_source = model_config.config_data.max_seq_len_source
        max_seq_len_target = model_config.config_data.max_seq_len_target
    else:
        max_seq_len_source, max_seq_len_target = args.max_seq_len

    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)

    # Load the existing vocabs created when starting the training run.
    source_vocabs = vocab.load_source_vocabs(model_folder)
    target_vocab = vocab.load_target_vocab(model_folder)

    sources = [args.source] + args.source_factors
    sources = [str(os.path.abspath(source)) for source in sources]

    score_iter = data_io.get_scoring_data_iters(
        sources=sources,
        target=os.path.abspath(args.target),
        source_vocabs=source_vocabs,
        target_vocab=target_vocab,
        batch_size=args.batch_size,
        batch_num_devices=batch_num_devices,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target)

    return score_iter, source_vocabs, target_vocab, model_config


def score(args: argparse.Namespace):

    setup_main_logger(file_logging=False, console=not args.quiet)

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

        # This call has a number of different parameters compared to training which reflect our need to get scores
        # one-for-one and in the same order as the input data.
        # To enable code reuse, we stuff the `args` parameter with some values.
        # Bucketing and permuting need to be turned off in order to preserve the ordering of sentences.
        # Finally, 'resume_training' needs to be set to True because it causes the model to be loaded instead of initialized.
        args.no_bucketing = True
        args.bucket_width = 10
        score_iter, source_vocabs, target_vocab, model_config = get_data_iters_and_vocabs(
            args=args,
            model_folder=args.model)

        scoring_model = scoring.ScoringModel(config=model_config,
                                             model_dir=args.model,
                                             context=context,
                                             provide_data=score_iter.provide_data,
                                             provide_label=score_iter.provide_label,
                                             default_bucket_key=score_iter.default_bucket_key,
                                             score_type=args.score_type,
                                             length_penalty=inference.LengthPenalty(alpha=args.length_penalty_alpha,
                                                                                    beta=args.length_penalty_beta),
                                             softmax_temperature=args.softmax_temperature)

        scorer = scoring.Scorer(scoring_model, source_vocabs, target_vocab)

        scorer.score(score_iter=score_iter,
                     output_handler=get_output_handler(output_type=args.output_type,
                                                       output_fname=args.output))


if __name__ == "__main__":
    main()

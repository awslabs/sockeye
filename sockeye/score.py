# Copyright 2018--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from . import arguments
from . import constants as C
from . import data_io
from . import scoring
from . import utils
from .beam_search import CandidateScorer
from .log import setup_main_logger
from .model import load_model
from .output_handler import get_output_handler
from .utils import check_condition

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = logging.getLogger(__name__)


def main():
    params = arguments.ConfigArgumentParser(description='Score data with an existing model.')
    arguments.add_score_cli_args(params)
    args = params.parse_args()
    score(args)


def score(args: argparse.Namespace):
    setup_main_logger(file_logging=False,
                      console=not args.quiet,
                      level=args.loglevel)  # pylint: disable=no-member

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

        model, source_vocabs, target_vocab = load_model(args.model, context=context, dtype=args.dtype)

        max_seq_len_source = model.max_supported_len_source
        max_seq_len_target = model.max_supported_len_target
        if args.max_seq_len is not None:
            max_seq_len_source = min(args.max_seq_len[0] + C.SPACE_FOR_XOS, max_seq_len_source)
            max_seq_len_target = min(args.max_seq_len[1] + C.SPACE_FOR_XOS, max_seq_len_target)

        hybridize = not args.no_hybridization

        sources = [args.source] + args.source_factors
        sources = [str(os.path.abspath(source)) for source in sources]
        target = os.path.abspath(args.target)

        score_iter = data_io.get_scoring_data_iters(
            sources=sources,
            target=target,
            source_vocabs=source_vocabs,
            target_vocab=target_vocab,
            batch_size=args.batch_size,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target)

        constant_length_ratio = args.brevity_penalty_constant_length_ratio
        if args.brevity_penalty_type == C.BREVITY_PENALTY_CONSTANT:
            if constant_length_ratio <= 0.0:
                constant_length_ratio = model.length_ratio_mean
                logger.info("Using constant length ratio saved in the model config: %f", constant_length_ratio)
        else:
            constant_length_ratio = -1.0

        batch_scorer = scoring.BatchScorer(scorer=CandidateScorer(length_penalty_alpha=args.length_penalty_alpha,
                                                                  length_penalty_beta=args.length_penalty_beta,
                                                                  brevity_penalty_weight=args.brevity_penalty_weight),
                                           score_type=args.score_type,
                                           constant_length_ratio=constant_length_ratio)
        if hybridize:
            batch_scorer.hybridize(static_alloc=True)

        scorer = scoring.Scorer(model=model,
                                batch_scorer=batch_scorer,
                                source_vocabs=source_vocabs,
                                target_vocab=target_vocab,
                                context=context)

        scorer.score(score_iter=score_iter,
                     output_handler=get_output_handler(output_type=args.output_type,
                                                       output_fname=args.output))


if __name__ == "__main__":
    main()

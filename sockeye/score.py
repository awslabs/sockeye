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

"""
Scoring CLI.
"""
import argparse
import logging
import time
from contextlib import ExitStack
from typing import List, DefaultDict


from . import arguments
from . import constants as C
from . import utils
from . import data_io
from . import translate
from . import scoring

from sockeye.log import setup_main_logger
from sockeye.utils import check_condition
from sockeye.output_handler import ScoreOutputHandler, OutputHandler

logger = setup_main_logger(__name__, file_logging=False)


def score(output_handler: OutputHandler,
          models: List[scoring.ScoringModel],
          data_iters: List[data_io.BaseParallelSampleIter],
          mapids: DefaultDict[DefaultDict[int, int]],
          scorer: scoring.Scorer) -> None:
    """
    """
    logger.info("Scoring...")

    tic = time.time()
    scored_outputs = scorer.score(models=models,
                                  data_iters=data_iters,
                                  mapids=mapids)

    for scored_output in scored_outputs:
        output_handler.handle(scored_output)

    total_time = time.time() - tic

    logger.info("Total time taken for scoring: %.4f", total_time)


def main():
    # TODO: test with factors
    params = argparse.ArgumentParser(description='Scoring CLI')

    arguments.add_scoring_args(params)
    args = params.parse_args()

    global logger
    if args.output is not None:
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))
    else:
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=False)
    utils.log_basic_info(args)

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")

    with ExitStack() as exit_stack:
        context = translate._setup_context(args, exit_stack)

        # if --max-seq-len given, use this, else get maximum sentence length from test data
        if(args.max_seq_len is not None):
            max_len_source, max_len_target = args.max_seq_len
        else:
            max_len_source, max_len_target = scoring.get_max_source_and_target(args)
        logger.info("Using max length source %d, max length target %d", max_len_source, max_len_target)

        # create iterator for each model (vocabularies can be different)
        data_iters, configs, mapids = [], [], []
        for model in args.models:
            data_iter, config, mapid = scoring.create_data_iter_and_vocab(args=args,
                                                                          max_seq_len_source=max_len_source,
                                                                          max_seq_len_target=max_len_target,
                                                                          model_dir=model)
            data_iters.append(data_iter)
            configs.append(config)
            mapids.append(mapid)

        models = scoring.load_models(context=context,
                                     batch_size=args.batch_size,
                                     model_folders=args.models,
                                     data_iters=data_iters,
                                     configs=configs,
                                     checkpoints=args.checkpoints,
                                     bucketing=not args.no_bucketing)

        scorer = scoring.Scorer(batch_size=args.batch_size,
                                no_bucketing=args.no_bucketing)

        output_handler = ScoreOutputHandler

        score(output_handler=output_handler,
              models=models,
              data_iters=data_iters,
              mapids=mapids,
              scorer=scorer)


if __name__ == '__main__':
    main()

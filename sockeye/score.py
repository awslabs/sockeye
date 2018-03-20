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
from contextlib import ExitStack
import logging


from . import arguments
from . import constants as C
from . import utils
from . import translate
from . import scoring

from sockeye.log import setup_main_logger
from sockeye.utils import check_condition
from sockeye.output_handler import get_output_handler, OutputHandler

logger = setup_main_logger(__name__, file_logging=False)

def score():
    pass

def read_and_score():
    pass

def main():
    # TODO adapt output_handler?
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
                                                           max_seq_len_source=max_len_source, max_seq_len_target=max_len_target, model_dir=model)
            data_iters.append(data_iter)
            configs.append(config)
            mapids.append(mapid)

        models = scoring.load_models(context,
             args.batch_size,
             args.models,
             data_iters,
             configs,
             args.checkpoints,
             not args.no_bucketing)

        results = []
        for model, data_iter, mapid in zip(models, data_iters, mapids):
            result = model.score(data_iter, mapid, args.batch_size, args.no_bucketing)
            results.append(result)

        for sentence_id in range(len(results[0])):
            for model in range(len(results)):
                print("model {} : score {}".format(model, results[model][sentence_id]), end=" ")
            print()


if __name__ == '__main__':
    main()

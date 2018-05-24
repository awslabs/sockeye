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
Scoring CLI. Used to determine the model score of an input force-decoded to a specified output.
This output can be provided in two ways. If input is read from STDIN, each desired output should
follow the input, separated by a tab. If reading from files, the last file presented should contain
the name of the targets.

No <s> or </s> tokens should be added since that is done internally automatically.
"""
import argparse
import sys
import time
from contextlib import ExitStack
from typing import Generator, Optional, List

import mxnet as mx
from math import ceil

from sockeye.log import setup_main_logger
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import acquire_gpus, get_num_gpus, log_basic_info, check_condition, grouper
from . import arguments
from . import constants as C
from . import data_io
from . import inference
from .translate import _setup_context

logger = setup_main_logger(__name__, file_logging=False)


def main():
    params = argparse.ArgumentParser(description='Scoring CLI')
    arguments.add_score_cli_args(params)
    args = params.parse_args()

    if args.output is not None:
        global logger
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")

    logger.info('Forcing beam_size=1, beam_prune=0')
    args.beam_size = 1
    args.beam_prune = 0

    log_basic_info(args)

    output_handler = get_output_handler(args.output_type,
                                        args.output,
                                        args.sure_align_threshold)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

        models, source_vocabs, target_vocab = inference.load_models(
            context=context,
            max_input_len=args.max_input_len,
            beam_size=1,
            batch_size=args.batch_size,
            model_folders=args.models,
            checkpoints=args.checkpoints,
            softmax_temperature=args.softmax_temperature,
            max_output_length_num_stds=args.max_output_length_num_stds,
            decoder_return_logit_inputs=False,
            cache_output_layer_w_b=False)
        translator = inference.Translator(context=context,
                                          ensemble_mode=args.ensemble_mode,
                                          bucket_source_width=args.bucket_width,
                                          length_penalty=inference.LengthPenalty(args.length_penalty_alpha,
                                                                                 args.length_penalty_beta),
                                          beam_prune=args.beam_prune,
                                          beam_search_stop=args.beam_search_stop,
                                          models=models,
                                          source_vocabs=source_vocabs,
                                          target_vocab=target_vocab,
                                          restrict_lexicon=None,
                                          store_beam=False,
                                          strip_unknown_words=False)
        read_and_translate(translator=translator,
                           output_handler=output_handler,
                           chunk_size=args.chunk_size,
                           input_file=args.input,
                           input_factors=args.input_factors)


def make_inputs(input_file: Optional[str],
                translator: inference.Translator,
                input_factors: Optional[List[str]] = None) -> Generator[inference.TranslatorInput, None, None]:
    """
    Generates TranslatorInput instances from input. If input is None, reads from stdin. If num_input_factors > 1,
    the function will look for factors attached to each token, separated by '|'. A second field (tab-delimited) will
    contain the target.

    If source is not None, reads from the source file. If num_source_factors > 1, num_source_factors source factor
    filenames are required. In this case, the last file name given will contain the target

    :param input_file: The source file (possibly None).
    :param translator: Translator that will translate each line of input.
    :param input_factors: Source factor files.
    :return: TranslatorInput objects.
    """
    if input_file is not None:
        logger.warn('Scoring only supports input on STDIN')

    check_condition(input_factors is None, "Translating from STDIN, not expecting any factor files.")
    for sentence_id, line in enumerate(sys.stdin, 1):
        input_str, target_str = line.split('\t')
        if not target_str:
            logger.warn('Sentence %d has no target', sentence_id)
        item = inference.make_input_from_factored_string(sentence_id=sentence_id,
                                                         factored_string=input_str,
                                                         translator=translator)
        item.target_tokens = list(data_io.get_tokens(target_str))
        yield item


def read_and_translate(translator: inference.Translator,
                       output_handler: OutputHandler,
                       chunk_size: Optional[int],
                       input_file: Optional[str] = None,
                       input_factors: Optional[List[str]] = None) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param chunk_size: The size of the portion to read at a time from the input.
    :param input_file: Optional path to file which will be translated line-by-line if included, if none use stdin.
    :param input_factors: Optional list of paths to files that contain source factors.
    """
    batch_size = translator.batch_size
    if chunk_size is None:
        if translator.batch_size == 1:
            # No batching, therefore there is not need to read segments in chunks.
            chunk_size = C.CHUNK_SIZE_NO_BATCHING
        else:
            # Get a constant number of batches per call to Translator.translate.
            chunk_size = C.CHUNK_SIZE_PER_BATCH_SEGMENT * translator.batch_size
    else:
        if chunk_size < translator.batch_size:
            logger.warning("You specified a chunk size (%d) smaller than the batch size (%d). This will lead to "
                           "a reduction in translation speed. Consider choosing a larger chunk size." % (chunk_size,
                                                                                                         batch_size))

    logger.info("Scoring...")

    total_time, total_lines = 0.0, 0
    for inputs in grouper(make_inputs(input_file, translator, input_factors), size=chunk_size):
        score_time = score(output_handler, inputs, translator)
        total_lines += len(inputs)
        total_time += score_time

    if total_lines != 0:
        logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                    total_lines, ceil(total_lines / batch_size), total_time,
                    total_time / total_lines, total_lines / total_time)
    else:
        logger.info("Processed 0 lines.")


def score(output_handler: OutputHandler,
          trans_inputs: List[inference.TranslatorInput],
          translator: inference.Translator) -> float:
    """
    Translates each line from source_data, calling output handler after translating a batch.

    :param output_handler: A handler that will be called once with the output of each translation.
    :param trans_inputs: A enumerable list of translator inputs.
    :param translator: The translator that will be used for each line of input.
    :return: Total time taken.
    """
    tic = time.time()
    trans_outputs = translator.score(trans_inputs)
    total_time = time.time() - tic
    batch_time = total_time / len(trans_inputs)
    for trans_input, trans_output in zip(trans_inputs, trans_outputs):
        output_handler.handle(trans_input, trans_output, batch_time)
    return total_time


if __name__ == '__main__':
    main()

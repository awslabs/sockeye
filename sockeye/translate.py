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

"""
Translation CLI.
"""
# Called before importing mxnet or any module that imports mxnet
from . import initial_setup
initial_setup.handle_env_cli_arg()

import argparse
import logging
import sys
import time
from . import inference
from contextlib import ExitStack
from typing import Dict, Generator, List, Optional, Union

from sockeye.lexicon import TopKLexicon
from sockeye.log import setup_main_logger
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import determine_context, log_basic_info, check_condition, grouper
from . import arguments
from . import constants as C
from . import data_io

from . import utils

logger = logging.getLogger(__name__)


def main():
    params = arguments.ConfigArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    args = params.parse_args()
    run_translate(args)


def run_translate(args: argparse.Namespace):

    # Seed randomly unless a seed has been passed
    utils.seed_rngs(args.seed if args.seed is not None else int(time.time()))

    if args.output is not None:
        setup_main_logger(console=not args.quiet,
                          file_logging=not args.no_logfile,
                          path="%s.%s" % (args.output, C.LOG_NAME),
                          level=args.loglevel)
    else:
        setup_main_logger(file_logging=False, level=args.loglevel)

    log_basic_info(args)

    if args.nbest_size > 1:
        if args.output_type != C.OUTPUT_HANDLER_JSON:
            logger.warning("For nbest translation, you must specify `--output-type '%s'; overriding your setting of '%s'.",
                           C.OUTPUT_HANDLER_JSON, args.output_type)
            args.output_type = C.OUTPUT_HANDLER_JSON
    output_handler = get_output_handler(args.output_type,
                                        args.output)
    hybridize = not args.no_hybridization

    with ExitStack() as exit_stack:
        check_condition(len(args.device_ids) == 1, "translate only supports single device for now")
        context = determine_context(device_ids=args.device_ids,
                                    use_cpu=args.use_cpu,
                                    disable_device_locking=args.disable_device_locking,
                                    lock_dir=args.lock_dir,
                                    exit_stack=exit_stack)[0]
        logger.info("Translate Device: %s", context)
        from sockeye.model import load_models
        models, source_vocabs, target_vocabs = load_models(context=context,
                                                           model_folders=args.models,
                                                           checkpoints=args.checkpoints,
                                                           dtype=args.dtype,
                                                           hybridize=hybridize,
                                                           inference_only=True,
                                                           mc_dropout=args.mc_dropout)

        restrict_lexicon = None  # type: Optional[Union[TopKLexicon, Dict[str, TopKLexicon]]]
        if args.restrict_lexicon is not None:
            logger.info(str(args.restrict_lexicon))
            if len(args.restrict_lexicon) == 1:
                # Single lexicon used for all inputs.
                restrict_lexicon = TopKLexicon(source_vocabs[0], target_vocabs[0])
                # Handle a single arg of key:path or path (parsed as path:path)
                restrict_lexicon.load(args.restrict_lexicon[0][1], k=args.restrict_lexicon_topk)
            else:
                check_condition(args.json_input,
                                "JSON input is required when using multiple lexicons for vocabulary restriction")
                # Multiple lexicons with specified names
                restrict_lexicon = dict()
                for key, path in args.restrict_lexicon:
                    lexicon = TopKLexicon(source_vocabs[0], target_vocabs[0])
                    lexicon.load(path, k=args.restrict_lexicon_topk)
                    restrict_lexicon[key] = lexicon

        brevity_penalty_weight = args.brevity_penalty_weight
        if args.brevity_penalty_type == C.BREVITY_PENALTY_CONSTANT:
            if args.brevity_penalty_constant_length_ratio > 0.0:
                constant_length_ratio = args.brevity_penalty_constant_length_ratio
            else:
                constant_length_ratio = sum(model.length_ratio_mean for model in models) / len(models)
                logger.info("Using average of constant length ratios saved in the model configs: %f",
                            constant_length_ratio)
        elif args.brevity_penalty_type == C.BREVITY_PENALTY_LEARNED:
            constant_length_ratio = -1.0
        elif args.brevity_penalty_type == C.BREVITY_PENALTY_NONE:
            brevity_penalty_weight = 0.0
            constant_length_ratio = -1.0
        else:
            raise ValueError("Unknown brevity penalty type %s" % args.brevity_penalty_type)


        scorer = inference.CandidateScorer(
            length_penalty_alpha=args.length_penalty_alpha,
            length_penalty_beta=args.length_penalty_beta,
            brevity_penalty_weight=brevity_penalty_weight)

        translator = inference.Translator(context=context,
                                          ensemble_mode=args.ensemble_mode,
                                          scorer=scorer,
                                          batch_size=args.batch_size,
                                          beam_size=args.beam_size,
                                          beam_search_stop=args.beam_search_stop,
                                          nbest_size=args.nbest_size,
                                          models=models,
                                          source_vocabs=source_vocabs,
                                          target_vocabs=target_vocabs,
                                          restrict_lexicon=restrict_lexicon,
                                          avoid_list=args.avoid_list,
                                          strip_unknown_words=args.strip_unknown_words,
                                          sample=args.sample,
                                          output_scores=output_handler.reports_score(),
                                          constant_length_ratio=constant_length_ratio,
                                          max_output_length_num_stds=args.max_output_length_num_stds,
                                          max_input_length=args.max_input_length,
                                          max_output_length=args.max_output_length,
                                          hybridize=hybridize,
                                          softmax_temperature=args.softmax_temperature,
                                          prevent_unk=args.prevent_unk,
                                          greedy=args.greedy)
        read_and_translate(translator=translator,
                           output_handler=output_handler,
                           chunk_size=args.chunk_size,
                           input_file=args.input,
                           input_factors=args.input_factors,
                           input_is_json=args.json_input)


def make_inputs(input_file: Optional[str],
                translator: inference.Translator,
                input_is_json: bool,
                input_factors: Optional[List[str]] = None) -> Generator[inference.TranslatorInput, None, None]:
    """
    Generates TranslatorInput instances from input. If input is None, reads from stdin. If num_input_factors > 1,
    the function will look for factors attached to each token, separated by '|'.
    If source is not None, reads from the source file. If num_source_factors > 1, num_source_factors source factor
    filenames are required.

    :param input_file: The source file (possibly None).
    :param translator: Translator that will translate each line of input.
    :param input_is_json: Whether the input is in json format.
    :param input_factors: Source factor files.
    :return: TranslatorInput objects.
    """
    if input_file is None:
        check_condition(input_factors is None, "Translating from STDIN, not expecting any factor files.")
        for sentence_id, line in enumerate(sys.stdin, 1):
            if input_is_json:
                yield inference.make_input_from_json_string(sentence_id=sentence_id,
                                                            json_string=line,
                                                            translator=translator)
            else:
                yield inference.make_input_from_factored_string(sentence_id=sentence_id,
                                                                factored_string=line,
                                                                translator=translator)
    else:
        input_factors = [] if input_factors is None else input_factors
        inputs = [input_file] + input_factors
        if not input_is_json:
            check_condition(translator.num_source_factors == len(inputs),
                            "Model(s) require %d factors, but %d given (through --input and --input-factors)." % (
                                translator.num_source_factors, len(inputs)))
        with ExitStack() as exit_stack:
            streams = [exit_stack.enter_context(data_io.smart_open(i)) for i in inputs]
            for sentence_id, inputs in enumerate(zip(*streams), 1):
                if input_is_json:
                    yield inference.make_input_from_json_string(sentence_id=sentence_id,
                                                                json_string=inputs[0],
                                                                translator=translator)
                else:
                    yield inference.make_input_from_multiple_strings(sentence_id=sentence_id, strings=list(inputs))


def read_and_translate(translator: inference.Translator,
                       output_handler: OutputHandler,
                       chunk_size: Optional[int],
                       input_file: Optional[str] = None,
                       input_factors: Optional[List[str]] = None,
                       input_is_json: bool = False) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param chunk_size: The size of the portion to read at a time from the input.
    :param input_file: Optional path to file which will be translated line-by-line if included, if none use stdin.
    :param input_factors: Optional list of paths to files that contain source factors.
    :param input_is_json: Whether the input is in json format.
    """
    batch_size = translator.max_batch_size
    if chunk_size is None:
        if translator.max_batch_size == 1:
            # No batching, therefore there is not need to read segments in chunks.
            chunk_size = C.CHUNK_SIZE_NO_BATCHING
        else:
            # Get a constant number of batches per call to Translator.translate.
            chunk_size = C.CHUNK_SIZE_PER_BATCH_SEGMENT * translator.max_batch_size
    else:
        if chunk_size < translator.max_batch_size:
            logger.warning("You specified a chunk size (%d) smaller than the max batch size (%d). This will lead to "
                           "a reduction in translation speed. Consider choosing a larger chunk size." % (chunk_size,
                                                                                                         batch_size))

    logger.info("Translating...")

    total_time, total_lines = 0.0, 0
    for chunk in grouper(make_inputs(input_file, translator, input_is_json, input_factors), size=chunk_size):
        chunk_time = translate(output_handler, chunk, translator)
        total_lines += len(chunk)
        total_time += chunk_time

    if total_lines != 0:
        logger.info("Processed %d lines. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                    total_lines, total_time, total_time / total_lines, total_lines / total_time)
    else:
        logger.info("Processed 0 lines.")


def translate(output_handler: OutputHandler,
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
    trans_outputs = translator.translate(trans_inputs)
    total_time = time.time() - tic
    batch_time = total_time / len(trans_inputs)
    for trans_input, trans_output in zip(trans_inputs, trans_outputs):
        output_handler.handle(trans_input, trans_output, batch_time)  # type: ignore
    return total_time


if __name__ == '__main__':
    main()

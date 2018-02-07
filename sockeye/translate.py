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
Translation CLI.
"""
import argparse
import sys
import time
from contextlib import ExitStack
from typing import Generator, Optional, Iterable, List

import mxnet as mx
from math import ceil

from sockeye.lexicon import TopKLexicon
from sockeye.log import setup_main_logger
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import acquire_gpus, get_num_gpus, log_basic_info, check_condition, grouper
from . import arguments
from . import constants as C
from . import data_io
from . import inference

logger = setup_main_logger(__name__, file_logging=False)


def main():
    params = argparse.ArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    args = params.parse_args()

    if args.output is not None:
        global logger
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")

    log_basic_info(args)

    output_handler = get_output_handler(args.output_type,
                                        args.output,
                                        args.sure_align_threshold)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

        models, vocab_source, source_factor_vocabs, vocab_target = inference.load_models(
            context,
            args.max_input_len,
            args.beam_size,
            args.batch_size,
            args.models,
            args.checkpoints,
            args.softmax_temperature,
            args.max_output_length_num_stds,
            decoder_return_logit_inputs=args.restrict_lexicon is not None,
            cache_output_layer_w_b=args.restrict_lexicon is not None)
        restrict_lexicon = None  # type: TopKLexicon
        if args.restrict_lexicon:
            restrict_lexicon = TopKLexicon(vocab_source, vocab_target)
            restrict_lexicon.load(args.restrict_lexicon)
        translator = inference.Translator(context,
                                          args.ensemble_mode,
                                          args.bucket_width,
                                          inference.LengthPenalty(args.length_penalty_alpha,
                                                                  args.length_penalty_beta),
                                          models,
                                          vocab_source,
                                          vocab_target,
                                          source_factor_vocabs,
                                          restrict_lexicon)
        read_and_translate(translator,
                           output_handler,
                           chunk_size=args.chunk_size,
                           source=args.input,
                           source_factors=args.input_factors)


def _make_input(source: Optional[str],
                source_factors: Optional[List[str]] = None,
                num_source_factors: int = 0) -> Generator[inference.TranslatorInput, None, None]:
    """
    Transform both accepted inputs (STDIN and `--input`, factored or not) into a stream of TranslatorInput objects.
    :param source: The source file (possibly None).
    :param source_factors: Source factor files
    :param num_source_factors: Number of source factors required by the model.
    :return: TranslatorInput objects
    """
    if source is None:
        check_condition(source_factors is None, "Translating from STDIN, not expecting any factor files.")
        for sentence_id, line in enumerate(sys.stdin, 1):
            yield inference.Translator.make_input(sentence_id=sentence_id, raw_sentence=line,
                                                  num_factors=num_source_factors)
    else:
        source_factors = [] if source_factors is None else source_factors
        check_condition(len(source_factors) == num_source_factors,
                        "Model(s) require(s) %d factor files, got %d" % (num_source_factors,
                                                                         len(source_factors)))
        with ExitStack() as exit_stack:
            streams = [exit_stack.enter_context(data_io.smart_open(x)) for x in [source] + source_factors]
            for sentence_id, (source, *factors) in enumerate(zip(*streams), 1):
                yield inference.Translator.make_input_multiple(sentence_id=sentence_id,
                                                               raw_sentence=source,
                                                               num_factors=num_source_factors,
                                                               raw_factors=factors)


def read_and_translate(translator: inference.Translator,
                       output_handler: OutputHandler,
                       chunk_size: Optional[int],
                       source: Optional[str] = None,
                       source_factors: Optional[List[str]] = None) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param chunk_size: The size of the portion to read at a time from the input.
    :param source: Optional path to file which will be translated line-by-line if included, if none use stdin.
    :param source_factors: Optional list of paths to files that contain source factors.
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

    logger.info("Translating...")

    total_time, total_lines = 0.0, 0
    for chunk in grouper(_make_input(source, source_factors, translator.num_source_factors), size=chunk_size):
        chunk_time = translate(output_handler, chunk, translator)
        total_lines += len(chunk)
        total_time += chunk_time

    if total_lines != 0:
        logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                    total_lines, ceil(total_lines / batch_size), total_time,
                    total_time / total_lines, total_lines / total_time)
    else:
        logger.info("Processed 0 lines.")


def translate(output_handler: OutputHandler, trans_inputs: List[inference.TranslatorInput],
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
        output_handler.handle(trans_input, trans_output, batch_time)
    return total_time


def _setup_context(args, exit_stack):
    if args.use_cpu:
        context = mx.cpu()
    else:
        num_gpus = get_num_gpus()
        check_condition(num_gpus >= 1,
                        "No GPUs found, consider running on the CPU with --use-cpu "
                        "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi "
                        "binary isn't on the path).")
        check_condition(len(args.device_ids) == 1, "cannot run on multiple devices for now")
        gpu_id = args.device_ids[0]
        if args.disable_device_locking:
            if gpu_id < 0:
                # without locking and a negative device id we just take the first device
                gpu_id = 0
        else:
            gpu_ids = exit_stack.enter_context(acquire_gpus([gpu_id], lock_dir=args.lock_dir))
            gpu_id = gpu_ids[0]

        context = mx.gpu(gpu_id)
    return context


if __name__ == '__main__':
    main()

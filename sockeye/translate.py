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
from math import ceil
from contextlib import ExitStack
from typing import Optional, Iterable, List, Dict

import mxnet as mx

import sockeye
import sockeye.arguments as arguments
import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
from sockeye.lexicon import TopKLexicon
import sockeye.output_handler
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus, log_basic_info
from sockeye.utils import check_condition, grouper


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

    output_handler = sockeye.output_handler.get_output_handler(args.output_type,
                                                               args.output,
                                                               args.sure_align_threshold)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

        models, vocab_source, vocab_target, source_factor_vocabs = sockeye.inference.load_models(
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
        restrict_lexicon = None # type: TopKLexicon
        if args.restrict_lexicon:
            restrict_lexicon = TopKLexicon(vocab_source, vocab_target)
            restrict_lexicon.load(args.restrict_lexicon)
        translator = sockeye.inference.Translator(context,
                                                  args.ensemble_mode,
                                                  args.bucket_width,
                                                  sockeye.inference.LengthPenalty(args.length_penalty_alpha,
                                                                                  args.length_penalty_beta),
                                                  models,
                                                  vocab_source,
                                                  vocab_target,
                                                  source_factor_vocabs,
                                                  restrict_lexicon)
        read_and_translate(translator, output_handler, args.chunk_size, args.input, args.input_factors)


def merge_factors(source: str,
                  source_factors: Optional[List[str]] = None):
    """
    This function merges multiple factor streams into a single stream with Moses-style factors present on each token.

    Background:
    Decoding from the command line can take input from either STDIN or from a file (`--input`).
    When adding source factors, the input format from STDIN is a sequence of tokens, with factors attached to each word
    in order and delimited by |, Moses style. If instead using `--input`, factors are passed via `--input-factors`,
    which will contain a list of token-parallel files.

    :param source: The source file.
    :param source_factors: A list of source factor files.
    :return: A single stream with the factors merged into the source stream.
    """

    num_streams = 1 + len(source_factors)

    with ExitStack() as exit_stack:
        fhs = [ sockeye.data_io.smart_open(x) for x in [source] + source_factors ]
        for factors in zip(*fhs):
            tokens = [f.rstrip().split() for f in factors]
            source_len = len(tokens[0])
            for i, toks in enumerate(tokens[1:]):
                check_condition(len(toks) == source_len,
                                'Factor {:d} has the wrong number of tokens (found {:d}, needed {:d})'.format(
                                    i, len(toks), source_len))

            # Single-line nested comprehensions are unreadable
            merged = []
            for i in range(source_len):
                merged.append(C.FACTOR_DELIM.join([tokens[j][i] for j in range(num_streams)]))

            yield ' '.join(merged)


def read_and_translate(translator: sockeye.inference.Translator, output_handler: sockeye.output_handler.OutputHandler,
                       chunk_size: Optional[int], source: Optional[str] = None, source_factors: Optional[List[str]] = []) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param chunk_size: The size of the portion to read at a time from the input.
    :param source: Path to file which will be translated line-by-line if included, if none use stdin.
    """
    source_data = sys.stdin if source is None else merge_factors(source, source_factors)

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
                           "a degregation of translation speed. Consider choosing a larger chunk size." % (chunk_size,
                                                                                                           batch_size))

    logger.info("Translating...")

    total_time, total_lines = 0.0, 0
    for chunk in grouper(source_data, chunk_size):
        chunk_time = translate(output_handler, chunk, translator, total_lines)
        total_lines += len(chunk)
        total_time += chunk_time

    if total_lines != 0:
        logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                    total_lines, ceil(total_lines / batch_size), total_time,
                    total_time / total_lines, total_lines / total_time)
    else:
        logger.info("Processed 0 lines.")


def translate(output_handler: sockeye.output_handler.OutputHandler, source_data: Iterable[Dict],
                    translator: sockeye.inference.Translator, chunk_id: int = 0) -> float:
    """
    Translates each line from source_data, calling output handler after translating a batch.

    :param output_handler: A handler that will be called once with the output of each translation.
    :param source_data: A enumerable list of source sentence JSON objects that will be translated.
    :param translator: The translator that will be used for each line of input.
    :param chunk_id: Global id of the chunk.
    :return: Total time taken.
    """

    tic = time.time()
    trans_inputs = [translator.make_input(i, line) for i, line in enumerate(source_data, chunk_id + 1)]
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

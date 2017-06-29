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
from typing import Optional, Iterable, Tuple

import mxnet as mx

import sockeye.arguments as arguments
import sockeye.data_io
import sockeye.inference
import sockeye.output_handler
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus

logger = setup_main_logger(__name__, file_logging=False)


def main():
    params = argparse.ArgumentParser(description='Translate CLI')
    arguments.add_inference_args(params)
    arguments.add_device_args(params)
    args = params.parse_args()

    assert args.beam_size > 0, "Beam size must be 1 or greater."
    if args.checkpoints is not None:
        assert len(args.checkpoints) == len(args.models), "must provide checkpoints for each model"

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    output_handler = sockeye.output_handler.get_output_handler(args.output_type,
                                                               args.output,
                                                               args.sure_align_threshold)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

        translator = sockeye.inference.Translator(context,
                                                  args.ensemble_mode,
                                                  *sockeye.inference.load_models(context,
                                                                                 args.max_input_len,
                                                                                 args.beam_size,
                                                                                 args.models,
                                                                                 args.checkpoints,
                                                                                 args.softmax_temperature))
        read_and_translate(translator, output_handler, args.input)


def read_and_translate(translator: sockeye.inference.Translator, output_handler: sockeye.output_handler.OutputHandler,
                       source: Optional[str] = None) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param source: Path to file which will be translated line-by-line if included, if none use stdin.
    """

    source_data = sys.stdin if source is None else sockeye.data_io.smart_open(source)

    i, total_time = translate_lines(output_handler, source_data, translator)

    if i != 0:
        logger.info("Processed %d lines. Total time: %.4f sec/sent: %.4f sent/sec: %.4f", i, total_time,
                    total_time / i, i / total_time)
    else:
        logger.info("Processed 0 lines.")


def translate_lines(output_handler: sockeye.output_handler.OutputHandler, source_data: Iterable[str],
                    translator: sockeye.inference.Translator) -> Tuple[int, float]:
    """
    Translates each line from source_data, calling output handler for each result.

    :param output_handler: A handler that will be called once with the output of each translation.
    :param source_data: A enumerable list of source sentences that will be translated.
    :param translator: The translator that will be used for each line of input.
    :return: The number of lines translated, and the total time taken.
    """

    i = 0
    total_time = 0.0
    for i, line in enumerate(source_data, 1):
        trans_input = translator.make_input(i, line)
        logger.debug(" IN: %s", trans_input)
        tic = time.time()
        trans_output = translator.translate(trans_input)
        trans_wall_time = time.time() - tic
        total_time += trans_wall_time
        logger.debug("OUT: %s", trans_output)
        logger.debug("OUT: time=%.2f", trans_wall_time)
        output_handler.handle(trans_input, trans_output)
    return i, total_time


def _setup_context(args, exit_stack):
    if args.use_cpu:
        context = mx.cpu()
    else:
        num_gpus = get_num_gpus()
        assert num_gpus > 0, "No GPUs found, consider running on the CPU with --use-cpu " \
                             "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi " \
                             "binary isn't on the path)."
        assert len(args.device_ids) == 1, "cannot run on multiple devices for now"
        gpu_id = args.device_ids[0]
        if args.disable_device_locking:
            # without locking and a negative device id we just take the first device
            gpu_id = 0
        else:
            if gpu_id < 0:
                # get a single (!) gpu id automatically:
                gpu_ids = exit_stack.enter_context(acquire_gpus([-1], lock_dir=args.lock_dir))
                gpu_id = gpu_ids[0]
        context = mx.gpu(gpu_id)
    return context


if __name__ == '__main__':
    main()

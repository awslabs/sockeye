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

import mxnet as mx

import sockeye.arguments as arguments
import sockeye.data_io
import sockeye.inference
import sockeye.output_handler
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpu, get_num_gpus


def main():
    params = argparse.ArgumentParser(description='Translate from STDIN to STDOUT')
    params = arguments.add_inference_args(params)
    params = arguments.add_device_args(params)
    args = params.parse_args()

    logger = setup_main_logger(__name__, file_logging=False)

    assert args.beam_size > 0, "Beam size must be 1 or greater."
    if args.checkpoints is not None:
        assert len(args.checkpoints) == len(args.models), "must provide checkpoints for each model"

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    output_stream = sys.stdout
    output_handler = sockeye.output_handler.get_output_handler(args.output_type,
                                                               output_stream,
                                                               args.align_plot_prefix,
                                                               args.sure_align_threshold)

    with ExitStack() as exit_stack:
        if args.use_cpu:
            context = mx.cpu()
        else:
            num_gpus = get_num_gpus()
            assert num_gpus > 0, "No GPUs found, consider running on the CPU with --use-cpu " \
                                 "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi " \
                                 "binary isn't on the path)."
            assert len(args.device_ids) == 1, "cannot run on multiple devices for now"
            gpu_id = args.device_ids[0]
            if gpu_id < 0:
                # get a gpu id automatically:
                gpu_id = exit_stack.enter_context(acquire_gpu())
            context = mx.gpu(gpu_id)

        translator = sockeye.inference.Translator(context,
                                                  args.ensemble_mode,
                                                  *sockeye.inference.load_models(context,
                                                                                 args.max_input_len,
                                                                                 args.beam_size,
                                                                                 args.models,
                                                                                 args.checkpoints,
                                                                                 args.softmax_temperature))
        total_time = 0
        i = 0
        for i, line in enumerate(sys.stdin, 1):
            trans_input = translator.make_input(i, line)
            logger.debug(" IN: %s", trans_input)

            tic = time.time()
            trans_output = translator.translate(trans_input)
            trans_wall_time = time.time() - tic
            total_time += trans_wall_time

            logger.debug("OUT: %s", trans_output)
            logger.debug("OUT: time=%.2f", trans_wall_time)

            output_handler.handle(trans_input, trans_output)

        logger.info("Processed %d lines. Total time: %.4f sec/sent: %.4f sent/sec: %.4f", i, total_time, total_time / i,
                    i / total_time)


if __name__ == '__main__':
    main()

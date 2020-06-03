# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import logging
import os

import sockeye.constants as C
from sockeye.log import setup_main_logger, log_sockeye_version
import sockeye.model
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


def annotate_model_params(model_dir: str):
    log_sockeye_version(logger)

    params_best = os.path.join(model_dir, C.PARAMS_BEST_NAME)
    params_best_float32 = os.path.join(model_dir, C.PARAMS_BEST_NAME_FLOAT32)
    config = os.path.join(model_dir, C.CONFIG_NAME)
    config_float32 = os.path.join(model_dir, C.CONFIG_NAME_FLOAT32)

    for fname in params_best_float32, config_float32:
        check_condition(not os.path.exists(fname),
                        'File "%s" exists, indicating this model has already been quantized.' % fname)

    # Load model and compute scaling factors
    model = sockeye.model.load_model(model_dir, for_disk_saving='float32', dtype='int8')
    # Move original params and config files
    os.rename(params_best, params_best_float32)
    os.rename(config, config_float32)
    # Write new params and config files with annotated scaling factors
    model[0].save_parameters(params_best)
    model[0].save_config(model_dir)


def main():
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(
        description='Annotate trained model with scaling factors for fast loading/quantization for int8 inference.')
    params.add_argument('--model', '-m', required=True, help='Trained Sockeye model directory.')
    args = params.parse_args()

    annotate_model_params(args.model)


if __name__ == '__main__':
    main()

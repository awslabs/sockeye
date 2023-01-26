# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Quantization CLI.
"""

import argparse
import logging
import os

import torch as pt

from . import arguments
from . import constants as C
from . import utils
from .log import setup_main_logger
from .model import load_model


logger = logging.getLogger(__name__)


def main():
    params = arguments.ConfigArgumentParser(description='Quantize an existing model.')
    arguments.add_quantize_args(params)
    arguments.add_logging_args(params)
    args = params.parse_args()
    quantize(args)


def quantize(args: argparse.Namespace):
    setup_main_logger(file_logging=False, console=not args.quiet, level=args.loglevel)
    utils.log_basic_info(args)

    params_fname = os.path.join(args.model, C.PARAMS_BEST_NAME)
    config_fname = os.path.join(args.model, C.CONFIG_NAME)

    model, _, _ = load_model(args.model, device=pt.device('cpu'))
    original_dtype = model.config.dtype

    if original_dtype == args.dtype:
        logger.info(f'Model already has dtype {args.dtype}. Skipping quantization.')
        return

    backup_params_fname = f'{params_fname}.{original_dtype}'
    backup_config_fname = f'{config_fname}.{original_dtype}'

    for fname in (backup_params_fname, backup_config_fname):
        if os.path.exists(fname):
            raise FileExistsError(f'File {fname} exists. To quantize this model, make sure that {params_fname} and '
                                  f'{config_fname} match the original dtype and that {backup_params_fname} and '
                                  f'{backup_config_fname} do not exist.')

    logger.info(f'Moving {params_fname} -> {backup_params_fname}')
    os.rename(params_fname, backup_params_fname)
    logger.info(f'Moving {config_fname} -> {backup_config_fname}')
    os.rename(config_fname, backup_config_fname)

    model.cast(args.dtype)

    model.save_parameters(params_fname)
    model.save_config(args.model)


if __name__ == '__main__':
    main()

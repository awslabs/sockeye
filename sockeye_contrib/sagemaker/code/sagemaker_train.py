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

import os
import logging
import shutil
import subprocess
import tempfile
from typing import Optional
import yaml

import sockeye.constants as C
from sockeye.log import setup_main_logger


INPUT_ARG_PREPARED_DATA = 'prepared_data'
INPUT_ARGS_SINGLE_VAL = [INPUT_ARG_PREPARED_DATA, 'source',  'target', 'validation_source', 'validation_target']
INPUT_ARGS_MULTIPLE_VALS = ['source_factors', 'target_factors',
                            'validation_source_factors', 'validation_target_factors']
OUTPUT_ARG = 'output'

SM_CHANNEL_PREFIX = 'SM_CHANNEL'
SM_OUTPUT = 'SM_OUTPUT_DATA_DIR'
SM_NUM_GPUS = 'SM_NUM_GPUS'

CONFIG_ARG = 'config'
CONFIG_FNAME = 'args.yaml'


logger = logging.getLogger(__name__)


def get_input_path(input_arg: str, fname: Optional[str]) -> Optional[str]:
    # Is a channel specified for this input?
    dirname = os.environ.get(f'{SM_CHANNEL_PREFIX}_{input_arg.upper()}', None)
    if dirname is not None:
        # Prepared data is a directory
        if input_arg == INPUT_ARG_PREPARED_DATA:
            return dirname
        # Other inputs are single files in dedicated directories
        if fname is None:
            return None
        return os.path.join(dirname, fname)
    # This input is not specified
    return None


def main():

    setup_main_logger(console=True, file_logging=False)

    # Load base config
    config_dirname = os.environ[f'{SM_CHANNEL_PREFIX}_{CONFIG_ARG.upper()}']
    with open(os.path.join(config_dirname, os.listdir(config_dirname)[0]), 'r') as inp:
        config = yaml.safe_load(inp)

    # Populate inputs from SageMaker environment variables and config basenames
    for input_arg in INPUT_ARGS_SINGLE_VAL:
        config[input_arg] = get_input_path(input_arg, config.get(input_arg, None))
    for input_arg in INPUT_ARGS_MULTIPLE_VALS:
        i = 0
        fnames = config.get(input_arg, [])
        while True:
            input_arg_i = f'{input_arg}_{i}'
            fname = get_input_path(input_arg_i, fnames[i] if i < len(fnames) else None)
            if fname is None:
                break
            fnames[i] = fname
            i += 1

    # Record final S3 output path and replace with intermediate output directory
    s3_output_path = config[OUTPUT_ARG]
    config[OUTPUT_ARG] = os.environ[SM_OUTPUT]

    # Write temporary config for SageMaker training
    with tempfile.TemporaryDirectory(prefix='sagemaker_train.') as tmp_dir:
        tmp_config = os.path.join(tmp_dir, CONFIG_FNAME)
        with open(tmp_config, 'w') as out:
            yaml.safe_dump(config, out, default_flow_style=False)

        num_gpus = int(os.environ[SM_NUM_GPUS])
        if num_gpus > 1:
            # Multiple GPUs: use torchrun to launch multiple sockeye-train
            # processes
            subprocess.run(['torchrun', '--nproc_per_node', os.environ[SM_NUM_GPUS],
                            '-m', 'sockeye.train', '--dist', '--config', tmp_config], check=True)
        else:
            # CPU or single GPU: launch a single sockeye-train process
            subprocess.run(['python', '-m', 'sockeye.train', '--config', tmp_config], check=True)

    # Copy best params file to avoid symlink issues
    params_best_fname = os.path.join(os.environ[SM_OUTPUT], C.PARAMS_BEST_NAME)
    params_best_actual_fname = os.path.join(os.environ[SM_OUTPUT], os.readlink(params_best_fname))
    os.remove(params_best_fname)
    shutil.copyfile(params_best_actual_fname, params_best_fname)

    logger.info(f'Trained model will be uploaded to: {s3_output_path}')


if __name__ == '__main__':
    main()

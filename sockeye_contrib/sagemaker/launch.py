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

import argparse
import datetime
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
import yaml

import sagemaker
from sagemaker.pytorch import PyTorch

from sockeye.log import setup_main_logger


CODE_DIRNAME = 'code'
SOCKEYE_DIRNAME = 'sockeye'
SOCKEYE_CONTRIB_DIRNAME = 'sockeye_contrib'
SOCKEYE_CONTRIB_FNAMES = ['rouge.py']
REQUIREMENTS_DIRNAME = 'requirements'
REQUIREMENTS_FNAME = 'requirements.txt'
SOURCE_DIR_ARCHIVE_FNAME = 'sourcedir.tar.gz'

INPUT_ARGS = ['prepared_data', 'source', 'source_factors', 'target', 'target_factors',
              'validation_source', 'validation_source_factors', 'validation_target', 'validation_target_factors']
OUTPUT_ARG = 'output'

CONFIG_ARG = 'config'
CONFIG_FNAME = 'args.yaml'

S3_PREFIX = 's3://'


logger = logging.getLogger(__name__)


def main():

    setup_main_logger(console=True, file_logging=False)

    params = argparse.ArgumentParser(description='Launch sockeye-train on Amazon SageMaker '
                                                 '(https://aws.amazon.com/sagemaker/).')
    params.add_argument('--config', '-c',
                        required=True,
                        help='Config file for sockeye-train.')
    params.add_argument('--job-prefix', '-j',
                        default='sockeye-train',
                        help='Prefix used for SageMaker job name and job/data S3 directories. Default: %(default)s.')
    params.add_argument('--instance-type', '-i',
                        default='ml.c5.2xlarge',
                        help='Training instance type. Default: %(default)s.')
    params.add_argument('--framework-version', '-f',
                        default='1.11.0',
                        help='Version of PyTorch. Default: %(default)s.')
    args = params.parse_args()

    # Set up SageMaker session and job
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = session.default_bucket()
    job_key = f'{args.job_prefix}-{datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")}-{uuid.uuid1()}'
    logger.info(f'SageMaker execution role: {role}')
    logger.info(f'SageMaker job prefix: {args.job_prefix}')
    logger.info(f'SageMaker job S3 path: {S3_PREFIX}{bucket}/{job_key}')

    # Create a source archive for this training job and upload it to S3
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CODE_DIRNAME)
    sockeye_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory(prefix='sockeye_sagemaker.') as tmp_dir:
        # Copy files
        tmp_source_dir = os.path.join(tmp_dir, CODE_DIRNAME)
        shutil.copytree(code_dir, tmp_source_dir)
        shutil.copytree(os.path.join(sockeye_dir, SOCKEYE_DIRNAME), os.path.join(tmp_source_dir, SOCKEYE_DIRNAME))
        os.mkdir(os.path.join(tmp_source_dir, SOCKEYE_CONTRIB_DIRNAME))
        for fname in SOCKEYE_CONTRIB_FNAMES:
            shutil.copy(os.path.join(sockeye_dir, SOCKEYE_CONTRIB_DIRNAME, fname),
                        os.path.join(tmp_source_dir, SOCKEYE_CONTRIB_DIRNAME, fname))
        shutil.copy(os.path.join(sockeye_dir, REQUIREMENTS_DIRNAME, REQUIREMENTS_FNAME),
                    os.path.join(tmp_source_dir, REQUIREMENTS_FNAME))
        # Create archive
        source_dir_archive = os.path.join(tmp_dir, SOURCE_DIR_ARCHIVE_FNAME)
        with tarfile.open(source_dir_archive, 'w:gz') as tar:
            for fname in os.listdir(tmp_source_dir):
                tar.add(os.path.join(tmp_source_dir, fname), arcname=fname)
        # Upload to S3
        source_dir = session.upload_data(path=source_dir_archive, bucket=bucket, key_prefix=job_key)

    # Read original sockeye-train config file
    with open(args.config, 'r') as inp:
        config = yaml.safe_load(inp)

    # If inputs are not already S3 paths, upload the files to S3. Add the S3
    # paths to the job's input dictionary and track basenames in the config.
    # This enables the training script to construct input paths at runtime.
    inputs = {}
    for key in INPUT_ARGS:
        val = config.get(key, None)
        if val:
            if isinstance(val, str):
                if not val.startswith(S3_PREFIX):
                    # Upload file into/as its own directory to avoid collisions
                    val = session.upload_data(path=val, bucket=bucket, key_prefix=f'{job_key}/{key}')
                inputs[key] = val
                config[key] = os.path.basename(val)
            elif isinstance(val, list):
                for i, fname in enumerate(val):
                    key_i = f'{key}_{i}'
                    if not fname.startswith(S3_PREFIX):
                        # Upload file into/as its own directory to avoid
                        # collisions
                        fname = session.upload_data(path=fname, bucket=bucket, key_prefix=f'{job_key}/{key_i}')
                    inputs[key_i] = fname
                    val[i] = os.path.basename(fname)

    # Specify and track S3 path for training outputs
    output_path = f'{S3_PREFIX}{bucket}/{job_key}/{OUTPUT_ARG}'
    config[OUTPUT_ARG] = output_path

    # Upload updated config to S3
    with tempfile.TemporaryDirectory(prefix='sockeye_sagemaker.') as tmp_dir:
        tmp_config = os.path.join(tmp_dir, CONFIG_FNAME)
        with open(tmp_config, 'w') as out:
            yaml.safe_dump(config, out, default_flow_style=False)
            inputs[CONFIG_ARG] = session.upload_data(path=tmp_config, bucket=bucket,
                                                     key_prefix=f'{job_key}/{CONFIG_ARG}')

    # Report training setup before launching SageMaker job
    logger.info(f'SageMaker sockeye-train config: {config}')
    logger.info(f'SageMaker inputs: {inputs}')
    logger.info(f'SageMaker instance type: {args.instance_type}')

    # Launch SageMaker training job
    estimator = PyTorch(
        role=role,
        instance_count=1,  # Sockeye + SageMaker currently runs on a single node
        instance_type=args.instance_type,
        output_path=output_path,
        base_job_name=args.job_prefix,
        source_dir=source_dir,
        entry_point='sagemaker_train.py',
        framework_version=args.framework_version,
        py_version='py38',  # Current minimum version required by Sockeye
    )
    estimator.fit(inputs)


if __name__ == '__main__':
    main()

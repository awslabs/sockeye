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
Extract specific parameters.
"""

import argparse
import os
import logging
from typing import Dict, List

import mxnet as mx
import numpy as np

from . import arguments
from . import constants as C
from . import utils
from .log import setup_main_logger, log_sockeye_version

logger = logging.getLogger(__name__)


def _extract(param_names: List[str],
             params: Dict[str, mx.nd.NDArray],
             ext_params: Dict[str, np.ndarray]) -> List[str]:
    """
    Extract specific parameters from a given base.

    :param param_names: Names of parameters to be extracted.
    :param params: Mapping from parameter names to the actual NDArrays parameters.
    :param ext_params: Extracted parameter dictionary.
    :return: Remaining names of parameters to be extracted.
    """
    remaining_param_names = list(param_names)
    for name in param_names:
        if name in params:
            logger.info("\tFound '%s': shape=%s", name, str(params[name].shape))
            ext_params[name] = params[name].asnumpy()
            remaining_param_names.remove(name)
    return remaining_param_names


def extract(param_path: str,
            param_names: List[str],
            list_all: bool) -> Dict[str, np.ndarray]:
    """
    Extract specific parameters given their names.

    :param param_path: Path to the parameter file.
    :param param_names: Names of parameters to be extracted.
    :param list_all: List names of all available parameters.
    :return: Extracted parameter dictionary.
    """
    logger.info("Loading parameters from '%s'", param_path)
    params = mx.nd.load(param_path)

    ext_params = {}  # type: Dict[str, np.ndarray]
    param_names = _extract(param_names, params, ext_params)
    param_names = _extract(param_names, params, ext_params)

    if len(param_names) > 0:
        logger.info("The following parameters were not found:")
        for name in param_names:
            logger.info("\t%s", name)
        logger.info("Check the following availabilities")
        list_all = True

    if list_all:
        if params:
            logger.info("Available arg parameters:")
            for name in params:
                logger.info("\t%s: shape=%s", name, str(params[name].shape))

    return ext_params


def main():
    """
    Commandline interface to extract parameters.
    """
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description="Extract specific parameters.")
    arguments.add_extract_args(params)
    args = params.parse_args()
    extract_parameters(args)


def extract_parameters(args: argparse.Namespace):
    log_sockeye_version(logger)

    if os.path.isdir(args.input):
        param_path = os.path.join(args.input, C.PARAMS_BEST_NAME)
    else:
        param_path = args.input
    extracted_parameters = extract(param_path, args.names, args.list_all)

    if len(extracted_parameters) > 0:
        utils.check_condition(args.output is not None, "An output filename must be specified. (Use --output)")
        logger.info("Writing extracted parameters to '%s'", args.output)
        np.savez_compressed(args.output, **extracted_parameters)


if __name__ == "__main__":
    main()

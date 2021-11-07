# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# Called before importing mxnet or any module that imports mxnet
from . import initial_setup
initial_setup.handle_env_cli_arg()

import logging
from typing import Optional

from . import arguments
from sockeye.log import setup_main_logger
import torch as pt
from .model_pt import make_pytorch_model_from_mxnet_model
from . import constants as C

import os

logger = logging.getLogger(__name__)


def convert(model: str, checkpoint: Optional[int]):
    setup_main_logger(file_logging=False)
    import sockeye.model
    logger.info(f"############### Loading MXNet model from '{model}', checkpoint={checkpoint}")
    model_mx, _, _ = sockeye.model.load_model(model,
                                              checkpoint=checkpoint,
                                              hybridize=False,
                                              inference_only=False)
    logger.info(f"############### Constructing PyTorch model...")
    model_pt = make_pytorch_model_from_mxnet_model(model_mx)
    if checkpoint is None:
        params_fname = os.path.join(model, f'{C.PARAMS_BEST_NAME}.{C.TORCH_SUFFIX}')
    else:
        params_fname = os.path.join(model, f'{C.PARAMS_NAME % checkpoint}.{C.TORCH_SUFFIX}')
    logger.info(f"############### Saving PyTorch parameters to '{params_fname}...'")
    model_pt.save_parameters(params_fname)
    model_pt.load_parameters(params_fname, device=pt.device('cpu'))


def main():
    params = arguments.ConfigArgumentParser(description='PyTorch converter CLI')
    params.add_argument("--model", "-m", required=True,
                        help="Model directory containing trained model.")
    params.add_argument('--checkpoint', '-c',
                        default=None,
                        type=int,
                        nargs='+',
                        help='Checkpoint index to load (int). If not given, chooses best checkpoints for model(s). ')
    args = params.parse_args()

    convert(args.model, args.checkpoint)


if __name__ == '__main__':
    main()

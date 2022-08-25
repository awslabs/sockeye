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
import shutil
import tempfile

import pytest

import torch

import sockeye.constants as C
from sockeye.convert_deepspeed import convert_model_checkpoints

# Only run tests in this file if DeepSpeed is installed
try:
    import deepspeed
    deepspeed_installed = True
except:
    deepspeed_installed = False


@pytest.mark.skipif(not deepspeed_installed, reason='DeepSpeed is not installed')
def test_convert_model_checkpoints():
    with tempfile.TemporaryDirectory() as work_dir:
        model_dir = os.path.join(work_dir, 'model')
        shutil.copytree(os.path.join('test', 'data', 'deepspeed', 'model'), model_dir, symlinks=True)
        # Convert
        convert_model_checkpoints(model_dirname=model_dir, keep_deepspeed=False)
        # Check
        for fname in os.listdir(model_dir):
            if fname.startswith(C.PARAMS_PREFIX) and fname[len(C.PARAMS_PREFIX):].isdigit():
                converted_params = torch.load(os.path.join(model_dir, fname))
                reference_params = torch.load(os.path.join('test', 'data', 'deepspeed', 'converted', fname))
                for key in converted_params.keys() | reference_params.keys():
                    assert torch.allclose(converted_params[key], reference_params[key])

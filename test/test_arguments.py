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

import pytest

import sockeye.arguments as arguments
import argparse


@pytest.mark.parametrize("test_params, expected_params", [
    # mandatory parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=False)),

    # all parameters
    ('--source test_src --target test_tgt '
     '--validation-source test_validation_src --validation-target test_validation_tgt '
     '--output test_output '
     '--source-vocab test_src_vocab --target-vocab test_tgt_vocab '
     '--use-tensorboard --overwrite-output --quiet',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=True,
          source_vocab='test_src_vocab', target_vocab='test_tgt_vocab', use_tensorboard=True, quiet=True)),

    # short parameters
    ('-s test_src -t test_tgt '
     '-vs test_validation_src -vt test_validation_tgt '
     '-o test_output -q',
     dict(source='test_src', target='test_tgt',
          validation_source='test_validation_src', validation_target='test_validation_tgt',
          output='test_output', overwrite_output=False,
          source_vocab=None, target_vocab=None, use_tensorboard=False, quiet=True))
])
def test_io_args(test_params, expected_params):
    test_parser = argparse.ArgumentParser()
    arguments.add_io_args(test_parser)
    parsed_params = test_parser.parse_args(test_params.split())
    assert dict(vars(parsed_params)) == expected_params

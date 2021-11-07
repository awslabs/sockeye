# Copyright 2020--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os
import sys
from tempfile import TemporaryDirectory

import pytest

from unittest.mock import patch

logger = logging.getLogger(__name__)

EXPECTED_OUTPUT = """6 3 0 9 3 2 0 5 3 8 0 1 0 4 1 6 2 8 9 10 3 7 0 4 9 7 5 2 7 7
2 1 7 7 5 0 5 7 1 7 10 4 0 9 10 5 0 5
6 8 4 6 1 8 1 3 2 4 0 1 6 4 6 1 0 6 5 4 7 0 5
8 1 7 6 9 10 10 3 4 7 8 1 9 6 9 5 2 3 1 1
0 6 4 2 0 6 8 1 0 8 3 7 4 0 8 0 1 2 0 0 8 9 4 1 7 4
10 1 8 2 3 4 2 3 7 6 6
7 7 0 1 5 8 8 8 10 1 7 6 7 4 4 0 9 4 2 7 6 3 8 2
5 7 3 5 3 7 5 5 9 9 7 5 5 5 8 0 10 8 8 5 3 10 5 6 2 9 8 3 7 7
6 9 4 4 7 6 9 4 5 9 10 1 8 2
7 0 1 6 0 6 9 7 2 4 3
"""


def test_backwards_compatibility():
    """
    This test checks whether the current code can still produce translations with a model that was trained with the
    same major version.
    """
    pytest.importorskip('mxnet')
    import sockeye.translate
    with TemporaryDirectory() as work_dir:
        output_file = os.path.join(work_dir, "out")
        params = """{sockeye} --use-cpu --models {model} --input {input} --output {output} """.format(
            sockeye=sockeye.translate.__file__,
            model="test/data/model_2.3.x",
            input="test/data/model_2.3.x/model_input",
            output=output_file
        )
        logger.info("Translating with params %s", params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        with open(output_file) as model_out:
            assert model_out.read() == EXPECTED_OUTPUT


def test_backwards_compatibility_pt():
    """
    This test checks whether the current code can still produce translations with a model that was trained with the
    same major version.
    """
    import sockeye.translate_pt
    with TemporaryDirectory() as work_dir:
        output_file = os.path.join(work_dir, "out")
        params = """{sockeye} --use-cpu --models {model} --input {input} --output {output} """.format(
            sockeye=sockeye.translate_pt.__file__,
            model="test/data/model_2.3.x",
            input="test/data/model_2.3.x/model_input",
            output=output_file
        )
        logger.info("Translating with params %s", params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate_pt.main()

        with open(output_file) as model_out:
            assert model_out.read() == EXPECTED_OUTPUT

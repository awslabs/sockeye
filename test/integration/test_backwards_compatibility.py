# Copyright 2020--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

EXPECTED_OUTPUT = """3 7 2 3 2 1 0 2 3 10 2 5 7
1 3 9 9 3 10 1 2 9 2 1 10
4 1 3 10 9 3 10 5 9 0 7
5 2 0 8 2 6 10 5 8 10 10 7 7
0 7 6 4 6 1 0 10 3 4 3 4 5 6 5 8 4 9 0 3 2 7 10 10 9 10 10
9 2 7 4 10 5 10 10 4 8 8 2 0 6 8 0 8 7 3 5 7 9
8 6 10 7 6 7 1 4 0 2 2 3 3 10 3 8 0 10 0 2
10 5 8 8 1 9 6 6 10 2 10 6 5 10 5 1 3 0 2
8 10 7 7 7 3 10 8 2 4 0 2 2 3 0 4 0 7 4
7 1 10 9 10 3 4 8 0 0 9 4 3 7 5 1 5 3 8 1 5 1 6 7 9 6 7
"""


def test_backwards_compatibility():
    """
    This test checks whether the current code can still produce translations with a model that was trained with the
    same major version.
    """
    import sockeye.translate
    with TemporaryDirectory() as work_dir:
        output_file = os.path.join(work_dir, "out")
        params = """{sockeye} --use-cpu --models {model} --input {input} --output {output} """.format(
            sockeye=sockeye.translate.__file__,
            model="test/data/model_3.0.x",
            input="test/data/model_3.0.x/model_input",
            output=output_file
        )
        logger.info("Translating with params %s", params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        with open(output_file) as model_out:
            assert model_out.read() == EXPECTED_OUTPUT

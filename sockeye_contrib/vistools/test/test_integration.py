# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest
from filecmp import dircmp

from generate_graphs import generate

CWD = os.path.dirname(os.path.realpath(__file__))

BEAM_COMPARISONS = [(os.path.join(CWD, "resources", "beams.json"),
                     os.path.join(CWD, "resources", "output"))]

@pytest.mark.parametrize("beams, expected_output", BEAM_COMPARISONS)
def test_beam_generation(beams, expected_output, tmpdir):
    generate(beams, str(tmpdir))

    # Same files in each dir, does not check contents
    result = dircmp(expected_output, str(tmpdir))
    assert result.left_list == result.right_list

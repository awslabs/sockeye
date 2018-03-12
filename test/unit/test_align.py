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
import numpy as np
from sockeye import align


def test_generate_pointer_labels():
    aligner = align.Aligner()

    # TODO: create test fixtures
    src = [[2, 3, 4], [12, 1, 2], [1, 1, 1, 12]]
    trg = [[2, 1, 1], [1, 22, 21], [9, 1, 1]]
    expected_out = [[1, -1, -1], [2, -1, -1], [-1, 1, 2]]

    for i in range(len(src)):
        assert (expected_out[i] == aligner.get_copy_alignment(src[i], trg[i])).all()

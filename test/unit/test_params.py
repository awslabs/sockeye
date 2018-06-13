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

import itertools
import glob
import os.path
import tempfile

import sockeye.training
import sockeye.constants as C
import sockeye.utils


def test_cleanup_param_files():
    with tempfile.TemporaryDirectory() as tmpDir:
        for n in itertools.chain(range(1, 20, 2), range(21, 41)):
            # Create empty files
            open(os.path.join(tmpDir, C.PARAMS_NAME % n), "w").close()
        sockeye.utils.cleanup_params_files(tmpDir, 5, 40, 17)

        expectedSurviving = set([os.path.join(tmpDir, C.PARAMS_NAME % n)
                                 for n in [17, 36, 37, 38, 39, 40]])
        # 17 must survive because it is the best one
        assert set(glob.glob(os.path.join(tmpDir, C.PARAMS_PREFIX + "*"))) == expectedSurviving

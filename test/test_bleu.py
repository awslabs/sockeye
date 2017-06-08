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
import sockeye.bleu


test_cases = [(["this is a test", "another test"], ["ref1", "ref2"], 0.003799178428257963),
              (["this is a test"], ["this is a test"], 1.0),
              (["this is a fest"], ["this is a test"], 0.223606797749979)]


@pytest.mark.parametrize("hypotheses, references, expected_bleu", test_cases)
def test_bleu(hypotheses, references, expected_bleu):
    bleu = sockeye.bleu.corpus_bleu(hypotheses, references)
    assert abs(bleu - expected_bleu) < 1e-8
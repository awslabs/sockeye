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

import sockeye.chrf as chrf


@pytest.mark.parametrize("hypothesis, reference, expected_chrf",
                         [("a b c", "a b c", 1.0),
                          ("a b c", "abc", 1.0),
                          ("", "c", 0.0)])
def test_sentence_chrf(hypothesis, reference, expected_chrf):
    value = chrf.sentence_chrf(hypothesis, reference)
    assert np.isclose(value, expected_chrf)


@pytest.mark.parametrize("hypotheses, references, expected_chrf",
                         [(["a b c"], ["a b c"], 1.0),
                          (["a b c"], ["abc"], 1.0),
                          ([""], ["c"], 0.0),
                          (["a", "b"], ["a", "c"], 0.5)])
def test_corpus_chrf(hypotheses, references, expected_chrf):
    value = chrf.corpus_chrf(hypotheses, references)
    assert np.isclose(value, expected_chrf)
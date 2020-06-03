# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sockeye_contrib import rouge

test_cases = [(["this is a test", "another test case"], ["this is a test case", "another test case"], 0.9444444394753087, 0.928571423622449, 0.9338624338620563),
              (["this is a single test case"], ["this is a single test case"], 0.999999995, 0.999999995, 0.9999999999995),
              (["single test case"], ["another single test case"], 0.8571428522448981, 0.7999999952000001, 0.8241758241756372),
              (["no overlap between sentences"], ["this is another test case"], 0.0, 0.0, 0.0),
              (["exact match in the test case", "another exact match"], ["exact match in the test case", "another exact match"], 0.999999995, 0.999999995, 0.9999999999995)]

@pytest.mark.parametrize("hypotheses, references, rouge1_score, rouge2_score, rougel_score", test_cases)
def test_rouge_1(hypotheses, references, rouge1_score, rouge2_score, rougel_score):
    rouge_score = rouge.rouge_1(hypotheses, references)
    assert rouge_score == rouge1_score

@pytest.mark.parametrize("hypotheses, references, rouge1_score, rouge2_score, rougel_score", test_cases)
def test_rouge_2(hypotheses, references, rouge1_score, rouge2_score, rougel_score):
     rouge_score = rouge.rouge_2(hypotheses, references)
     assert rouge_score == rouge2_score

@pytest.mark.parametrize("hypotheses, references, rouge1_score, rouge2_score, rougel_score", test_cases)
def test_rouge_l(hypotheses, references, rouge1_score, rouge2_score, rougel_score):
     rouge_score = rouge.rouge_l(hypotheses, references)
     assert rouge_score == rougel_score

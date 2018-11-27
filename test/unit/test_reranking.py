# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as np
import pytest

import sockeye.rerank as rerank


@pytest.mark.parametrize("hypotheses, reference, expected_output, metric", [
    (["No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      'No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament'], "bleu"),
    # test chrf as metric
    (["No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      'No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament'], "chrf"),
    # test empty hypothesis
    (["",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      ''], "bleu")
])
def test_rerank_hypotheses(hypotheses, reference, expected_output, metric):
    reranker = rerank.Reranker(metric=metric, return_score=False)
    hypotheses = {'translations': hypotheses}
    reranked_hypotheses = reranker.rerank(hypotheses, reference)
    assert reranked_hypotheses['translations'] == expected_output


@pytest.mark.parametrize("hypotheses, reference, expected_scores", [
    (["Completely different",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     [60.26404810093175, 3.2102598200446005e-05, ])
])
def test_rerank_return_score(hypotheses, reference, expected_scores):
    reranker = rerank.Reranker(metric="bleu", return_score=True)
    hypotheses = {'translations': hypotheses}
    reranked_hypotheses = reranker.rerank(hypotheses, reference)
    assert 'scores' in reranked_hypotheses
    actual_scores = reranked_hypotheses['scores']
    assert np.allclose(actual_scores, expected_scores)

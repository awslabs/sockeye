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

    reranked = reranker.rerank_hypotheses(hypotheses, reference)
    actual_list = reranked.hypotheses

    assert actual_list == expected_output


@pytest.mark.parametrize("hypotheses, reference, expected_scores", [
    (["Completely different",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     [60.26404810093175, 3.2102598200446005e-05, ])
])
def test_rerank_return_score(hypotheses, reference, expected_scores):
    reranker = rerank.Reranker(metric="bleu", return_score=True)

    reranked_with_scores = reranker.rerank_hypotheses(hypotheses, reference)

    actual_scores = reranked_with_scores.scores

    assert np.allclose(actual_scores, expected_scores)


@pytest.mark.parametrize("hypotheses, reference, expected_best", [
    (["Completely different",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament")
])
def test_rerank_top1(hypotheses, reference, expected_best):
    reranker = rerank.Reranker(metric="bleu", return_score=False)

    reranked = reranker.rerank_top1(hypotheses, reference)

    assert len(reranked.hypotheses) == 1, "Rerank top1 should not return more than 1 hypothesis."
    actual_hypothesis = reranked.hypotheses[0]

    assert actual_hypothesis == expected_best


@pytest.mark.parametrize("hypotheses, reference, expected_best, expected_score", [
    (["Completely different",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament",
     60.26404810093175)
])
def test_rerank_top1_score(hypotheses, reference, expected_best, expected_score):
    reranker = rerank.Reranker(metric="bleu", return_score=True)

    reranked_with_scores = reranker.rerank_top1(hypotheses, reference)

    assert len(reranked_with_scores.hypotheses) == 1, "Rerank top1 should not return more than 1 hypothesis."
    actual_hypothesis = reranked_with_scores.hypotheses[0]
    actual_score = reranked_with_scores.scores[0]

    assert actual_hypothesis == expected_best
    assert np.isclose(actual_score, expected_score)

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

import numpy as np
import pytest

import sockeye.rerank as rerank


@pytest.mark.parametrize("source, hypotheses, reference, expected_output, metric", [
    # test bleu as metric
    ("El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko",
     ["No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      'No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament'], "bleu"),
    # test chrf as metric
    ("El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko",
     ["No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      'No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament'], "chrf"),
    # test empty hypothesis
    ("El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko",
     ["",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      ''], "bleu"),
])
def test_rerank_hypotheses(source, hypotheses, reference, expected_output, metric):
    reranker = rerank.Reranker(metric=metric, isometric_alpha=0.5, return_score=False)
    hypotheses = {'sentence_id': 0,
                  'text': source,
                  'translation': '',
                  'translations': hypotheses}
    reranked_hypotheses = reranker.rerank(hypotheses, reference)
    assert reranked_hypotheses['translations'] == expected_output


@pytest.mark.parametrize("source, hypotheses, scores, reference, expected_output, metric", [
    # test isometric-ratio as metric
    ("El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko",
     ["No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     [[0.377], [0.455]],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      'No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament'],
     "isometric-ratio"),
    # test isometric-lc
    ("El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko",
     ["No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     [[0.377], [0.455]],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     ['No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament',
      'No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament'],
     "isometric-lc"),
])
def test_rerank_hypotheses_isometric(source, hypotheses, scores, reference, expected_output, metric):
    reranker = rerank.Reranker(metric=metric, isometric_alpha=0.5, return_score=False)
    hypotheses = {'sentence_id': 0,
                  'text': source,
                  'translation': '',
                  'scores': scores,
                  'translations': hypotheses}
    reranked_hypotheses = reranker.rerank(hypotheses, reference)
    assert reranked_hypotheses['translations'] == expected_output


@pytest.mark.parametrize("source, hypotheses, reference, expected_scores", [
    ("El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko",
     ["Completely different",
      "No Liber@@ ating Ty@@ mo@@ sh@@ en@@ ko by Parliament"],
     "Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko",
     [61.69564583930634, 0.0])
])
def test_rerank_return_score(source, hypotheses, reference, expected_scores):
    reranker = rerank.Reranker(metric="bleu", isometric_alpha=0.5, return_score=True)
    hypotheses = {'sentence_id': 0,
                  'text': source,
                  'translation': '',
                  'translations': hypotheses}
    reranked_hypotheses = reranker.rerank(hypotheses, reference)
    assert 'scores' in reranked_hypotheses
    actual_scores = reranked_hypotheses['scores']
    assert np.allclose(actual_scores, expected_scores)

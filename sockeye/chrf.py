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

"""
Computes chrF scores as described in
'CHRF: character n-gram F-score for automatic MT evaluation' by Maja Popovic.
[http://www.statmt.org/wmt15/pdf/WMT49.pdf]
"""

import re
from collections import Counter
from typing import Iterable, Tuple

import numpy as np

ORDER = 6
BETA = 3.0
TRIM_WS = True


def extract_ngrams(s: str, n: int) -> Counter:
    """
    Yields counts of character n-grams from string s of order n.
    """
    return Counter([s[i:i + n] for i in range(len(s) - n + 1)])


def delete_whitespace(text: str) -> str:
    """
    Removes whitespaces from text.
    """
    return re.sub("\s+", "", text)


def get_sentence_statistics(hypothesis: str,
                            reference: str,
                            order: int = ORDER,
                            trim_whitespaces: bool = TRIM_WS) -> np.array:
    hypothesis = delete_whitespace(hypothesis) if trim_whitespaces else hypothesis
    reference = delete_whitespace(reference) if trim_whitespaces else reference
    statistics = np.zeros((order * 3))
    for i in range(order):
        n = i + 1
        hypothesis_ngrams = extract_ngrams(hypothesis, n)
        reference_ngrams = extract_ngrams(reference, n)
        common_ngrams = hypothesis_ngrams & reference_ngrams
        statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
        statistics[3 * i + 1] = sum(reference_ngrams.values())
        statistics[3 * i + 2] = sum(common_ngrams.values())
    return statistics


def get_corpus_statistics(hypotheses: Iterable[str],
                          references: Iterable[str],
                          order: int = ORDER,
                          trim_whitespaces: bool = TRIM_WS) -> np.array:
    corpus_statistics = np.zeros((order * 3))
    for hypothesis, reference in zip(hypotheses, references):
        statistics = get_sentence_statistics(hypothesis, reference, order=order, trim_whitespaces=trim_whitespaces)
        corpus_statistics += statistics
    return corpus_statistics


def _avg_precision_and_recall(statistics: np.array, order: int) -> Tuple[float, float]:
    avg_precision = 0.0
    avg_recall = 0.0
    effective_order = 0
    for i in range(order):
        hypotheses_ngrams = statistics[3 * i + 0]
        references_ngrams = statistics[3 * i + 1]
        common_ngrams = statistics[3 * i + 2]
        if hypotheses_ngrams > 0 and references_ngrams > 0:
            avg_precision += common_ngrams / hypotheses_ngrams
            avg_recall += common_ngrams / references_ngrams
            effective_order += 1
    if effective_order == 0:
        return 0.0, 0.0
    avg_precision /= effective_order
    avg_recall /= effective_order
    return avg_precision, avg_recall


def _chrf(avg_precision, avg_recall, beta: float = BETA) -> float:
    if avg_precision + avg_recall == 0:
        return 0.0
    beta_square = beta ** 2
    return (1 + beta_square) * (avg_precision * avg_recall) / ((beta_square * avg_precision) + avg_recall)


def corpus_chrf(hypotheses: Iterable[str],
                references: Iterable[str],
                order: int = ORDER,
                trim_whitespaces: bool = TRIM_WS,
                beta: float = BETA) -> float:
    """
    Computes Chrf on a corpus.

    :param hypotheses: Stream of hypotheses.
    :param references: Stream of references
    :param order: Maximum n-gram order.
    :param trim_whitespaces: Whether to trim whitespaces from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    corpus_statistics = get_corpus_statistics(hypotheses, references, order=order, trim_whitespaces=trim_whitespaces)
    avg_precision, avg_recall = _avg_precision_and_recall(corpus_statistics, order)
    return _chrf(avg_precision, avg_recall, beta=beta)


def sentence_chrf(hypothesis: str,
                  reference: str,
                  order: int = ORDER,
                  trim_whitespaces: bool = TRIM_WS,
                  beta: float = BETA) -> float:
    """
    Computes Chrf on a single sentence pair.

    :param hypothesis: Hypothesis string.
    :param reference: Reference string.
    :param order: Maximum n-gram order.
    :param trim_whitespaces: Whether to trim whitespaces from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    statistics = get_sentence_statistics(hypothesis, reference, order=order, trim_whitespaces=trim_whitespaces)
    avg_precision, avg_recall = _avg_precision_and_recall(statistics, order)
    return _chrf(avg_precision, avg_recall, beta=beta)

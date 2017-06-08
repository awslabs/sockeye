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
Implementation of the BLEU-4 score [Papineni, 2002]
"""
import logging
from collections import Counter, namedtuple
from itertools import tee, islice
from math import log, exp
from typing import List, AnyStr

logger = logging.getLogger(__name__)

ORDER = 4
Statistics = namedtuple('Statistics', ['common', 'total'])


def zipngram(words, n):
    return zip(*(islice(it, pos, None) for pos, it in enumerate(tee(words, n + 1))))


def bleu_from_counts(count_triple, offset=0.01):
    counts, hyp_count, ref_count = count_triple
    bleu = 0.0
    brevity = 0.0
    effective_order = 0
    for n in range(ORDER):
        count = counts.common[n]
        total = counts.total[n]
        if total <= 0:
            if n == 0:
                return 0
            else:
                break
        effective_order += 1
        if count == 0:
            count = offset
        bleu += log(float(count) / total)
    if hyp_count > 0:
        brevity = min(0., 1. - float(ref_count) / hyp_count)
    return exp(bleu / effective_order + brevity)


def bleu_counts(hyp, ref):
    counts = Statistics([0, 0, 0, 0], [0, 0, 0, 0])

    hyp_words = hyp.split()
    ref_words = ref.split()

    hyp_wcount = len(hyp_words)
    ref_wcount = len(ref_words)
    for n in range(ORDER):
        h_grams = Counter(zipngram(hyp_words, n))
        r_grams = Counter(zipngram(ref_words, n))

        # do clipping
        inter = (min(h_grams[g], r_grams[g]) for g in h_grams if g in r_grams)
        counts.common[n] += sum(inter)
        counts.total[n] += sum(h_grams.values())

    return counts, hyp_wcount, ref_wcount


def add_counts_in_place(c1, c2):
    for n in range(ORDER):
        c1.common[n] += c2.common[n]
        c1.total[n] += c2.total[n]


def corpus_bleu_counts(hyps, refs):
    counts = Statistics([0, 0, 0, 0], [0, 0, 0, 0])
    hyp_total_wcount, ref_total_wcount = 0, 0

    if len(hyps) != len(refs):
        logger.error("Hyps and refs lengths are not the same")

    for hyp, ref in zip(hyps, refs):
        sent_counts, hyp_wcount, ref_wcount = bleu_counts(hyp, ref)

        add_counts_in_place(counts, sent_counts)
        hyp_total_wcount += hyp_wcount
        ref_total_wcount += ref_wcount

    return counts, hyp_total_wcount, ref_total_wcount


def corpus_bleu(hyps: List[AnyStr], refs: List[AnyStr], offset: float = 0.01) -> float:
    """
    Computes corpus BLEU.

    :param hyps: List of hypotheses.
    :param refs: List of references.
    :param offset: Smoothing value.
    :return: BLEU score.
    """
    return bleu_from_counts(corpus_bleu_counts(hyps, refs), offset)

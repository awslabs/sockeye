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
Evaluation CLI. Prints corpus BLEU
"""
import argparse
import logging
import sys
import numpy as np
from typing import Iterable, Optional
from collections import defaultdict

from contrib import sacrebleu
from sockeye.log import setup_main_logger, log_sockeye_version
from . import arguments
from . import constants as C
from . import data_io
from . import utils

logger = setup_main_logger(__name__, file_logging=False)


def raw_corpus_bleu(hypotheses: Iterable[str], references: Iterable[str], offset: Optional[float] = 0.01) -> float:
    """
    Simple wrapper around sacreBLEU's BLEU without tokenization and smoothing.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :param offset: Smoothing constant.
    :return: BLEU score as float between 0 and 1.
    """
    return sacrebleu.raw_corpus_bleu(hypotheses, [references], smooth_floor=offset).score / 100.0


def raw_corpus_chrf(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around sacreBLEU's chrF implementation, without tokenization.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: chrF score as float between 0 and 1.
    """
    return sacrebleu.corpus_chrf(hypotheses, references, order=sacrebleu.CHRF_ORDER, beta=sacrebleu.CHRF_BETA,
                                 remove_whitespace=True)


def main():
    params = argparse.ArgumentParser(description='Evaluate translations by calculating metrics with '
                                                 'respect to a reference set. If multiple hypotheses files are given'
                                                 'the mean and standard deviation of the metrics are reported.')
    arguments.add_evaluate_args(params)
    arguments.add_logging_args(params)
    args = params.parse_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)

    utils.check_condition(args.offset >= 0, "Offset should be non-negative.")
    log_sockeye_version(logger)

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    references = [' '.join(e) for e in data_io.read_content(args.references)]
    all_hypotheses = [[h.strip() for h in hypotheses] for hypotheses in args.hypotheses]
    metrics = args.metrics
    logger.info("%d hypotheses | %d references", len(all_hypotheses), len(references))

    if not args.not_strict:
        for hypotheses in all_hypotheses:
            utils.check_condition(len(hypotheses) == len(references),
                                  "Number of hypotheses (%d) and references (%d) does not match." % (len(hypotheses),
                                                                                                     len(references)))
    metric_info = []
    for metric in metrics:
        metric_info.append("%s\t(s_opt)" % metric)
    logger.info("\t".join(metric_info))

    if not args.sentence:
        scores = defaultdict(list)
        for hypotheses in all_hypotheses:
            for metric in metrics:
                if metric == C.BLEU:
                    score = raw_corpus_bleu(hypotheses, references, args.offset)
                elif metric == C.CHRF:
                    score = raw_corpus_chrf(hypotheses, references)
                else:
                    raise ValueError("Unknown metric %s." % metric)
                scores[metric].append(score)
        _print_mean_std_score(metrics, scores)
    else:
        for hypotheses in all_hypotheses:
            for h, r in zip(hypotheses, references):
                scores = defaultdict(list)
                for metric in metrics:
                    if metric == C.BLEU:
                        score = raw_corpus_bleu([h], [r], args.offset)
                    elif metric == C.CHRF:
                        score = raw_corpus_chrf(h, r)
                    else:
                        raise ValueError("Unknown metric %s." % metric)
                    scores[metric].append(score)
                _print_mean_std_score(metrics, scores)


def _print_mean_std_score(metrics, scores):
    scores_mean_std = []
    for metric in metrics:
        if len(scores[metric]) > 1:
            score_mean = np.asscalar(np.mean(scores[metric]))
            score_std = np.asscalar(np.std(scores[metric], ddof=1))
            scores_mean_std.append("%.3f\t%.3f" % (score_mean, score_std))
        else:
            score = scores[metric][0]
            scores_mean_std.append("%.3f\t(-)" % score)
    print("\t".join(scores_mean_std))


if __name__ == '__main__':
    main()

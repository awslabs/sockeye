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
CLI to rerank an nbest list of translations.
"""

import argparse
import json
import sys
from collections import namedtuple
from typing import List

import numpy as np

from sockeye_contrib import sacrebleu
from . import arguments
from . import constants as C
from . import log
from . import utils

logger = log.setup_main_logger(__name__, console=True, file_logging=False)

RerankOutput = namedtuple('RerankOutput', ['hypotheses', 'scores'])


class Reranker:
    """
    Reranks a list of hypotheses according to a sentence-level metric.

    :param metric: Sentence-level metric such as smoothed BLEU.
    :param return_score: If True, also return the sentence-level score.
    """

    def __init__(self, metric: str,
                 return_score: bool = False) -> None:
        if metric == C.RERANK_BLEU:
            self.scoring_function = sacrebleu.sentence_bleu
        elif metric == C.RERANK_CHRF:
            self.scoring_function = sacrebleu.sentence_chrf
        else:
            raise utils.SockeyeError("Scoring metric '%s' unknown. Choices are: %s" % (metric, C.RERANK_METRICS))

        self.return_score = return_score

    def rerank_hypotheses(self, hypotheses: List[str],
                          reference: str) -> RerankOutput:
        """
        Reranks a set of hypotheses that belong to one single reference
        translation.

        :param hypotheses: List of nbest translations.
        :param reference: A single string with the actual reference translation.
        :return: A sorted list of hypotheses, possibly with scores.
        """
        scores = [self.scoring_function(hypothesis, reference) for hypothesis in hypotheses]

        sorted_indexes = np.argsort(scores)[::-1]  # descending
        sorted_hypotheses = [hypotheses[i] for i in sorted_indexes]

        if self.return_score:
            sorted_scores = [scores[i] for i in sorted_indexes]
            return RerankOutput(hypotheses=sorted_hypotheses,
                                scores=sorted_scores)
        else:
            return RerankOutput(hypotheses=sorted_hypotheses,
                                scores=[])

    def rerank_top1(self, hypotheses: List[str],
                    reference: str) -> RerankOutput:
        """
        Reranks a set of hypotheses that belong to one single reference
        translation and outputs the best hypothesis.

        :param hypotheses: List of nbest translations.
        :param reference: A single string with the actual reference translation.
        :return: The single best hypothesis, possibly with its score.
        """
        scores = [self.scoring_function(hypothesis, reference) for hypothesis in hypotheses]
        best_index = np.argmax(scores)  # type: int

        if self.return_score:
            return RerankOutput(hypotheses=[hypotheses[best_index]],
                                scores=[scores[best_index]])
        else:
            return RerankOutput(hypotheses=[hypotheses[best_index]],
                                scores=[])


def rerank(args: argparse.Namespace):
    """
    Reranks a list of hypotheses acoording to a sentence-level metric.
    Writes all output to STDOUT.

    :param args: Namespace object holding CLI arguments.
    """
    reranker = Reranker(args.metric)

    with utils.smart_open(args.reference) as reference, utils.smart_open(args.hypotheses) as hypotheses:
        for reference_line, hypothesis_line in zip(reference, hypotheses):
            reference_line = reference_line.strip()
            hypotheses = json.loads(hypothesis_line)

            utils.check_condition(len(hypotheses) > 1, "Reranking strictly needs more than 1 hypothesis.")

            if args.output_best:
                rank_output = reranker.rerank_top1(hypotheses, reference_line)
                sys.stdout.write(rank_output.hypotheses[0] + "\n")
            else:
                rank_output = reranker.rerank_hypotheses(hypotheses, reference_line)
                sys.stdout.write(json.dumps(rank_output.hypotheses) + "\n")


def main():
    """
    Commandline interface to rerank nbest lists.
    """
    log.log_sockeye_version(logger)

    params = argparse.ArgumentParser(description="Rerank nbest lists of translations."
                                                 " Reranking sorts a list of hypotheses according"
                                                 " to their score compared to a common reference.")
    arguments.add_rerank_args(params)
    args = params.parse_args()

    logger.info(args)

    rerank(args)


if __name__ == "__main__":
    main()

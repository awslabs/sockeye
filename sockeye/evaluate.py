# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Evaluation CLI.
"""
import argparse
import logging
import sys
from collections import defaultdict
from functools import partial, lru_cache
from typing import Callable, Iterable, Dict, List, Tuple, Optional

import numpy as np
import sacrebleu

from sockeye_contrib import rouge
from . import arguments
from . import constants as C
from . import data_io
from . import utils
from .log import setup_main_logger, log_sockeye_version

logger = logging.getLogger(__name__)


DEFAULT_OFFSET = sacrebleu.BLEU.SMOOTH_DEFAULTS['floor']  # 0.1


def raw_corpus_bleu(hypotheses: Iterable[str], references: Iterable[str],
                    offset: Optional[float] = DEFAULT_OFFSET) -> float:
    """
    Simple wrapper around sacreBLEU's BLEU without tokenization and floor smoothing.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :param offset: Smoothing constant.
    :return: BLEU score as float between 0 and 1.
    """
    return sacrebleu.raw_corpus_bleu(hypotheses, [references], smooth_value=offset).score / 100.0  # type: ignore


def raw_corpus_chrf(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around sacreBLEU's chrF implementation, without tokenization.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: chrF score as float between 0 and 1.
    """
    return sacrebleu.corpus_chrf(hypotheses, [references]).score  # type: ignore


def raw_corpus_ter(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around sacreBLEU's TER implementation, without tokenization.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: TER score as float between 0 and 1.
    """
    ter = sacrebleu.metrics.TER()
    return ter.corpus_score(hypotheses, [references]).score  # type: ignore


def raw_corpus_rouge1(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around ROUGE-1 implementation.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: ROUGE-1 score as float between 0 and 1.
    """
    return rouge.rouge_1(hypotheses, references)


def raw_corpus_rouge2(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around ROUGE-2 implementation.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: ROUGE-2 score as float between 0 and 1.
    """
    return rouge.rouge_2(hypotheses, references)


def raw_corpus_rougel(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around ROUGE-L implementation.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: ROUGE-L score as float between 0 and 1.
    """
    return rouge.rouge_l(hypotheses, references)


def raw_corpus_length_ratio(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    """
    Simple wrapper around length ratio implementation.

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :return: Length ratio score as float.
    """
    ratios = [len(h.split())/len(r.split()) for h, r in zip(hypotheses, references)]
    return sum(ratios)/len(ratios) if len(ratios) else 0.0


def serialize_factors(factors: List[Iterable[str]]) -> List[str]:
    factors_list = zip(*factors)
    for factors in factors_list:
        factors_tokens = [f.strip().split(" ") for f in factors]
        inverse_factors = zip(*factors_tokens)
        yield " ".join([" ".join(f) for f in inverse_factors])


def detokenize_signwriting(strings: List[str]) -> List[str]:
    from signwriting.tokenizer import SignWritingTokenizer
    tokenizer = SignWritingTokenizer()
    signwriting_texts = [tokenizer.tokens_to_text(s.split(" ")) for s in strings]
    # Regex Replace ([RBLM])00 with the capture group
    import re
    return [re.sub(r"([RBLM])00", r"\1", s) for s in signwriting_texts]


def raw_corpus_signwriting_similarity(hypotheses_factors: List[Iterable[str]],
                                      references_factors: List[Iterable[str]]) -> float:
    """
    Simple wrapper around the signwriting-evaluation similarity score.

    :param hypotheses_factors: Hypothesis factors streams.
    :param references_factors: Reference factors streams.
    :return: Similarity score as float between 0 and 1.
    """
    try:
        from signwriting_evaluation.metrics.similarity import SignWritingSimilarityMetric
    except ImportError:
        raise ImportError("Please install signwriting-evaluation to use the SignWriting Similarity metric.")

    hypotheses = detokenize_signwriting(serialize_factors(hypotheses_factors))
    references = detokenize_signwriting(serialize_factors(references_factors))

    metric = SignWritingSimilarityMetric()
    return metric.corpus_score(hypotheses, [references])


@lru_cache(maxsize=1)
def load_signwriting_clip():
    try:
        from signwriting_evaluation.metrics.clip import SignWritingCLIPScore
    except ImportError:
        raise ImportError("Please install signwriting-evaluation to use the SignWriting CLIP metric.")

    return SignWritingCLIPScore()


def raw_corpus_signwriting_clip(hypotheses_factors: List[Iterable[str]],
                                references_factors: List[Iterable[str]]) -> float:
    """
    Simple wrapper around the signwriting-evaluation clip score.

    :param hypotheses_factors: Hypothesis factors streams.
    :param references_factors: Reference factors streams.
    :return: CLIPScore score as float between -1 and 1.
    """
    metric = load_signwriting_clip()

    hypotheses = detokenize_signwriting(serialize_factors(hypotheses_factors))
    references = detokenize_signwriting(serialize_factors(references_factors))
    return metric.corpus_score(hypotheses, [references])


def main():
    params = argparse.ArgumentParser(description='Evaluate translations by calculating metrics with '
                                                 'respect to a reference set. If multiple hypotheses files are given '
                                                 'the mean and standard deviation of the metrics are reported.')
    arguments.add_evaluate_args(params)
    arguments.add_logging_args(params)
    args = params.parse_args()
    setup_main_logger(file_logging=False)

    if args.quiet:
        logger.setLevel(logging.ERROR)

    utils.check_condition(args.offset >= 0, "Offset should be non-negative.")
    log_sockeye_version(logger)

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    references = [' '.join(e) for e in data_io.read_content(args.references)]
    all_hypotheses = [[h.strip() for h in hypotheses] for hypotheses in args.hypotheses]
    if not args.not_strict:
        for hypotheses in all_hypotheses:
            utils.check_condition(len(hypotheses) == len(references),
                                  "Number of hypotheses (%d) and references (%d) does not match." % (len(hypotheses),
                                                                                                     len(references)))
    logger.info("%d hypothesis set(s) | %d hypotheses | %d references",
                len(all_hypotheses), len(all_hypotheses[0]), len(references))

    metric_info = ["%s\t(s_opt)" % name for name in args.metrics]
    logger.info("\t".join(metric_info))

    metrics = []  # type: List[Tuple[str, Callable]]
    for name in args.metrics:
        if name == C.BLEU:
            func = partial(raw_corpus_bleu, offset=args.offset)
        elif name == C.CHRF:
            func = raw_corpus_chrf
        elif name == C.ROUGE1:
            func = raw_corpus_rouge1
        elif name == C.ROUGE2:
            func = raw_corpus_rouge2
        elif name == C.ROUGEL:
            func = raw_corpus_rougel
        elif name == C.TER:
            func = raw_corpus_ter
        elif name == C.SIGNWRITING_CLIP:
            func = raw_corpus_signwriting_clip
        elif name == C.SIGNWRITING_SIMILARITY:
            func = raw_corpus_signwriting_similarity
        else:
            raise ValueError("Unknown metric %s." % name)
        metrics.append((name, func))

    if not args.sentence:
        scores = defaultdict(list)  # type: Dict[str, List[float]]
        for hypotheses in all_hypotheses:
            for name, metric in metrics:
                scores[name].append(metric(hypotheses, references))
        _print_mean_std_score(metrics, scores)
    else:
        for hypotheses in all_hypotheses:
            for h, r in zip(hypotheses, references):
                scores = defaultdict(list)  # type: Dict[str, List[float]]
                for name, metric in metrics:
                    scores[name].append(metric([h], [r]))
                _print_mean_std_score(metrics, scores)


def _print_mean_std_score(metrics: List[Tuple[str, Callable]], scores: Dict[str, List[float]]):
    scores_mean_std = []  # type: List[str]
    for name, _ in metrics:
        if len(scores[name]) > 1:
            score_mean = np.mean(scores[name]).item()
            score_std = np.std(scores[name], ddof=1).item()
            scores_mean_std.append("%.3f\t%.3f" % (score_mean, score_std))
        else:
            score = scores[name][0]
            scores_mean_std.append("%.3f\t(-)" % score)
    print("\t".join(scores_mean_std))


if __name__ == '__main__':
    main()

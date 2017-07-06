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
import sys
import time
import logging

from sockeye.log import setup_main_logger, log_sockeye_version
from sockeye.bleu import corpus_bleu, bleu_from_counts, corpus_bleu_counts, bleu_counts
from sockeye.data_io import read_content
from sockeye.utils import check_condition

def main():
    params = argparse.ArgumentParser(description='Evaluate translations by calculating 4-BLEU score with respect to a reference set')
    params.add_argument('--references', '-r', required=True, type=str, help="File with references")
    params.add_argument('--hypotheses', '-i', required=True, type=str, help="File with references")
    params.add_argument('--quiet', '-q', action="store_true", help="Do not print logging information")
    params.add_argument('--sentence', '-s', action="store_true", help="Show sentence-BLEU")
    params.add_argument('--offset', type=float, default = 0.01,
                         help="Numerical value of the offset of zero n-gram counts")
    args = params.parse_args()

    check_condition(args.offset >= 0, "Offset should be non-negative.")

    logger = setup_main_logger(__name__, file_logging=False)
    log_sockeye_version(logger)

    if args.quiet:
        logger.setLevel(logging.ERROR)

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    hypotheses = [' '.join(e) for e in read_content(args.hypotheses)]
    references = [' '.join(e) for e in read_content(args.references)]

    logger.info("Loaded %d hypotheses", len(hypotheses))
    logger.info("Loaded %d references", len(references))

    check_condition(len(hypotheses) == len(references), "Hypotheses and references have different number of lines.")

    if not args.sentence:
        bleu = corpus_bleu(hypotheses, references, args.offset)
        print(bleu, file=sys.stdout)
    else:
        for h, r in zip(hypotheses, references):
            bleu = bleu_from_counts(bleu_counts(h, r), args.offset)
            print(bleu, file=sys.stdout)


if __name__ == '__main__':
    main()


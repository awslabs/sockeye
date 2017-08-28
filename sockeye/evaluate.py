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

from sockeye.bleu import corpus_bleu, bleu_from_counts, bleu_counts
from sockeye.log import setup_main_logger, log_sockeye_version
from . import arguments
from . import data_io
from . import utils

logger = setup_main_logger(__name__, file_logging=False)


def main():
    params = argparse.ArgumentParser(description='Evaluate translations by calculating 4-BLEU '
                                                 'score with respect to a reference set.')
    arguments.add_evaluate_args(params)
    args = params.parse_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)

    utils.check_condition(args.offset >= 0, "Offset should be non-negative.")
    log_sockeye_version(logger)

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    references = [' '.join(e) for e in data_io.read_content(args.references)]
    hypotheses = [h.strip() for h in args.hypotheses]
    logger.info("%d hypotheses | %d references", len(hypotheses), len(references))

    if not args.not_strict:
        utils.check_condition(len(hypotheses) == len(references),
                              "Number of hypotheses (%d) and references (%d) does not match." % (len(hypotheses),
                                                                                                 len(references)))

    if not args.sentence:
        bleu = corpus_bleu(hypotheses, references, args.offset)
        print(bleu, file=sys.stdout)
    else:
        for h, r in zip(hypotheses, references):
            bleu = bleu_from_counts(bleu_counts(h, r), args.offset)
            print(bleu, file=sys.stdout)


if __name__ == '__main__':
    main()

# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Code for scoring.
"""
import logging
import math
import time
from typing import List

import numpy as np

from .. import constants as C
from ..scoring import ScoringModel
from .. import data_io
from .. import vocab
from ..inference import TranslatorInput, TranslatorOutput
from ..output_handler import OutputHandler

logger = logging.getLogger(__name__)


class Scorer:
    """
    Scorer class takes a ScoringModel and uses it to score a stream of parallel image-sentence pairs.
    It also takes the vocabularies so that the original sentences can be printed out, if desired.

    :param model: The model to score with.
    :param source_vocabs: The source vocabularies. Not used, kept for consistency with main sockeye.score.Scorer.
    :param target_vocab: The target vocabulary.
    """
    def __init__(self,
                 model: ScoringModel,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 constant_length_ratio: float = -1.0) -> None:
        self.target_vocab_inv = vocab.reverse_vocab(target_vocab)
        self.model = model
        self.exclude_list = {None, target_vocab[C.EOS_SYMBOL], C.PAD_ID}
        self.constant_length_ratio = constant_length_ratio

    def score(self,
              score_iter,
              output_handler: OutputHandler):

        total_time = 0.
        sentence_no = 0
        batch_no = 0
        for batch_no, batch in enumerate(score_iter, 1):
            batch_tic = time.time()

            # Run the model and get the outputs
            scores = self.model.run(batch)[0]

            batch_time = time.time() - batch_tic
            total_time += batch_time

            batch_size = len(batch.data[0])

            for sentno, (source, target, score) in enumerate(zip(batch.data[0], batch.data[1], scores), 1):

                # The last batch may be underfilled, in which case batch.pad will be set
                if sentno > (batch_size - batch.pad):
                    break

                sentence_no += 1

                # Transform arguments in preparation for printing
                target_ids = [int(x) for x in target.asnumpy().tolist()]
                target_string = C.TOKEN_SEPARATOR.join(
                    data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))

                # Report a score of -inf for invalid sentence pairs (empty source and/or target)
                if target[0] == C.PAD_ID:
                    score = -np.inf
                else:
                    score = score.asscalar()

                # Output handling routines require us to make use of inference classes.
                output_handler.handle(TranslatorInput(sentence_no, None),
                                      TranslatorOutput(sentence_no, target_string, None, None, score),
                                      batch_time)

        if sentence_no != 0:
            logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                        sentence_no, math.ceil(sentence_no / batch_no), total_time,
                        total_time / sentence_no, sentence_no / total_time)
        else:
            logger.info("Processed 0 lines.")

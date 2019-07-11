# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Dict, List, Optional, Union

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import inference
from . import vocab
from .inference import TranslatorInput, TranslatorOutput
from .model import SockeyeModel
from .output_handler import OutputHandler

logger = logging.getLogger(__name__)


class Scorer:
    """
    Scorer class takes a ScoringModel and uses it to score a stream of parallel sentences.
    It also takes the vocabularies so that the original sentences can be printed out, if desired.

    :param model: The model to score with.
    :param source_vocabs: The source vocabularies.
    :param target_vocab: The target vocabulary.
    """
    def __init__(self,
                 model: SockeyeModel,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 context: Union[List[mx.context.Context], mx.context.Context],
                 length_penalty: inference.LengthPenalty,
                 brevity_penalty: inference.BrevityPenalty,
                 constant_length_ratio: float = 0.0,
                 softmax_temperature: Optional[float] = None,
                 score_type: str = C.SCORING_TYPE_DEFAULT,
                 brevity_penalty_type: str = '') -> None:
        self.source_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
        self.target_vocab_inv = vocab.reverse_vocab(target_vocab)
        self.model = model
        self.context = context
        self.exclude_list = {source_vocabs[0][C.BOS_SYMBOL], target_vocab[C.EOS_SYMBOL], C.PAD_ID}
        self.softmax_temperature = softmax_temperature
        self.score_type = score_type
        self.length_penalty = length_penalty
        self.brevity_penalty = brevity_penalty
        if brevity_penalty_type == C.BREVITY_PENALTY_CONSTANT:
            if constant_length_ratio <= 0.0:
                self.constant_length_ratio = self.model.length_ratio_mean
                logger.info("Using constant length ratio saved in the model config: %f",
                            self.constant_length_ratio)
        else:
            self.constant_length_ratio = -1.0

    def score_batch(self, batch: data_io.Batch) -> mx.nd.NDArray:
        # split batch into shards
        batch = batch.split_and_load(ctx=self.context)
        batch_scores = []  # type: List[mx.nd.NDArray]
        for inputs, labels in batch.shards():
            if self.model.dtype == C.DTYPE_FP16:
                inputs = (i.astype(C.DTYPE_FP16, copy=False) for i in inputs)

            source, source_length, target, target_length = inputs
            outputs = self.model(*inputs)  # type: Dict[str, mx.nd.NDArray]
            logits = outputs[C.LOGITS_NAME]  # type: mx.nd.NDArray

            if self.softmax_temperature is not None:
                logits /= self.softmax_temperature
            target_dists = logits.softmax(axis=-1)

            # Select the label probability, then take their logs.
            # probs and scores: (batch_size, target_seq_len)
            probs = target_dists.pick(labels[C.TARGET_LABEL_NAME], axis=-1)
            token_scores = mx.nd.log(probs)
            if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
                token_scores *= -1

            # Sum, then apply length penalty. The call to `mx.sym.where` masks out invalid values from scores.
            # zeros and sums: (batch_size,)
            scores = mx.nd.sum(mx.nd.where(labels[C.TARGET_LABEL_NAME] != 0,
                                           token_scores,
                                           mx.nd.zeros_like(token_scores)), axis=1) / (
                         self.length_penalty(target_length - 1))

            # Deal with the potential presence of brevity penalty
            # length_ratio: (batch_size,)
            if self.constant_length_ratio > 0.0:
                # override all ratios with the constant value
                length_ratio = self.constant_length_ratio * mx.nd.ones_like(scores)
            else:
                # predict length ratio if supported
                length_ratio = outputs.get(C.LENRATIO_NAME, mx.nd.zeros_like(scores))
            scores = scores - self.brevity_penalty(target_length - 1, length_ratio * source_length)

            batch_scores.append(scores)

        # shape: (batch_size,).
        scores = mx.nd.concat(*batch_scores, dim=0)  # type: mx.nd.NDArray
        return scores

    def score(self, score_iter: data_io.BatchedRawParallelSampleIter, output_handler: OutputHandler):
        total_time = 0.
        sentence_no = 0
        batch_no = 0
        for batch_no, batch in enumerate(score_iter, 1):
            batch_tic = time.time()
            scores = self.score_batch(batch)
            batch_time = time.time() - batch_tic
            total_time += batch_time

            for sentno, (source, target, score) in enumerate(zip(batch.source, batch.target, scores), 1):
                sentence_no += 1

                # Transform arguments in preparation for printing
                source_ids = [int(x) for x in source[:, 0].asnumpy().tolist()]
                source_tokens = list(data_io.ids2tokens(source_ids, self.source_vocab_inv, self.exclude_list))
                target_ids = [int(x) for x in target.asnumpy().tolist()]
                target_string = C.TOKEN_SEPARATOR.join(
                    data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))

                # Report a score of -inf for invalid sentence pairs (empty source and/or target)
                if source[0][0] == C.PAD_ID or target[0] == C.PAD_ID:
                    score = -np.inf
                else:
                    score = score.asscalar()

                # Output handling routines require us to make use of inference classes.
                output_handler.handle(TranslatorInput(sentence_no, source_tokens),
                                      TranslatorOutput(sentence_no, target_string, None, None, score),
                                      batch_time)

        if sentence_no != 0:
            logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                        sentence_no, math.ceil(sentence_no / batch_no), total_time,
                        total_time / sentence_no, sentence_no / total_time)
        else:
            logger.info("Processed 0 lines.")

# Copyright 2018--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Dict, List, Optional

import numpy as np
import torch as pt

from . import constants as C
from . import data_io_pt
from . import inference_pt
from . import vocab
from .beam_search_pt import CandidateScorer
from .model_pt import PyTorchSockeyeModel
from .output_handler import OutputHandler

logger = logging.getLogger(__name__)


class BatchScorer(pt.nn.Module):

    def __init__(self,
                 scorer: CandidateScorer,
                 score_type: str = C.SCORING_TYPE_DEFAULT,
                 constant_length_ratio: Optional[float] = None,
                 softmax_temperature: Optional[float] = None) -> None:
        super().__init__()
        self.score_type = score_type
        self.scorer = scorer
        self.constant_length_ratio = constant_length_ratio
        assert softmax_temperature is None, 'not implemented'

    def forward(self, logits, labels, length_ratio, source_length, target_length):
        """
        :param logits: Model logits. Shape: (batch, length, vocab_size).
        :param labels: Gold targets. Shape: (batch, length).
        :param length_ratio: Length Ratios. Shape: (batch,).
        :param source_length: Source lengths. Shape: (batch,).
        :param target_length: Target lengths. Shape: (batch,).
        :return: Sequence scores. Shape: (batch,).
        """
        logprobs = pt.log_softmax(logits, dim=-1)

        # Select the label log probability
        # logprobs and scores: (batch_size, target_seq_len)
        token_scores = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze()
        if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
            token_scores = token_scores * -1

        # Sum, then apply length penalty. The call to `pt.where` masks out invalid values from scores.
        # zeros and sums: (batch_size,)
        scores = token_scores.where(labels.not_equal(0), pt.zeros_like(token_scores)).sum(dim=1, keepdims=True)

        if self.constant_length_ratio is not None and self.constant_length_ratio > 0.0:
            predicted_output_length = source_length * self.constant_length_ratio
        else:
            predicted_output_length = source_length * length_ratio

        scores = self.scorer(scores, target_length, predicted_output_length)

        return scores


class Scorer:
    """
    Scorer class takes a ScoringModel and uses it to score a stream of parallel sentences.
    It also takes the vocabularies so that the original sentences can be printed out, if desired.

    :param model: The model to score with.
    :param batch_scorer: BatchScorer block to score each batch.
    :param source_vocabs: The source vocabularies.
    :param target_vocabs: The target vocabularies.
    :param device: Torch device to load batches to (should be set to model device).
    """
    def __init__(self,
                 model: PyTorchSockeyeModel,
                 batch_scorer: BatchScorer,
                 source_vocabs: List[vocab.Vocab],
                 target_vocabs: List[vocab.Vocab],
                 device: pt.device) -> None:
        self.source_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
        self.target_vocab_inv = vocab.reverse_vocab(target_vocabs[0])
        self.model = model
        self.traced_model = None  # type: Optional[pt.jit.ScriptModule]
        self.batch_scorer = batch_scorer
        self.traced_batch_scorer = None  # type: Optional[pt.jit.ScriptModule]
        self.device = device
        self.exclude_list = {C.BOS_ID, C.EOS_ID, C.PAD_ID}

    def score_batch(self, batch: data_io_pt.Batch):
        # TODO: scoring should support multiple devices
        batch = batch.load(self.device)

        model_inputs = (batch.source, batch.source_length, batch.target, batch.target_length)
        if self.traced_model is None:
            self.traced_model = pt.jit.trace(self.model, model_inputs, strict=False)
        outputs = self.traced_model(*model_inputs)  # type: Dict[str, pt.Tensor]

        scorer_inputs = (outputs[C.LOGITS_NAME],
                         batch.labels[C.TARGET_LABEL_NAME].long(),
                         outputs.get(C.LENRATIO_NAME, pt.zeros_like(batch.source_length)),
                         batch.source_length,
                         batch.target_length)
        if self.traced_batch_scorer is None:
            self.traced_batch_scorer = pt.jit.trace(self.batch_scorer, scorer_inputs, strict=True)
        scores = self.traced_batch_scorer(*scorer_inputs)

        return scores.squeeze(1).numpy()

    @pt.inference_mode(True)
    def score(self, score_iter: data_io_pt.BaseParallelSampleIter, output_handler: OutputHandler):
        total_time = 0.
        sentence_no = 0
        batch_no = 0
        for batch_no, batch in enumerate(score_iter, 1):
            batch_tic = time.time()
            scores = self.score_batch(batch)
            batch_time = time.time() - batch_tic
            total_time += batch_time
            for sentno, (source, target, score) in enumerate(zip(batch.source[:, :, 0],
                                                                 batch.target[:, :, 0],
                                                                 scores), 1):
                sentence_no += 1

                # Transform arguments in preparation for printing
                source_ids = source.tolist()
                source_tokens = list(data_io_pt.ids2tokens(source_ids, self.source_vocab_inv, self.exclude_list))
                target_ids = target.tolist()
                target_tokens = list(data_io_pt.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))
                target_string = C.TOKEN_SEPARATOR.join(target_tokens)

                # Report a score of -inf for invalid sentence pairs (empty source and/or target)
                if source[0] == C.PAD_ID or target[0] == C.PAD_ID:
                    score = -np.inf

                # Output handling routines require us to make use of inference classes.
                output_handler.handle(inference_pt.TranslatorInput(sentence_no, source_tokens),
                                      inference_pt.TranslatorOutput(sentence_no, target_string,
                                                                    target_tokens, score),
                                      batch_time)

        if sentence_no != 0:
            logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                        sentence_no, math.ceil(sentence_no / batch_no), total_time,
                        total_time / sentence_no, sentence_no / total_time)
        else:
            logger.info("Processed 0 lines.")

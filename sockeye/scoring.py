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
import multiprocessing as mp
import os
import pickle
import random
import shutil
import time
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import mxnet as mx
import numpy as np
from math import sqrt

from . import constants as C
from . import data_io
from . import inference
from . import model
from . import utils
from . import vocab

from .output_handler import OutputHandler
from .inference import TranslatorInput, TranslatorOutput

logger = logging.getLogger(__name__)


class ScoringModel(model.SockeyeModel):
    """
    ScoringModel is a TrainingModel (which is in turn a SockeyeModel) that scores a pair of sentences.
    That is, it full unrolls over source and target sequences, running the encoder and decoder, but stopping short of computing a loss and backpropagating.
    It is analogous to TrainingModel, but more limited.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU).
    :param output_dir: Directory where this model is stored.
    :param provide_data: List of input data descriptions.
    :param provide_label: List of label descriptions.
    :param default_bucket_key: Default bucket key.
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 model_dir: str,
                 context: List[mx.context.Context],
                 provide_data: List[mx.io.DataDesc],
                 provide_label: List[mx.io.DataDesc],
                 bucketing: bool,
                 default_bucket_key: Tuple[int, int],
                 score_type: str,
                 length_penalty: inference.LengthPenalty) -> None:
        super().__init__(config)
        self.context = context
        self.bucketing = bucketing
        self.score_type = score_type
        self.length_penalty = length_penalty

        # Create the computation graph
        self._initialize(provide_data, provide_label, default_bucket_key)

        # Load model parameters into graph
        params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
        self.load_params_from_file(params_fname)
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=False)

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    provide_label: List[mx.io.DataDesc],
                    default_bucket_key: Tuple[int, int]) -> None:
        """
        Initializes model components, creates training symbol and module, and binds it.

        :param provide_data: List of data descriptors.
        :param provide_label: List of label descriptors.
        :param default_bucket_key: The default maximum (source, target) lengths.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)

        # labels shape: (batch_size, target_length) (usually the maximum target sequence length)
        labels = mx.sym.Variable(C.TARGET_LABEL_NAME)

        data_names = [C.SOURCE_NAME, C.TARGET_NAME]
        label_names = [C.TARGET_LABEL_NAME]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))
        provide_label_names = [d[0] for d in provide_label]
        utils.check_condition(provide_label_names == label_names,
                              "incompatible provide_label: %s, names should be %s" % (provide_label_names, label_names))

        def sym_gen(seq_lens):
            """
            Returns a (grouped) symbol containing the summed score for each sentence, as well as the entire target distributions for each word.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(mx.sym.reshape(data=target_decoded, shape=(-3, 0)))
            # logits after reshape: (batch_size, target_seq_len, target_vocab_size)
            logits = mx.sym.reshape(data=logits, shape=(-4,-1,target_embed_seq_len,0))

            # Compute the softmax along the final dimension.
            # target_dists: (batch_size, target_seq_len, target_vocab_size)
            target_dists = mx.sym.softmax(data=logits, axis=2, name=C.SOFTMAX_NAME)

            # Select the label probability, then take their logs.
            probs = mx.sym.pick(target_dists, labels)
            scores = mx.sym.log(probs)
            if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
                scores = -1 * scores

            # Sum and normalize
            # sums: (batch_size,)
            zeros = mx.sym.zeros_like(scores)
            sums = mx.sym.sum(mx.sym.where(labels != 0, scores, zeros), axis=1) / (self.length_penalty(target_length) - 1)

            # Return the sums and the target distributions
            # sums: (batch_size,) target_dists: (batch_size, target_seq_len, target_vocab_size)
            return mx.sym.Group([sums, target_dists]), data_names, label_names

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", default_bucket_key)
            self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=default_bucket_key,
                                                 context=self.context)
        else:
            symbol, _, __ = sym_gen(default_bucket_key)
            self.module = mx.mod.Module(symbol=symbol,
                                        data_names=data_names,
                                        label_names=label_names,
                                        logger=logger,
                                        context=self.context)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=provide_label,
                         for_training=False,
                         force_rebind=False,
                         grad_req=None)

    def run_forward(self, batch: mx.io.DataBatch):
        """
        Runs forward pass.
        """
        self.module.forward(batch, is_train=False)

    def get_outputs(self):
        return self.module.get_outputs()

    def log_parameters(self):
        """
        Logs information about model parameters.
        """
        arg_params, aux_params = self.module.get_params()
        total_parameters = 0
        info = []  # type: List[str]
        for name, array in sorted(arg_params.items()):
            info.append("%s: %s" % (name, array.shape))
            total_parameters += reduce(lambda x, y: x * y, array.shape)
        logger.info("Model parameters: %s", ", ".join(info))
        logger.info("Total # of parameters: %d", total_parameters)

    def load_params_from_file(self, fname: str, allow_missing_params: bool = False):
        """
        Loads parameters from a file and sets the parameters of the underlying module and this model instance.

        :param fname: File name to load parameters from.
        :param allow_missing_params: If set, the given parameters are allowed to be a subset of the Module parameters.
        """
        super().load_params_from_file(fname)  # sets self.params & self.aux_params
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=allow_missing_params)


class Scorer:
    def __init__(self,
                 model: ScoringModel,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab) -> None:
        self.source_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
        self.target_vocab_inv = vocab.reverse_vocab(target_vocab)
        self.model = model

        self.exclude_list = set([source_vocabs[0][C.BOS_SYMBOL], target_vocab[C.EOS_SYMBOL], C.PAD_ID])

    def score(self,
              score_iter,
              score_type: str,
              output_handler: OutputHandler):

        tic = time.time()
        sentence_no = 0
        for i, batch in enumerate(score_iter):

            batch_tic = time.time()

            self.model.run_forward(batch)
            scores, __ = self.model.get_outputs()

            total_time = time.time() - tic
            batch_time = time.time() - batch_tic

            for source, target, score in zip(batch.data[0], batch.data[1], scores):

                # The "zeros" padding method will have filled remainder batches with zeros, so we can skip them here
                if source[0] == 0:
                    break

                sentence_no += 1

                # Transform arguments
                source_ids = [int(x) for x in source[:, 0].asnumpy().tolist()]
                source_tokens = data_io.ids2tokens(source_ids, self.source_vocab_inv, self.exclude_list)
                target_ids = [int(x) for x in target.asnumpy().tolist()]
                target_string = C.TOKEN_SEPARATOR.join(
                    data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))
                score = score.asscalar()

                output_handler.handle(TranslatorInput(sentence_no, source_tokens),
                                      TranslatorOutput(sentence_no, target_string, None, None, score),
                                      batch_time)

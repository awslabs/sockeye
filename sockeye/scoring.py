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

from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import inference
from . import lr_scheduler
from . import model
from . import utils
from . import vocab
from .optimizers import BatchState, CheckpointState, SockeyeOptimizer, OptimizerConfig

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
                 default_bucket_key: Tuple[int, int]) -> None:
        super().__init__(config)
        self.context = context
        self._initialize(provide_data, default_bucket_key)

        params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
        self.load_params_from_file(params_fname)
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=False)

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    default_bucket_key: Tuple[int, int]):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)

        data_names = [C.SOURCE_NAME, C.TARGET_NAME]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))

        def sym_gen(seq_lens):
            """
            Returns a (grouped) softmax symbol given source & target input lengths.
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

            # target_decoded: (batch_size * target_seq_len, decoder_depth)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(target_decoded)
            outputs = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            # return the outputs and the data names (we don't need the labels)
            return outputs, data_names, None

        logger.info("Using bucketing. Default max_seq_len=%s", default_bucket_key)
        self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                             logger=logger,
                                             default_bucket_key=default_bucket_key,
                                             context=self.context)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=None,
                         for_training=False,
                         force_rebind=True,
                         grad_req=None)


    def prepare_batch(self, batch: mx.io.DataBatch):
        """
        Pre-fetches the next mini-batch.

        :param batch: The mini-batch to prepare.
        """
        self.module.prepare(batch)

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
                 target_vocab: vocab.Vocab,
                 length_penalty: Optional[inference.LengthPenalty] = None) -> None:
        self.model = model
        self.length_penalty = length_penalty

    def score(self,
              score_iter):

        for i, batch in enumerate(score_iter):
            # data_io generates labels, too, which we don't need
            label, batch.provide_label = batch.provide_label, None
            labels = batch.label[0]
            self.model.prepare_batch(batch)
            self.model.run_forward(batch)
            outputs = self.model.get_outputs()

            batch_size, target_seq_len, _ = batch.provide_data[0][1]
            outputs = mx.nd.reshape(data=outputs[0], shape=(-4, batch_size, target_seq_len, -2))

            probs = mx.nd.pick(outputs, labels)
            ones = mx.nd.ones_like(probs)
            lengths = mx.nd.sum(labels != 0, axis=1) - 1
            logs = -mx.nd.log(mx.nd.where(labels != 0, probs, ones))
            # print('LOGS', logs)
            # print('LENS', lengths)
            sums = mx.nd.sum(logs, axis=1) / self.length_penalty(lengths)
            # print('SUMS', sums)
            sums = sums.asnumpy().tolist()
            for s in sums:
                print('{:.3f}'.format(s))

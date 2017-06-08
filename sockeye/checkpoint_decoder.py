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
Implements a thin wrapper around Translator to compute BLEU scores on (a sample of) validation data during training.
"""
import logging
import os
import random
from typing import Dict

import mxnet as mx

import sockeye.bleu
import sockeye.inference
import sockeye.output_handler
from sockeye import constants as C
from sockeye.data_io import smart_open

logger = logging.getLogger(__name__)


class CheckpointDecoder:
    """
    Decodes a (random sample of a) dataset using parameters at given checkpoint and computes BLEU against references.

    :param context: MXNet context to bind the model to.
    :param inputs: Path to file containing input sentences.
    :param references: Path to file containing references.
    :param model: Model to load.
    :param max_input_len: Maximum input length.
    :param beam_size: Size of the beam.
    :param limit: Maximum number of sentences to sample and decode. If <=0, all sentences are used.
    """

    def __init__(self,
                 context: mx.context.Context,
                 inputs: str,
                 references: str,
                 model: str,
                 max_input_len: int,
                 beam_size=C.DEFAULT_BEAM_SIZE,
                 limit: int = -1):
        self.context = context
        self.max_input_len = max_input_len
        self.beam_size = beam_size
        self.model = model
        with smart_open(inputs) as inputs_fin, smart_open(references) as references_fin:
            input_sentences = inputs_fin.readlines()
            target_sentences = references_fin.readlines()
            assert len(input_sentences) == len(target_sentences), "Number of sentence pairs do not match"
            if limit <= 0:
                limit = len(input_sentences)
            if limit < len(input_sentences):
                self.input_sentences, self.target_sentences = zip(
                    *random.sample(list(zip(input_sentences, target_sentences)),
                                   limit))
            else:
                self.input_sentences, self.target_sentences = input_sentences, target_sentences

        logger.info("Created CheckpointDecoder(max_input_len=%d, beam_size=%d, model=%s, num_sentences=%d)",
                    max_input_len, beam_size, model, len(self.input_sentences))

        with smart_open(os.path.join(self.model, C.DECODE_REF_NAME), 'w') as trg_out, \
                smart_open(os.path.join(self.model, C.DECODE_IN_NAME), 'w') as src_out:
            [trg_out.write(s) for s in self.target_sentences]
            [src_out.write(s) for s in self.input_sentences]

    def decode_and_evaluate(self, checkpoint: int) -> Dict[str, float]:
        """
        Decodes data set and evaluates given a checkpoint.

        :param checkpoint: Checkpoint to load parameters from.
        :return: Mapping of metric names to scores.
        """
        translator = sockeye.inference.Translator(self.context, 'linear',
                                                  *sockeye.inference.load_models(self.context,
                                                                                 self.max_input_len,
                                                                                 self.beam_size,
                                                                                 [self.model],
                                                                                 [checkpoint]))

        output_name = os.path.join(self.model, C.DECODE_OUT_NAME % checkpoint)
        with smart_open(output_name, 'w') as output:
            handler = sockeye.output_handler.StringOutputHandler(output)
            translations = []
            for sent_id, input_sentence in enumerate(self.input_sentences):
                trans_input = translator.make_input(sent_id, input_sentence)
                trans_output = translator.translate(trans_input)
                handler.handle(trans_input, trans_output)
                translations.append(trans_output.translation)
        logger.info("Checkpoint [%d] %d translations saved to '%s'", checkpoint, len(translations), output_name)
        # TODO(fhieber): eventually add more metrics (METEOR etc.)
        return {"bleu-val": sockeye.bleu.corpus_bleu(translations, self.target_sentences)}

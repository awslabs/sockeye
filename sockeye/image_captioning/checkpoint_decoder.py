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
Implements a thin wrapper around ImageCaptioner to compute BLEU scores on
(a sample of) validation data during training.
"""
import logging
import os
import time
from typing import Dict, Optional

from .. import inference
from . import inference as inference_image
from .. import constants as C
from .. import data_io
from .. import evaluate
from .. import output_handler
from ..checkpoint_decoder import CheckpointDecoder

logger = logging.getLogger(__name__)


class CheckpointDecoderImageModel(CheckpointDecoder):
    """
    Decodes a (random sample of a) dataset using parameters at given checkpoint
    and computes BLEU against references.

    :param source_image_size: Size of the image feed into the net.
    :param image_root: Root where the images are stored.
    :param max_output_length: Max length of the generated sentence.
    :param use_feature_loader: If True, features are loaded instead of images.
    :param kwargs: Arguments passed to `sockeye.checkpoint_decoder.CheckpointDecoder`.
    """

    def __init__(self,
                 source_image_size: tuple,
                 image_root: str,
                 max_output_length: int = 50,
                 use_feature_loader: bool = False,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.source_image_size = source_image_size
        self.image_root = image_root
        self.max_output_length = max_output_length
        self.use_feature_loader = use_feature_loader

    def decode_and_evaluate(self,
                            checkpoint: Optional[int] = None,
                            output_name: str = os.devnull) -> Dict[str, float]:
        """
        Decodes data set and evaluates given a checkpoint.

        :param checkpoint: Checkpoint to load parameters from.
        :param output_name: Filename to write translations to. Defaults to /dev/null.
        :return: Mapping of metric names to scores.
        """

        models, vocab_source, vocab_target = inference_image.load_models(
            context=self.context,
            max_input_len=self.max_input_len,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            model_folders=[self.model],
            checkpoints=[checkpoint],
            softmax_temperature=self.softmax_temperature,
            max_output_length_num_stds=self.max_output_length_num_stds,
            source_image_size=tuple(self.source_image_size),
            forced_max_output_len=self.max_output_length
        )
        translator = inference_image.ImageCaptioner(context=self.context,
                                              ensemble_mode=self.ensemble_mode,
                                              bucket_source_width=0,
                                              length_penalty=inference.LengthPenalty(
                                                  self.length_penalty_alpha,
                                                  self.length_penalty_beta),
                                              beam_prune=0.0,
                                              beam_search_stop='all',
                                              models=models,
                                              source_vocabs=None,
                                              target_vocab=vocab_target,
                                              restrict_lexicon=None,
                                              store_beam=False,
                                              source_image_size=tuple(
                                                  self.source_image_size),
                                              source_root=self.image_root,
                                              use_feature_loader=self.use_feature_loader)

        trans_wall_time = 0.0
        translations = []
        with data_io.smart_open(output_name, 'w') as output:
            handler = output_handler.StringOutputHandler(output)
            tic = time.time()
            trans_inputs = []  # type: List[inference.TranslatorInput]
            for i, inputs in enumerate(self.inputs_sentences):
                trans_inputs.append(
                    inference.make_input_from_multiple_strings(i, inputs))
            trans_outputs = translator.translate(trans_inputs)
            trans_wall_time = time.time() - tic
            for trans_input, trans_output in zip(trans_inputs, trans_outputs):
                handler.handle(trans_input, trans_output)
                translations.append(trans_output.translation)
        avg_time = trans_wall_time / len(self.target_sentences)

        # TODO(fhieber): eventually add more metrics (METEOR etc.)
        return {C.BLEU_VAL: evaluate.raw_corpus_bleu(hypotheses=translations,
                                                     references=self.target_sentences,
                                                     offset=0.01),
                C.CHRF_VAL: evaluate.raw_corpus_chrf(hypotheses=translations,
                                                     references=self.target_sentences),
                C.AVG_TIME: avg_time,
                C.DECODING_TIME: trans_wall_time}

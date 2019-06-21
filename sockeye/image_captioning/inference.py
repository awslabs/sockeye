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
Code for inference/captioning.
"""
import functools
import logging
import os
from typing import List, Optional, Tuple

import mxnet as mx

from . import utils as utils_image
from .. import constants as C
from .. import data_io
from .. import lexical_constraints as constrained
from .. import lexicon
from .. import model
from .. import utils
from .. import vocab
from ..inference import InferenceModel, Translator, \
    TranslatorInput, TranslatorOutput, models_max_input_output_length

logger = logging.getLogger(__name__)


class ImageInferenceModel(InferenceModel):
    """
    ImageInferenceModel is a InferenceModel that supports image models as encoders.
    """

    def __init__(self, input_size, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size

    def _get_encoder_data_shapes(self, bucket_key: int, batch_size: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :param batch_size: Batch size.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(batch_size,) + self.input_size,
                               layout=C.BATCH_MAJOR_IMAGE)]

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return None


class ImageCaptioner(Translator):
    """
    ImageCaptioner uses one or several models to output captions.
    It holds references to vocabularies to takes care of encoding input strings as word ids and conversion
    of target ids into a caption string.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param brevity_penalty: Brevity penalty instance.
    :param beam_prune: Beam pruning difference threshold.
    :param beam_search_stop: The stopping criterium.
    :param models: List of models.
    :param vocab_target: Target vocabulary.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary restriction.
    :param source_image_size: Shape of the image, input of the net
    :param source_root: Root where the images are stored
    :param use_feature_loader: Use precomputed features
    :param store_beam: If True, store the beam search history and return it in the TranslatorOutput.
    :param strip_unknown_words: If True, removes any <unk> symbols from outputs.
    """

    def __init__(self,
                 source_image_size: tuple,
                 source_root: str,
                 use_feature_loader: bool,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.source_image_size = source_image_size
        self.source_root = source_root
        self.use_feature_loader = use_feature_loader
        if self.use_feature_loader:
            self.data_loader = utils_image.load_features
        else:
            self.data_loader = functools.partial(utils_image.load_preprocess_images,
                                                 image_size=self.source_image_size)

    def translate(self, trans_inputs: List[TranslatorInput]) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Splits oversized sentences to sentence chunks of size less than max_input_length.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :return: List of translation results.
        """
        batch_size = self.max_batch_size
        # translate in batch-sized blocks over input chunks
        translations = []
        for batch_id, batch in enumerate(utils.grouper(trans_inputs, batch_size)):
            logger.debug("Translating batch %d", batch_id)
            # underfilled batch will be filled to a full batch size with copies of the 1st input
            rest = batch_size - len(batch)
            if rest > 0:
                logger.debug("Extending the last batch to the full batch size (%d)", batch_size)
                batch = batch + [batch[0]] * rest
            batch_translations = self._translate_nd(*self._get_inference_input(batch))
            # truncate to remove filler translations
            if rest > 0:
                batch_translations = batch_translations[:-rest]
            translations.extend(batch_translations)

        # Concatenate results
        results = []  # type: List[TranslatorOutput]
        for trans_input, translation in zip(trans_inputs, translations):
            results.append(self._make_result(trans_input, translation))
        return results

    def _get_inference_input(self,
                             trans_inputs: List[TranslatorInput]) -> Tuple[mx.nd.NDArray,
                                                                           int,
                                                                           Optional[lexicon.TopKLexicon],
                                                                           List[
                                                                               Optional[constrained.RawConstraintList]],
                                                                           List[
                                                                               Optional[constrained.RawConstraintList]],
                                                                           mx.nd.NDArray]:
        """
        Returns NDArray of images and corresponding bucket_key and an NDArray of maximum output lengths
        for each sentence in the batch.

        :param trans_inputs: List of TranslatorInputs. The path of the image/feature is in the token field.
        :param constraints: Optional list of constraints.
        :return: NDArray of images paths, bucket key, a list of raw constraint lists,
                an NDArray of maximum output lengths.
        """
        batch_size = len(trans_inputs)
        image_paths = [None for _ in range(batch_size)]  # type: List[Optional[str]]
        restrict_lexicon = None  # type: Optional[lexicon.TopKLexicon]
        raw_constraints = [None for _ in range(batch_size)]  # type: List[Optional[constrained.RawConstraintList]]
        raw_avoid_list = [None for _ in range(batch_size)]  # type: List[Optional[constrained.RawConstraintList]]
        for j, trans_input in enumerate(trans_inputs):
            # Join relative path with absolute
            path = trans_input.tokens[0]
            if self.source_root is not None:
                path = os.path.join(self.source_root, path)
            image_paths[j] = path
            # Preprocess constraints
            if trans_input.constraints is not None:
                raw_constraints[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                      trans_input.constraints]

        # Read data and zero pad if necessary
        images = self.data_loader(image_paths)
        images = utils_image.zero_pad_features(images, self.source_image_size)

        max_input_length = 0
        max_output_lengths = [self.models[0].get_max_output_length(max_input_length)] * len(image_paths)
        return mx.nd.array(images), max_input_length, restrict_lexicon, raw_constraints, raw_avoid_list, \
                mx.nd.array(max_output_lengths, ctx=self.context, dtype='int32')


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                beam_size: int,
                batch_size: int,
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None,
                max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                decoder_return_logit_inputs: bool = False,
                cache_output_layer_w_b: bool = False,
                source_image_size: tuple = None,
                forced_max_output_len: Optional[int] = None) -> Tuple[List[ImageInferenceModel], vocab.Vocab]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param batch_size: Batch size.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations to add to mean target-source length ratio
           to compute maximum output length.
    :param decoder_return_logit_inputs: Model decoders return inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Models cache weights and biases for logit computation as NumPy arrays (used with
                                   restrict lexicon).
    :param source_image_size: Size of the image to resize to. Used only for the image-text models
    :param forced_max_output_len: An optional overwrite of the maximum out length.
    :return: List of models, target vocabulary, source factor vocabularies.
    """
    models = []  # type: List[ImageInferenceModel]
    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        target_vocabs.append(vocab.vocab_from_json(os.path.join(model_folder, C.VOCAB_TRG_NAME)))

        model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", model_version)
        utils.check_version(model_version)
        model_config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))

        if checkpoint is None:
            params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

        inference_model = ImageInferenceModel(config=model_config,
                                              params_fname=params_fname,
                                              context=context,
                                              beam_size=beam_size,
                                              softmax_temperature=softmax_temperature,
                                              decoder_return_logit_inputs=decoder_return_logit_inputs,
                                              cache_output_layer_w_b=cache_output_layer_w_b,
                                              input_size=source_image_size,
                                              forced_max_output_len=forced_max_output_len)

        models.append(inference_model)

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = models_max_input_output_length(models,
                                                                          max_output_length_num_stds,
                                                                          max_input_len,
                                                                          forced_max_output_len=forced_max_output_len)

    for inference_model in models:
        inference_model.initialize(batch_size, max_input_len, get_max_output_length)

    return models, target_vocabs[0]

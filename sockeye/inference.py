# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Code for inference/translation
"""
import copy
import itertools
import json
import logging
import os
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, Set, Any

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import lexical_constraints as constrained
from . import lexicon
from . import model
from . import utils
from . import vocab
from .log import is_python34

logger = logging.getLogger(__name__)


class InferenceModel(model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param config: Configuration object holding details about the model.
    :param params_fname: File with model parameters.
    :param context: MXNet context to bind modules to.
    :param beam_size: Beam size.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    :param skip_softmax: If True, does not compute softmax for greedy decoding.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 beam_size: int,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False,
                 forced_max_output_len: Optional[int] = None,
                 skip_softmax: bool = False) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.beam_size = beam_size
        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')
        if skip_softmax:
            assert beam_size == 1, 'Skipping softmax does not have any effect for beam size > 1'
        self.skip_softmax = skip_softmax

        self.softmax_temperature = softmax_temperature
        self.max_input_length, self.get_max_output_length = models_max_input_output_length([self],
                                                                                           max_output_length_num_stds,
                                                                                           forced_max_output_len=forced_max_output_len)

        self.max_batch_size = None  # type: Optional[int]
        self.encoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.encoder_default_bucket_key = None  # type: Optional[int]
        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[Tuple[int, int]]
        self.decoder_return_logit_inputs = decoder_return_logit_inputs

        self.cache_output_layer_w_b = cache_output_layer_w_b
        self.output_layer_w = None  # type: Optional[mx.nd.NDArray]
        self.output_layer_b = None  # type: Optional[mx.nd.NDArray]

    @property
    def num_source_factors(self) -> int:
        """
        Returns the number of source factors of this InferenceModel (at least 1).
        """
        return self.config.config_data.num_source_factors

    def initialize(self, max_batch_size: int, max_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_batch_size: Maximum batch size.
        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        if self.max_input_length > self.training_max_seq_len_source:
            logger.warning("Model was only trained with sentences up to a length of %d, "
                           "but a max_input_len of %d is used.",
                           self.training_max_seq_len_source, self.max_input_length)
        self.get_max_output_length = get_max_output_length_function

        # check the maximum supported length of the encoder & decoder:
        if self.max_supported_seq_len_source is not None:
            utils.check_condition(self.max_input_length <= self.max_supported_seq_len_source,
                                  "Encoder only supports a maximum length of %d" % self.max_supported_seq_len_source)
        if self.max_supported_seq_len_target is not None:
            decoder_max_len = self.get_max_output_length(max_input_length)
            utils.check_condition(decoder_max_len <= self.max_supported_seq_len_target,
                                  "Decoder only supports a maximum length of %d, but %d was requested. Note that the "
                                  "maximum output length depends on the input length and the source/target length "
                                  "ratio observed during training." % (self.max_supported_seq_len_target,
                                                                       decoder_max_len))

        self.encoder_module, self.encoder_default_bucket_key = self._get_encoder_module()
        self.decoder_module, self.decoder_default_bucket_key = self._get_decoder_module()

        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key,
                                                                self.max_batch_size)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key,
                                                                self.max_batch_size * self.beam_size)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.params_fname)
        self.encoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)
        self.decoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)

        if self.cache_output_layer_w_b:
            if self.output_layer.weight_normalization:
                # precompute normalized output layer weight imperatively
                assert self.output_layer.weight_norm is not None
                weight = self.params[self.output_layer.weight_norm.weight.name].as_in_context(self.context)
                scale = self.params[self.output_layer.weight_norm.scale.name].as_in_context(self.context)
                self.output_layer_w = self.output_layer.weight_norm(weight, scale)
            else:
                self.output_layer_w = self.params[self.output_layer.w.name].as_in_context(self.context)
            self.output_layer_b = self.params[self.output_layer.b.name].as_in_context(self.context)

    def _get_encoder_module(self) -> Tuple[mx.mod.BucketingModule, int]:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Tuple of encoder module and default bucket key.
        """

        def sym_gen(source_seq_len: int):
            source = mx.sym.Variable(C.SOURCE_NAME)
            source_words = source.split(num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            source_length = utils.compute_lengths(source_words)

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # encoder
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # initial decoder states
            decoder_init_states = self.decoder.init_states(source_encoded,
                                                           source_encoded_length,
                                                           source_encoded_seq_len)

            data_names = [C.SOURCE_NAME]
            label_names = []  # type: List[str]

            # predict length ratios
            predicted_length_ratios = []  # type: List[mx.nd.NDArray]
            if self.length_ratio is not None:
                # predicted_length_ratios: List[(n, 1)]
                predicted_length_ratios = [self.length_ratio(source_encoded, source_encoded_length)]

            return mx.sym.Group(decoder_init_states + predicted_length_ratios), data_names, label_names

        default_bucket_key = self.max_input_length
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_decoder_module(self) -> Tuple[mx.mod.BucketingModule, Tuple[int, int]]:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the length of the source sequence
        and the current time-step in the inference procedure (e.g. beam search).
        The latter corresponds to the current length of the target sequences.

        :return: Tuple of decoder module and default bucket key.
        """

        def sym_gen(bucket_key: Tuple[int, int]):
            """
            Returns either softmax output (probs over target vocabulary) or inputs to logit
            computation, controlled by decoder_return_logit_inputs
            """
            source_seq_len, decode_step = bucket_key
            source_embed_seq_len = self.embedding_source.get_encoded_seq_len(source_seq_len)
            source_encoded_seq_len = self.encoder.get_encoded_seq_len(source_embed_seq_len)

            self.decoder.reset()
            target_prev = mx.sym.Variable(C.TARGET_NAME)
            states = self.decoder.state_variables(decode_step)
            state_names = [state.name for state in states]

            # embedding for previous word
            # (batch_size, num_embed)
            target_embed_prev, _, _ = self.embedding_target.encode(data=target_prev, data_length=None, seq_len=1)

            # decoder
            # target_decoded: (batch_size, decoder_depth)
            (target_decoded,
             attention_probs,
             states) = self.decoder.decode_step(decode_step,
                                                target_embed_prev,
                                                source_encoded_seq_len,
                                                *states)

            if self.decoder_return_logit_inputs:
                # skip output layer in graph
                outputs = mx.sym.identity(target_decoded, name=C.LOGIT_INPUTS_NAME)
            else:
                # logits: (batch_size, target_vocab_size)
                logits = self.output_layer(target_decoded)
                if self.softmax_temperature is not None:
                    logits = logits / self.softmax_temperature
                if self.skip_softmax:
                    # skip softmax for greedy decoding
                    outputs = logits
                else:
                    outputs = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            data_names = [C.TARGET_NAME] + state_names
            label_names = []  # type: List[str]
            return mx.sym.Group([outputs, attention_probs] + states), data_names, label_names

        # pylint: disable=not-callable
        default_bucket_key = (self.max_input_length, self.get_max_output_length(self.max_input_length))
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_encoder_data_shapes(self, bucket_key: int, batch_size: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(batch_size, bucket_key, self.num_source_factors),
                               layout=C.BATCH_MAJOR)]

    @lru_cache(maxsize=None)
    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int], batch_beam_size: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :param batch_beam_size: Batch size * beam size.
        :return: List of data descriptions.
        """
        source_max_length, target_max_length = bucket_key
        return [mx.io.DataDesc(name=C.TARGET_NAME, shape=(batch_beam_size,),
                               layout="NT")] + self.decoder.state_shapes(batch_beam_size,
                                                                         target_max_length,
                                                                         self.encoder.get_encoded_seq_len(
                                                                             source_max_length),
                                                                         self.encoder.get_num_hidden())

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int) -> Tuple['ModelState', mx.nd.NDArray]:
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens. Shape (batch_size, source length, num_source_factors).
        :param source_max_length: Bucket key.
        :return: Initial model state.
        """
        batch_size = source.shape[0]
        batch = mx.io.DataBatch(data=[source],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length, batch_size))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_init_states = self.encoder_module.get_outputs()

        if self.length_ratio is not None:
            estimated_length_ratio = decoder_init_states[-1]
            estimated_length_ratio = mx.nd.repeat(estimated_length_ratio, repeats=self.beam_size, axis=0)
            decoder_init_states = decoder_init_states[:-1]
        else:
            estimated_length_ratio = None
            decoder_init_states = decoder_init_states
        # replicate encoder/init module results beam size times
        decoder_init_states = [mx.nd.repeat(s, repeats=self.beam_size, axis=0) for s in decoder_init_states]
        return ModelState(decoder_init_states), estimated_length_ratio

    def run_decoder(self,
                    prev_word: mx.nd.NDArray,
                    bucket_key: Tuple[int, int],
                    model_state: 'ModelState') -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState']:
        """
        Runs forward pass of the single-step decoder.

        :param prev_word: Previous word ids. Shape: (batch*beam,).
        :param bucket_key: Bucket key.
        :param model_state: Model states.
        :return: Decoder stack output (logit inputs or probability distribution), attention scores, updated model state.
        """
        batch_beam_size = prev_word.shape[0]
        batch = mx.io.DataBatch(
            data=[prev_word.as_in_context(self.context)] + model_state.states,
            label=None,
            bucket_key=bucket_key,
            provide_data=self._get_decoder_data_shapes(bucket_key, batch_beam_size))
        self.decoder_module.forward(data_batch=batch, is_train=False)
        out, attention_probs, *model_state.states = self.decoder_module.get_outputs()
        return out, attention_probs, model_state

    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.encoder.get_max_seq_len()

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.decoder.get_max_seq_len()

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std

    @property
    def source_with_eos(self) -> bool:
        return self.config.config_data.source_with_eos


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
                forced_max_output_len: Optional[int] = None,
                override_dtype: Optional[str] = None,
                output_scores: bool = False,
                sampling: bool = False) -> Tuple[List[InferenceModel],
                                                 List[vocab.Vocab],
                                                 vocab.Vocab]:
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
    :param forced_max_output_len: An optional overwrite of the maximum output length.
    :param override_dtype: Overrides dtype of encoder and decoder defined at training time to a different one.
    :param output_scores: Whether the scores will be needed as outputs. If True, scores will be normalized, negative
           log probabilities. If False, scores will be negative, raw logit activations if decoding with beam size 1
           and a single model.
    :param sampling: True if the model is sampling instead of doing normal topk().
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    """
    logger.info("Loading %d model(s) from %s ...", len(model_folders), model_folders)
    load_time_start = time.time()
    models = []  # type: List[InferenceModel]
    source_vocabs = []  # type: List[List[vocab.Vocab]]
    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    else:
        utils.check_condition(len(checkpoints) == len(model_folders), "Must provide checkpoints for each model")

    skip_softmax = False
    # performance tweak: skip softmax for a single model, decoding with beam size 1, when not sampling and no scores are required in output.
    if len(model_folders) == 1 and beam_size == 1 and not output_scores and not sampling:
        skip_softmax = True
        logger.info("Enabled skipping softmax for a single model and greedy decoding.")

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model_source_vocabs = vocab.load_source_vocabs(model_folder)
        model_target_vocab = vocab.load_target_vocab(model_folder)
        source_vocabs.append(model_source_vocabs)
        target_vocabs.append(model_target_vocab)

        model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", model_version)
        utils.check_version(model_version)
        model_config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))

        logger.info("Disabling dropout layers for performance reasons")
        model_config.disable_dropout()

        if override_dtype is not None:
            model_config.config_encoder.dtype = override_dtype
            model_config.config_decoder.dtype = override_dtype
            if override_dtype == C.DTYPE_FP16:
                logger.warning('Experimental feature \'override_dtype=float16\' has been used. '
                               'This feature may be removed or change its behaviour in future. '
                               'DO NOT USE IT IN PRODUCTION!')

        if checkpoint is None:
            params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

        inference_model = InferenceModel(config=model_config,
                                         params_fname=params_fname,
                                         context=context,
                                         beam_size=beam_size,
                                         softmax_temperature=softmax_temperature,
                                         decoder_return_logit_inputs=decoder_return_logit_inputs,
                                         cache_output_layer_w_b=cache_output_layer_w_b,
                                         skip_softmax=skip_softmax)
        utils.check_condition(inference_model.num_source_factors == len(model_source_vocabs),
                              "Number of loaded source vocabularies (%d) does not match "
                              "number of source factors for model '%s' (%d)" % (len(model_source_vocabs), model_folder,
                                                                                inference_model.num_source_factors))
        models.append(inference_model)

    utils.check_condition(vocab.are_identical(*target_vocabs), "Target vocabulary ids do not match")
    first_model_vocabs = source_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[source_vocabs[i][fi] for i in range(len(source_vocabs))]),
                              "Source vocabulary ids do not match. Factor %d" % fi)

    source_with_eos = models[0].source_with_eos
    utils.check_condition(all(source_with_eos == m.source_with_eos for m in models),
                          "All models must agree on using source-side EOS symbols or not. "
                          "Did you try combining models trained with different versions?")

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = models_max_input_output_length(models,
                                                                          max_output_length_num_stds,
                                                                          max_input_len,
                                                                          forced_max_output_len=forced_max_output_len)

    for inference_model in models:
        inference_model.initialize(batch_size, max_input_len, get_max_output_length)

    load_time = time.time() - load_time_start
    logger.info("%d model(s) loaded in %.4fs", len(models), load_time)
    return models, source_vocabs[0], target_vocabs[0]


def models_max_input_output_length(models: List[InferenceModel],
                                   num_stds: int,
                                   forced_max_input_len: Optional[int] = None,
                                   forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param forced_max_input_len: An optional overwrite of the maximum input length.
    :param forced_max_output_len: An optional overwrite of the maximum output length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    max_mean = max(model.length_ratio_mean for model in models)
    max_std = max(model.length_ratio_std for model in models)

    supported_max_seq_len_source = min((model.max_supported_seq_len_source for model in models
                                        if model.max_supported_seq_len_source is not None),
                                       default=None)
    supported_max_seq_len_target = min((model.max_supported_seq_len_target for model in models
                                        if model.max_supported_seq_len_target is not None),
                                       default=None)
    training_max_seq_len_source = min(model.training_max_seq_len_source for model in models)

    return get_max_input_output_length(supported_max_seq_len_source,
                                       supported_max_seq_len_target,
                                       training_max_seq_len_source,
                                       length_ratio_mean=max_mean,
                                       length_ratio_std=max_std,
                                       num_stds=num_stds,
                                       forced_max_input_len=forced_max_input_len,
                                       forced_max_output_len=forced_max_output_len)


def get_max_input_output_length(supported_max_seq_len_source: Optional[int],
                                supported_max_seq_len_target: Optional[int],
                                training_max_seq_len_source: Optional[int],
                                length_ratio_mean: float,
                                length_ratio_std: float,
                                num_stds: int,
                                forced_max_input_len: Optional[int] = None,
                                forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length. It takes into account optional maximum source and target lengths.

    :param supported_max_seq_len_source: The maximum source length supported by the models.
    :param supported_max_seq_len_target: The maximum target length supported by the models.
    :param training_max_seq_len_source: The maximum source length observed during training.
    :param length_ratio_mean: The mean of the length ratio that was calculated on the raw sequences with special
           symbols such as EOS or BOS.
    :param length_ratio_std: The standard deviation of the length ratio.
    :param num_stds: The number of standard deviations the target length may exceed the mean target length (as long as
           the supported maximum length allows for this).
    :param forced_max_input_len: An optional overwrite of the maximum input length.
    :param forced_max_output_len: An optional overwrite of the maximum out length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    space_for_bos = 1
    space_for_eos = 1

    if num_stds < 0:
        factor = C.TARGET_MAX_LENGTH_FACTOR  # type: float
    else:
        factor = length_ratio_mean + (length_ratio_std * num_stds)

    if forced_max_input_len is None:
        # Make sure that if there is a hard constraint on the maximum source or target length we never exceed this
        # constraint. This is for example the case for learned positional embeddings, which are only defined for the
        # maximum source and target sequence length observed during training.
        if supported_max_seq_len_source is not None and supported_max_seq_len_target is None:
            max_input_len = supported_max_seq_len_source
        elif supported_max_seq_len_source is None and supported_max_seq_len_target is not None:
            max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
            if np.ceil(factor * training_max_seq_len_source) > max_output_len:
                max_input_len = int(np.floor(max_output_len / factor))
            else:
                max_input_len = training_max_seq_len_source
        elif supported_max_seq_len_source is not None or supported_max_seq_len_target is not None:
            max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
            if np.ceil(factor * supported_max_seq_len_source) > max_output_len:
                max_input_len = int(np.floor(max_output_len / factor))
            else:
                max_input_len = supported_max_seq_len_source
        else:
            # Any source/target length is supported and max_input_len was not manually set, therefore we use the
            # maximum length from training.
            max_input_len = training_max_seq_len_source
    else:
        max_input_len = forced_max_input_len

    def get_max_output_length(input_length: int):
        """
        Returns the maximum output length for inference given the input length.
        Explicitly includes space for BOS and EOS sentence symbols in the target sequence, because we assume
        that the mean length ratio computed on the training data do not include these special symbols.
        (see data_io.analyze_sequence_lengths)
        """
        if forced_max_output_len is not None:
            return forced_max_output_len
        else:
            return int(np.ceil(factor * input_length)) + space_for_bos + space_for_eos

    return max_input_len, get_max_output_length


BeamHistory = Dict[str, List]
Tokens = List[str]
SentenceId = Union[int, str]


class TranslatorInput:
    """
    Object required by Translator.translate().
    If not None, `pass_through_dict` is an arbitrary dictionary instantiated from a JSON object
    via `make_input_from_dict()`, and it contains extra fields found in an input JSON object.
    If `--output-type json` is selected, all such fields that are not fields used or changed by
    Sockeye will be included in the output JSON object. This provides a mechanism for passing
    fields through the call to Sockeye.

    :param sentence_id: Sentence id.
    :param tokens: List of input tokens.
    :param factors: Optional list of additional factor sequences.
    :param restrict_lexicon: Optional lexicon for vocabulary selection.
    :param constraints: Optional list of target-side constraints.
    :param pass_through_dict: Optional raw dictionary of arbitrary input data.
    """

    __slots__ = ('sentence_id',
                 'tokens',
                 'factors',
                 'restrict_lexicon',
                 'constraints',
                 'avoid_list',
                 'pass_through_dict')

    def __init__(self,
                 sentence_id: SentenceId,
                 tokens: Tokens,
                 factors: Optional[List[Tokens]] = None,
                 restrict_lexicon: Optional[lexicon.TopKLexicon] = None,
                 constraints: Optional[List[Tokens]] = None,
                 avoid_list: Optional[List[Tokens]] = None,
                 pass_through_dict: Optional[Dict] = None) -> None:
        self.sentence_id = sentence_id
        self.tokens = tokens
        self.factors = factors
        self.restrict_lexicon = restrict_lexicon
        self.constraints = constraints
        self.avoid_list = avoid_list
        self.pass_through_dict = pass_through_dict

    def __str__(self):
        return 'TranslatorInput(%s, %s, factors=%s, constraints=%s, avoid=%s)' \
            % (self.sentence_id, self.tokens, self.factors, self.constraints, self.avoid_list)

    def __len__(self):
        return len(self.tokens)

    @property
    def num_factors(self) -> int:
        """
        Returns the number of factors of this instance.
        """
        return 1 + (0 if not self.factors else len(self.factors))

    def chunks(self, chunk_size: int) -> Generator['TranslatorInput', None, None]:
        """
        Takes a TranslatorInput (itself) and yields TranslatorInputs for chunks of size chunk_size.

        :param chunk_size: The maximum size of a chunk.
        :return: A generator of TranslatorInputs, one for each chunk created.
        """

        if len(self.tokens) > chunk_size and self.constraints is not None:
            logger.warning(
                'Input %s has length (%d) that exceeds max input length (%d), '
                'triggering internal splitting. Placing all target-side constraints '
                'with the first chunk, which is probably wrong.',
                self.sentence_id, len(self.tokens), chunk_size)

        for chunk_id, i in enumerate(range(0, len(self), chunk_size)):
            factors = [factor[i:i + chunk_size] for factor in self.factors] if self.factors is not None else None
            # Constrained decoding is not supported for chunked TranslatorInputs. As a fall-back, constraints are
            # assigned to the first chunk
            constraints = self.constraints if chunk_id == 0 else None
            pass_through_dict = self.pass_through_dict if chunk_id == 0 else None
            yield TranslatorInput(sentence_id=self.sentence_id,
                                  tokens=self.tokens[i:i + chunk_size],
                                  factors=factors,
                                  restrict_lexicon=self.restrict_lexicon,
                                  constraints=constraints,
                                  avoid_list=self.avoid_list,
                                  pass_through_dict=pass_through_dict)

    def with_eos(self) -> 'TranslatorInput':
        """
        :return: A new translator input with EOS appended to the tokens and factors.
        """
        return TranslatorInput(sentence_id=self.sentence_id,
                               tokens=self.tokens + [C.EOS_SYMBOL],
                               factors=[factor + [C.EOS_SYMBOL] for factor in
                                        self.factors] if self.factors is not None else None,
                               restrict_lexicon=self.restrict_lexicon,
                               constraints=self.constraints,
                               avoid_list=self.avoid_list,
                               pass_through_dict=self.pass_through_dict)


class BadTranslatorInput(TranslatorInput):

    def __init__(self, sentence_id: SentenceId, tokens: Tokens) -> None:
        super().__init__(sentence_id=sentence_id, tokens=tokens, factors=None)


def _bad_input(sentence_id: SentenceId, reason: str = '') -> BadTranslatorInput:
    logger.warning("Bad input (%s): '%s'. Will return empty output.", sentence_id, reason.strip())
    return BadTranslatorInput(sentence_id=sentence_id, tokens=[])


def make_input_from_plain_string(sentence_id: SentenceId, string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a plain string.

    :param sentence_id: Sentence id.
    :param string: An input string.
    :return: A TranslatorInput.
    """
    return TranslatorInput(sentence_id, tokens=list(data_io.get_tokens(string)), factors=None)


def make_input_from_json_string(sentence_id: SentenceId,
                                json_string: str,
                                translator: 'Translator') -> TranslatorInput:
    """
    Returns a TranslatorInput object from a JSON object, serialized as a string.

    :param sentence_id: Sentence id.
    :param json_string: A JSON object serialized as a string that must contain a key "text", mapping to the input text,
           and optionally a key "factors" that maps to a list of strings, each of which representing a factor sequence
           for the input text. Constraints and an avoid list can also be added through the "constraints" and "avoid"
           keys.
    :param translator: A translator object.
    :return: A TranslatorInput.
    """
    try:
        jobj = json.loads(json_string, encoding=C.JSON_ENCODING)
        return make_input_from_dict(sentence_id, jobj, translator)

    except Exception as e:
        logger.exception(e, exc_info=True) if not is_python34() else logger.error(e)  # type: ignore
        return _bad_input(sentence_id, reason=json_string)


def make_input_from_dict(sentence_id: SentenceId,
                         input_dict: Dict,
                         translator: 'Translator') -> TranslatorInput:
    """
    Returns a TranslatorInput object from a JSON object, serialized as a string.

    :param sentence_id: Sentence id.
    :param input_dict: A dict that must contain a key "text", mapping to the input text, and optionally a key "factors"
           that maps to a list of strings, each of which representing a factor sequence for the input text.
           Constraints and an avoid list can also be added through the "constraints" and "avoid" keys.
    :param translator: A translator object.
    :return: A TranslatorInput.
    """
    try:
        tokens = input_dict[C.JSON_TEXT_KEY]
        tokens = list(data_io.get_tokens(tokens))
        factors = input_dict.get(C.JSON_FACTORS_KEY)
        if isinstance(factors, list):
            factors = [list(data_io.get_tokens(factor)) for factor in factors]
            lengths = [len(f) for f in factors]
            if not all(length == len(tokens) for length in lengths):
                logger.error("Factors have different length than input text: %d vs. %s", len(tokens), str(lengths))
                return _bad_input(sentence_id, reason=str(input_dict))

        # Lexicon for vocabulary selection/restriction:
        # This is only populated when using multiple lexicons, in which case the
        # restrict_lexicon key must exist and the value (name) must map to one
        # of the translator's known lexicons.
        restrict_lexicon = None
        restrict_lexicon_name = input_dict.get(C.JSON_RESTRICT_LEXICON_KEY)
        if isinstance(translator.restrict_lexicon, dict):
            if restrict_lexicon_name is None:
                logger.error("Must specify restrict_lexicon when using multiple lexicons. Choices: %s"
                             % ' '.join(sorted(translator.restrict_lexicon)))
                return _bad_input(sentence_id, reason=str(input_dict))
            restrict_lexicon = translator.restrict_lexicon.get(restrict_lexicon_name, None)
            if restrict_lexicon is None:
                logger.error("Unknown restrict_lexicon '%s'. Choices: %s"
                             % (restrict_lexicon_name, ' '.join(sorted(translator.restrict_lexicon))))
                return _bad_input(sentence_id, reason=str(input_dict))

        # List of phrases to prevent from occuring in the output
        avoid_list = input_dict.get(C.JSON_AVOID_KEY)

        # List of phrases that must appear in the output
        constraints = input_dict.get(C.JSON_CONSTRAINTS_KEY)

        # If there is overlap between positive and negative constraints, assume the user wanted
        # the words, and so remove them from the avoid_list (negative constraints)
        if constraints is not None and avoid_list is not None:
            avoid_set = set(avoid_list)
            overlap = set(constraints).intersection(avoid_set)
            if len(overlap) > 0:
                logger.warning("Overlap between constraints and avoid set, dropping the overlapping avoids")
                avoid_list = list(avoid_set.difference(overlap))

        # Convert to a list of tokens
        if isinstance(avoid_list, list):
            avoid_list = [list(data_io.get_tokens(phrase)) for phrase in avoid_list]
        if isinstance(constraints, list):
            constraints = [list(data_io.get_tokens(constraint)) for constraint in constraints]

        return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors,
                               restrict_lexicon=restrict_lexicon, constraints=constraints,
                               avoid_list=avoid_list, pass_through_dict=input_dict)

    except Exception as e:
        logger.exception(e, exc_info=True) if not is_python34() else logger.error(e)  # type: ignore
        return _bad_input(sentence_id, reason=str(input_dict))


def make_input_from_factored_string(sentence_id: SentenceId,
                                    factored_string: str,
                                    translator: 'Translator',
                                    delimiter: str = C.DEFAULT_FACTOR_DELIMITER) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a string with factor annotations on a token level, separated by delimiter.
    If translator does not require any source factors, the string is parsed as a plain token string.

    :param sentence_id: Sentence id.
    :param factored_string: An input string with additional factors per token, separated by delimiter.
    :param translator: A translator object.
    :param delimiter: A factor delimiter. Default: '|'.
    :return: A TranslatorInput.
    """
    utils.check_condition(bool(delimiter) and not delimiter.isspace(),
                          "Factor delimiter can not be whitespace or empty.")

    model_num_source_factors = translator.num_source_factors

    if model_num_source_factors == 1:
        return make_input_from_plain_string(sentence_id=sentence_id, string=factored_string)

    tokens = []  # type: Tokens
    factors = [[] for _ in range(model_num_source_factors - 1)]  # type: List[Tokens]
    for token_id, token in enumerate(data_io.get_tokens(factored_string)):
        pieces = token.split(delimiter)

        if not all(pieces) or len(pieces) != model_num_source_factors:
            logger.error("Failed to parse %d factors at position %d ('%s') in '%s'" % (model_num_source_factors,
                                                                                       token_id, token,
                                                                                       factored_string.strip()))
            return _bad_input(sentence_id, reason=factored_string)

        tokens.append(pieces[0])
        for i, factor in enumerate(factors):
            factors[i].append(pieces[i + 1])

    return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)


def make_input_from_multiple_strings(sentence_id: SentenceId, strings: List[str]) -> TranslatorInput:
    """
    Returns a TranslatorInput object from multiple strings, where the first element corresponds to the surface tokens
    and the remaining elements to additional factors. All strings must parse into token sequences of the same length.

    :param sentence_id: Sentence id.
    :param strings: A list of strings representing a factored input sequence.
    :return: A TranslatorInput.
    """
    if not bool(strings):
        return TranslatorInput(sentence_id=sentence_id, tokens=[], factors=None)

    tokens = list(data_io.get_tokens(strings[0]))
    factors = [list(data_io.get_tokens(factor)) for factor in strings[1:]]
    if not all(len(factor) == len(tokens) for factor in factors):
        logger.error("Length of string sequences do not match: '%s'", strings)
        return _bad_input(sentence_id, reason=str(strings))
    return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)


class TranslatorOutput:
    """
    Output structure from Translator.

    :param sentence_id: Sentence id.
    :param translation: Translation string without sentence boundary tokens.
    :param tokens: List of translated tokens.
    :param attention_matrix: Attention matrix. Shape: (target_length, source_length).
    :param score: Negative log probability of generated translation.
    :param pass_through_dict: Dictionary of key/value pairs to pass through when working with JSON.
    :param beam_histories: List of beam histories. The list will contain more than one
           history if it was split due to exceeding max_length.
    :param nbest_translations: List of nbest translations as strings.
    :param nbest_tokens: List of nbest translations as lists of tokens.
    :param nbest_attention_matrices: List of attention matrices, one for each nbest translation.
    :param nbest_scores: List of nbest scores, one for each nbest translation.
    """
    __slots__ = ('sentence_id',
                 'translation',
                 'tokens',
                 'attention_matrix',
                 'score',
                 'pass_through_dict',
                 'beam_histories',
                 'nbest_translations',
                 'nbest_tokens',
                 'nbest_attention_matrices',
                 'nbest_scores')

    def __init__(self,
                 sentence_id: SentenceId,
                 translation: str,
                 tokens: Tokens,
                 attention_matrix: np.ndarray,
                 score: float,
                 pass_through_dict: Optional[Dict[str,Any]] = None,
                 beam_histories: Optional[List[BeamHistory]] = None,
                 nbest_translations: Optional[List[str]] = None,
                 nbest_tokens: Optional[List[Tokens]] = None,
                 nbest_attention_matrices: Optional[List[np.ndarray]] = None,
                 nbest_scores: Optional[List[float]] = None) -> None:
        self.sentence_id = sentence_id
        self.translation = translation
        self.tokens = tokens
        self.attention_matrix = attention_matrix
        self.score = score
        self.pass_through_dict = copy.deepcopy(pass_through_dict) if pass_through_dict else {}
        self.beam_histories = beam_histories
        self.nbest_translations = nbest_translations
        self.nbest_tokens = nbest_tokens
        self.nbest_attention_matrices = nbest_attention_matrices
        self.nbest_scores = nbest_scores

    def json(self, align_threshold: float = 0.0) -> Dict:
        """
        Returns a dictionary suitable for json.dumps() representing all
        the information in the class. It is initialized with any keys
        present in the corresponding `TranslatorInput` object's pass_through_dict.
        Keys from here that are not overwritten by Sockeye will thus be passed
        through to the output.

        :param align_threshold: If alignments are defined, only print ones over this threshold.
        :return: A dictionary.
        """
        _d = self.pass_through_dict  # type: Dict[str, Any]
        _d['sentence_id'] = self.sentence_id
        _d['translation'] = self.translation
        _d['score'] = self.score

        if self.nbest_translations is not None and len(self.nbest_translations) > 1:
            _d['translations'] = self.nbest_translations
            _d['scores'] = self.nbest_scores
            if self.nbest_attention_matrices:
                extracted_alignments = []
                for alignment_matrix in self.nbest_attention_matrices:
                    extracted_alignments.append(list(utils.get_alignments(alignment_matrix, threshold=align_threshold)))
                _d['alignments'] = extracted_alignments

        return _d


TokenIds = List[int]


class NBestTranslations:
    __slots__ = ('target_ids_list',
                 'attention_matrices',
                 'scores')

    def __init__(self,
                 target_ids_list: List[TokenIds],
                 attention_matrices: List[np.ndarray],
                 scores: List[float]) -> None:

        self.target_ids_list = target_ids_list
        self.attention_matrices = attention_matrices
        self.scores = scores


class Translation:
    __slots__ = ('target_ids',
                 'attention_matrix',
                 'score',
                 'beam_histories',
                 'nbest_translations',
                 'estimated_reference_length')

    def __init__(self,
                 target_ids: TokenIds,
                 attention_matrix: np.ndarray,
                 score: float,
                 beam_histories: List[BeamHistory] = None,
                 nbest_translations: NBestTranslations = None,
                 estimated_reference_length: Optional[float] = None) -> None:
        self.target_ids = target_ids
        self.attention_matrix = attention_matrix
        self.score = score
        self.beam_histories = beam_histories if beam_histories is not None else []
        self.nbest_translations = nbest_translations
        self.estimated_reference_length = estimated_reference_length


def empty_translation(add_nbest: bool = False) -> Translation:
    """
    Return an empty translation.

    :param add_nbest: Include (empty) nbest_translations in the translation object.
    """
    return Translation(target_ids=[],
                       attention_matrix=np.asarray([[0]]),
                       score=-np.inf,
                       nbest_translations=NBestTranslations([], [], []) if add_nbest else None)


IndexedTranslatorInput = NamedTuple('IndexedTranslatorInput', [
    ('input_idx', int),
    ('chunk_idx', int),
    ('translator_input', TranslatorInput)
])
"""
Translation of a chunk of a sentence.

:param input_idx: Internal index of translation requests to keep track of the correct order of translations.
:param chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
:param input: The translator input.
"""


IndexedTranslation = NamedTuple('IndexedTranslation', [
    ('input_idx', int),
    ('chunk_idx', int),
    ('translation', Translation)
])
"""
Translation of a chunk of a sentence.

:param input_idx: Internal index of translation requests to keep track of the correct order of translations.
:param chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
:param translation: The translation of the input chunk.
"""


class ModelState:
    """
    A ModelState encapsulates information about the decoder states of an InferenceModel.
    """

    def __init__(self, states: List[mx.nd.NDArray]) -> None:
        self.states = states

    def sort_state(self, best_hyp_indices: mx.nd.NDArray):
        """
        Sorts states according to k-best order from last step in beam search.
        """
        self.states = [mx.nd.take(ds, best_hyp_indices) for ds in self.states]


class LengthPenalty(mx.gluon.HybridBlock):
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def hybrid_forward(self, F, lengths):
        if self.alpha == 0.0:
            if F is None:
                return 1.0
            else:
                return F.ones_like(lengths)
        else:
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator

    def get(self, lengths: Union[mx.nd.NDArray, int, float]) -> Union[mx.nd.NDArray, float]:
        """
        Calculate the length penalty for the given vector of lengths.

        :param lengths: A scalar or a matrix of sentence lengths of dimensionality (batch_size, 1).
        :return: The length penalty. A scalar or a matrix (batch_size, 1) depending on the input.
        """
        return self.hybrid_forward(None, lengths)


class BrevityPenalty(mx.gluon.HybridBlock):
    """
    Calculates the logarithmic brevity penalty as:
      weight * log min(1, exp(1 - ref_len / hyp_len)) = weight * min(0, 1 - ref_len / hyp_len).

    :param weight: Linear weight.
    """

    def __init__(self, weight: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = weight

    def hybrid_forward(self, F, hyp_lengths, reference_lengths):
        if self.weight == 0.0:
            if F is None:
                return 0.0
            else:
                # subtract to avoid MxNet's warning of not using both arguments
                # this branch should not and is not used during inference
                return F.zeros_like(hyp_lengths - reference_lengths)
        else:
            # log_bp is always <= 0.0
            if F is None:
                log_bp = min(0.0, 1.0 - reference_lengths / hyp_lengths)
            else:
                log_bp = F.minimum(F.zeros_like(hyp_lengths), 1.0 - reference_lengths / hyp_lengths)
            return self.weight * log_bp

    def get(self,
            hyp_lengths: Union[mx.nd.NDArray, int, float],
            reference_lengths: Optional[Union[mx.nd.NDArray, int, float]]) -> Union[mx.nd.NDArray, float]:
        """
        Calculate the length penalty for the given vector of lengths.

        :param hyp_lengths: Hypotheses lengths.
        :param reference_lengths: Reference lengths.
        :return: The length penalty. A scalar or a matrix (batch_size, 1) depending on the input.
        """
        if reference_lengths is None:
            return 0.0
        else:
            return self.hybrid_forward(None, hyp_lengths, reference_lengths)


def _concat_nbest_translations(translations: List[Translation], stop_ids: Set[int],
                               length_penalty: LengthPenalty,
                               brevity_penalty: Optional[BrevityPenalty] = None) -> Translation:
    """
    Combines nbest translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol,
        attention_matrix), score and length.
    :param stop_ids: The EOS symbols.
    :param length_penalty: LengthPenalty.
    :param brevity_penalty: Optional BrevityPenalty.
    :return: A concatenation of the translations with a score.
    """
    expanded_translations = (_expand_nbest_translation(translation) for translation in translations)

    concatenated_translations = []  # type: List[Translation]

    for translations_to_concat in zip(*expanded_translations):
        concatenated_translations.append(_concat_translations(translations=list(translations_to_concat),
                                                              stop_ids=stop_ids,
                                                              length_penalty=length_penalty,
                                                              brevity_penalty=brevity_penalty))

    return _reduce_nbest_translations(concatenated_translations)


def _reduce_nbest_translations(nbest_translations_list: List[Translation]) -> Translation:
    """
    Combines Translation objects that are nbest translations of the same sentence.

    :param nbest_translations_list: A list of Translation objects, all of them translations of
        the same source sentence.
    :return: A single Translation object where nbest lists are collapsed.
    """
    best_translation = nbest_translations_list[0]

    sequences = [translation.target_ids for translation in nbest_translations_list]
    attention_matrices = [translation.attention_matrix for translation in nbest_translations_list]
    scores = [translation.score for translation in nbest_translations_list]

    nbest_translations = NBestTranslations(sequences, attention_matrices, scores)

    return Translation(best_translation.target_ids,
                       best_translation.attention_matrix,
                       best_translation.score,
                       best_translation.beam_histories,
                       nbest_translations,
                       best_translation.estimated_reference_length)


def _expand_nbest_translation(translation: Translation) -> List[Translation]:
    """
    Expand nbest translations in a single Translation object to one Translation
        object per nbest translation.

    :param translation: A Translation object.
    :return: A list of Translation objects.
    """
    nbest_list = []  # type = List[Translation]
    for target_ids, attention_matrix, score in zip(translation.nbest_translations.target_ids_list,
                                                   translation.nbest_translations.attention_matrices,
                                                   translation.nbest_translations.scores):
        nbest_list.append(Translation(target_ids, attention_matrix, score, translation.beam_histories,
                                      estimated_reference_length=translation.estimated_reference_length))

    return nbest_list


def _concat_translations(translations: List[Translation],
                         stop_ids: Set[int],
                         length_penalty: LengthPenalty,
                         brevity_penalty: Optional[BrevityPenalty] = None) -> Translation:
    """
    Combines translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol, attention_matrix), score and length.
    :param stop_ids: The EOS symbols.
    :param length_penalty: Instance of the LengthPenalty class initialized with alpha and beta.
    :param brevity_penalty: Optional Instance of the BrevityPenalty class initialized with a brevity weight.
    :return: A concatenation of the translations with a score.
    """
    # Concatenation of all target ids without BOS and EOS
    target_ids = []
    attention_matrices = []
    beam_histories = []  # type: List[BeamHistory]
    estimated_reference_length = None  # type: float

    for idx, translation in enumerate(translations):
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids)
            attention_matrices.append(translation.attention_matrix)
        else:
            if translation.target_ids[-1] in stop_ids:
                target_ids.extend(translation.target_ids[:-1])
                attention_matrices.append(translation.attention_matrix[:-1, :])
            else:
                target_ids.extend(translation.target_ids)
                attention_matrices.append(translation.attention_matrix)
        beam_histories.extend(translation.beam_histories)
        if translation.estimated_reference_length is not None:
            if estimated_reference_length is None:
                estimated_reference_length = translation.estimated_reference_length
            else:
                estimated_reference_length += translation.estimated_reference_length
    # Combine attention matrices:
    attention_shapes = [attention_matrix.shape for attention_matrix in attention_matrices]
    attention_matrix_combined = np.zeros(np.sum(np.asarray(attention_shapes), axis=0))
    pos_t, pos_s = 0, 0
    for attention_matrix, (len_t, len_s) in zip(attention_matrices, attention_shapes):
        attention_matrix_combined[pos_t:pos_t + len_t, pos_s:pos_s + len_s] = attention_matrix
        pos_t += len_t
        pos_s += len_s

    def _brevity_penalty(hypothesis_length, reference_length):
        return 0.0 if brevity_penalty is None else brevity_penalty.get(hypothesis_length, reference_length)

    # Unnormalize + sum and renormalize the score:
    score = sum((translation.score + _brevity_penalty(len(translation.target_ids), translation.estimated_reference_length)) \
                    * length_penalty.get(len(translation.target_ids))
                 for translation in translations)
    score = score / length_penalty.get(len(target_ids)) - _brevity_penalty(len(target_ids), estimated_reference_length)
    return Translation(target_ids, attention_matrix_combined, score, beam_histories,
                       estimated_reference_length=estimated_reference_length)


class Translator:
    """
    Translator uses one or several models to translate input.
    The translator holds a reference to vocabularies to convert between word ids and text tokens for input and
    translation strings.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param beam_prune: Beam pruning difference threshold.
    :param beam_search_stop: The stopping criterion.
    :param models: List of models.
    :param source_vocabs: Source vocabularies.
    :param target_vocab: Target vocabulary.
    :param nbest_size: Size of nbest list of translations.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary selection. Can be a dict of
                             of named lexicons.
    :param avoid_list: Global list of phrases to exclude from the output.
    :param store_beam: If True, store the beam search history and return it in the TranslatorOutput.
    :param strip_unknown_words: If True, removes any <unk> symbols from outputs.
    :param skip_topk: If True, uses argmax instead of topk for greedy decoding.
    :param sample: If True, sample from softmax multinomial instead of using topk.
    :param constant_length_ratio: If > 0, will override models' prediction of the length ratio (if any).
    :param brevity_penalty: Optional BrevityPenalty.
    """

    def __init__(self,
                 context: mx.context.Context,
                 ensemble_mode: str,
                 bucket_source_width: int,
                 length_penalty: LengthPenalty,
                 beam_prune: float,
                 beam_search_stop: str,
                 models: List[InferenceModel],
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 nbest_size: int = 1,
                 restrict_lexicon: Optional[Union[lexicon.TopKLexicon, Dict[str, lexicon.TopKLexicon]]] = None,
                 avoid_list: Optional[str] = None,
                 store_beam: bool = False,
                 strip_unknown_words: bool = False,
                 skip_topk: bool = False,
                 sample: int = None,
                 constant_length_ratio: float = 0.0,
                 brevity_penalty: Optional[BrevityPenalty] = None) -> None:
        self.context = context
        self.length_penalty = length_penalty
        self.brevity_penalty = brevity_penalty
        self.constant_length_ratio = constant_length_ratio
        self.beam_prune = beam_prune
        self.beam_search_stop = beam_search_stop
        self.source_vocabs = source_vocabs
        self.vocab_target = target_vocab
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.restrict_lexicon = restrict_lexicon
        self.store_beam = store_beam
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        assert C.PAD_ID == 0, "pad id should be 0"
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}  # type: Set[int]
        self.strip_ids = self.stop_ids.copy()  # ids to strip from the output
        self.unk_id = self.vocab_target[C.UNK_SYMBOL]
        if strip_unknown_words:
            self.strip_ids.add(self.unk_id)
        self.models = models
        utils.check_condition(all(models[0].source_with_eos == m.source_with_eos for m in models),
                              "The source_with_eos property must match across models.")
        self.source_with_eos = models[0].source_with_eos
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        self.nbest_size = nbest_size
        utils.check_condition(self.beam_size >= nbest_size, 'nbest_size must be smaller or equal to beam_size.')
        if self.nbest_size > 1:
            utils.check_condition(self.beam_search_stop == C.BEAM_SEARCH_STOP_ALL,
                                  "nbest_size > 1 requires beam_search_stop to be set to 'all'")

        # maximum allowed batch size of this translator instance
        self.batch_size = self.models[0].max_batch_size

        if any(m.skip_softmax for m in self.models):
            utils.check_condition(len(self.models) == 1 and self.beam_size == 1,
                                  "Skipping softmax cannot be enabled for ensembles or beam sizes > 1.")

        self.skip_topk = skip_topk
        if self.skip_topk:
            utils.check_condition(self.beam_size == 1, "skip_topk has no effect if beam size is larger than 1")
            utils.check_condition(len(self.models) == 1, "skip_topk has no effect for decoding with more than 1 model")

        self.sample = sample
        utils.check_condition(not self.sample or self.restrict_lexicon is None,
                              "Sampling is not available when working with a restricted lexicon.")

        # after models are loaded we ensured that they agree on max_input_length, max_output_length and batch size
        self._max_input_length = self.models[0].max_input_length
        if bucket_source_width > 0:
            self.buckets_source = data_io.define_buckets(self._max_input_length, step=bucket_source_width)
        else:
            self.buckets_source = [self._max_input_length]

        self._update_scores = UpdateScores()
        self._update_scores.initialize(ctx=self.context)
        self._update_scores.hybridize(static_alloc=True, static_shape=True)

        # Vocabulary selection leads to different vocabulary sizes across requests. Hence, we cannot use a
        # statically-shaped HybridBlock for the topk operation in this case; resorting to imperative topk
        # function in this case.
        if not self.restrict_lexicon:
            if self.skip_topk:
                self._top = Top1()  # type: mx.gluon.HybridBlock
            elif self.sample is not None:
                self._top = SampleK(k=self.beam_size,
                                    n=self.sample,
                                    max_batch_size=self.max_batch_size)  # type: mx.gluon.HybridBlock
            else:
                self._top = TopK(k=self.beam_size,
                                 vocab_size=len(self.vocab_target))  # type: mx.gluon.HybridBlock

            self._top.initialize(ctx=self.context)
            self._top.hybridize(static_alloc=True, static_shape=True)
        else:
            if self.skip_topk:
                self._top = utils.top1  # type: Callable
            else:
                self._top = partial(utils.topk, k=self.beam_size)  # type: Callable

        self._sort_by_index = SortByIndex()
        self._sort_by_index.initialize(ctx=self.context)
        self._sort_by_index.hybridize(static_alloc=True, static_shape=True)

        brevity_penalty_weight = self.brevity_penalty.weight if self.brevity_penalty is not None else 0.0
        self._update_finished = NormalizeAndUpdateFinished(pad_id=C.PAD_ID,
                                                           eos_id=self.vocab_target[C.EOS_SYMBOL],
                                                           length_penalty_alpha=self.length_penalty.alpha,
                                                           length_penalty_beta=self.length_penalty.beta,
                                                           brevity_penalty_weight=brevity_penalty_weight)
        self._update_finished.initialize(ctx=self.context)
        self._update_finished.hybridize(static_alloc=True, static_shape=True)

        self._prune_hyps = PruneHypotheses(threshold=self.beam_prune, beam_size=self.beam_size)
        self._prune_hyps.initialize(ctx=self.context)
        self._prune_hyps.hybridize(static_alloc=True, static_shape=True)

        self.global_avoid_trie = None
        if avoid_list is not None:
            self.global_avoid_trie = constrained.AvoidTrie()
            for phrase in data_io.read_content(avoid_list):
                phrase_ids = data_io.tokens2ids(phrase, self.vocab_target)
                if self.unk_id in phrase_ids:
                    logger.warning("Global avoid phrase '%s' contains an %s; this may indicate improper preprocessing.",
                                   ' '.join(phrase), C.UNK_SYMBOL)
                self.global_avoid_trie.add_phrase(phrase_ids)

        self._concat_translations = partial(_concat_nbest_translations if self.nbest_size > 1 else _concat_translations,
                                            stop_ids=self.stop_ids,
                                            length_penalty=self.length_penalty,
                                            brevity_penalty=self.brevity_penalty)  # type: Callable

        logger.info("Translator (%d model(s) beam_size=%d beam_prune=%s beam_search_stop=%s "
                    "nbest_size=%s ensemble_mode=%s max_batch_size=%d buckets_source=%s avoiding=%d)",
                    len(self.models),
                    self.beam_size,
                    'off' if not self.beam_prune else "%.2f" % self.beam_prune,
                    self.beam_search_stop,
                    self.nbest_size,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.max_batch_size,
                    self.buckets_source,
                    0 if self.global_avoid_trie is None else len(self.global_avoid_trie))

    @property
    def max_input_length(self) -> int:
        """
        Returns maximum input length for TranslatorInput objects passed to translate()
        """
        if self.source_with_eos:
            return self._max_input_length - C.SPACE_FOR_XOS
        else:
            return self._max_input_length

    @property
    def max_batch_size(self) -> int:
        """
        Returns the maximum batch size allowed for this Translator.
        """
        return self.batch_size

    @property
    def num_source_factors(self) -> int:
        return self.models[0].num_source_factors

    @staticmethod
    def _get_interpolation_func(ensemble_mode):
        if ensemble_mode == 'linear':
            return Translator._linear_interpolation
        elif ensemble_mode == 'log_linear':
            return Translator._log_linear_interpolation
        else:
            raise ValueError("unknown interpolation type")

    @staticmethod
    def _linear_interpolation(predictions):
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = utils.average_arrays([p.log() for p in predictions])
        # pylint: disable=invalid-unary-operand-type
        return -log_probs.log_softmax()

    def translate(self, trans_inputs: List[TranslatorInput], fill_up_batches: bool = True) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Empty or bad inputs are skipped.
        Splits inputs longer than Translator.max_input_length into segments of size max_input_length,
        and then groups segments into batches of at most Translator.max_batch_size.
        Too-long segments that were split are reassembled into a single output after translation.
        If fill_up_batches is set to True, underfilled batches are padded to Translator.max_batch_size, otherwise
        dynamic batch sizing is used, which comes at increased memory usage.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :param fill_up_batches: If True, underfilled batches are padded to Translator.max_batch_size.
        :return: List of translation results.
        """
        num_inputs = len(trans_inputs)
        translated_chunks = []  # type: List[IndexedTranslation]

        # split into chunks
        input_chunks = []  # type: List[IndexedTranslatorInput]
        for trans_input_idx, trans_input in enumerate(trans_inputs):
            # bad input
            if isinstance(trans_input, BadTranslatorInput):
                translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0,
                                                            translation=empty_translation(add_nbest=(self.nbest_size > 1))))
            # empty input
            elif len(trans_input.tokens) == 0:
                translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0,
                                                            translation=empty_translation(add_nbest=(self.nbest_size > 1))))
            else:
                # TODO(tdomhan): Remove branch without EOS with next major version bump, as future models will always be trained with source side EOS symbols
                if self.source_with_eos:
                    max_input_length_without_eos = self.max_input_length
                    # oversized input
                    if len(trans_input.tokens) > max_input_length_without_eos:
                        logger.debug(
                            "Input %s has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            self.buckets_source[-1], max_input_length_without_eos)
                        chunks = [trans_input_chunk.with_eos()
                                  for trans_input_chunk in trans_input.chunks(max_input_length_without_eos)]
                        input_chunks.extend([IndexedTranslatorInput(trans_input_idx, chunk_idx, chunk_input)
                                             for chunk_idx, chunk_input in enumerate(chunks)])
                    # regular input
                    else:
                        input_chunks.append(IndexedTranslatorInput(trans_input_idx,
                                                                   chunk_idx=0,
                                                                   translator_input=trans_input.with_eos()))
                else:
                    if len(trans_input.tokens) > self.max_input_length:
                        # oversized input
                        logger.debug(
                            "Input %s has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            self.buckets_source[-1], self.max_input_length)
                        chunks = [trans_input_chunk
                                  for trans_input_chunk in
                                  trans_input.chunks(self.max_input_length)]
                        input_chunks.extend([IndexedTranslatorInput(trans_input_idx, chunk_idx, chunk_input)
                                             for chunk_idx, chunk_input in enumerate(chunks)])
                    else:
                        # regular input
                        input_chunks.append(IndexedTranslatorInput(trans_input_idx,
                                                                   chunk_idx=0,
                                                                   translator_input=trans_input))

            if trans_input.constraints is not None:
                logger.info("Input %s has %d %s: %s", trans_input.sentence_id,
                            len(trans_input.constraints),
                            "constraint" if len(trans_input.constraints) == 1 else "constraints",
                            ", ".join(" ".join(x) for x in trans_input.constraints))

        num_bad_empty = len(translated_chunks)

        # Sort longest to shortest (to rather fill batches of shorter than longer sequences)
        input_chunks = sorted(input_chunks, key=lambda chunk: len(chunk.translator_input.tokens), reverse=True)
        # translate in batch-sized blocks over input chunks
        batch_size = self.max_batch_size if fill_up_batches else min(len(input_chunks), self.max_batch_size)

        num_batches = 0
        for batch_id, batch in enumerate(utils.grouper(input_chunks, batch_size)):
            logger.debug("Translating batch %d", batch_id)

            rest = batch_size - len(batch)
            if fill_up_batches and rest > 0:
                logger.debug("Padding batch of size %d to full batch size (%d)", len(batch), batch_size)
                batch = batch + [batch[0]] * rest

            translator_inputs = [indexed_translator_input.translator_input for indexed_translator_input in batch]
            batch_translations = self._translate_nd(*self._get_inference_input(translator_inputs))

            # truncate to remove filler translations
            if fill_up_batches and rest > 0:
                batch_translations = batch_translations[:-rest]

            for chunk, translation in zip(batch, batch_translations):
                translated_chunks.append(IndexedTranslation(chunk.input_idx, chunk.chunk_idx, translation))
            num_batches += 1
        # Sort by input idx and then chunk id
        translated_chunks = sorted(translated_chunks)
        num_chunks = len(translated_chunks)

        # Concatenate results
        results = []  # type: List[TranslatorOutput]
        chunks_by_input_idx = itertools.groupby(translated_chunks, key=lambda translation: translation.input_idx)
        for trans_input, (input_idx, translations_for_input_idx) in zip(trans_inputs, chunks_by_input_idx):
            translations_for_input_idx = list(translations_for_input_idx)  # type: ignore
            if len(translations_for_input_idx) == 1:  # type: ignore
                translation = translations_for_input_idx[0].translation  # type: ignore
            else:
                translations_to_concat = [translated_chunk.translation
                                          for translated_chunk in translations_for_input_idx]
                translation = self._concat_translations(translations_to_concat)

            results.append(self._make_result(trans_input, translation))

        num_outputs = len(results)

        logger.debug("Translated %d inputs (%d chunks) in %d batches to %d outputs. %d empty/bad inputs.",
                     num_inputs, num_chunks, num_batches, num_outputs, num_bad_empty)

        return results

    def _get_inference_input(self,
                             trans_inputs: List[TranslatorInput]) -> Tuple[mx.nd.NDArray,
                                                                           int,
                                                                           Optional[lexicon.TopKLexicon],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           mx.nd.NDArray]:
        """
        Assembles the numerical data for the batch. This comprises an NDArray for the source sentences,
        the bucket key (padded source length), and a list of raw constraint lists, one for each sentence in the batch,
        an NDArray of maximum output lengths for each sentence in the batch.
        Each raw constraint list contains phrases in the form of lists of integers in the target language vocabulary.

        :param trans_inputs: List of TranslatorInputs.
        :return NDArray of source ids (shape=(batch_size, bucket_key, num_factors)),
                bucket key, lexicon for vocabulary restriction, list of raw constraint
                lists, and list of phrases to avoid, and an NDArray of maximum output
                lengths.
        """
        batch_size = len(trans_inputs)
        bucket_key = data_io.get_bucket(max(len(inp.tokens) for inp in trans_inputs), self.buckets_source)
        source = mx.nd.zeros((batch_size, bucket_key, self.num_source_factors), ctx=self.context)
        restrict_lexicon = None  # type: Optional[lexicon.TopKLexicon]
        raw_constraints = [None] * batch_size  # type: List[Optional[constrained.RawConstraintList]]
        raw_avoid_list = [None] * batch_size  # type: List[Optional[constrained.RawConstraintList]]

        max_output_lengths = []  # type: List[int]
        for j, trans_input in enumerate(trans_inputs):
            num_tokens = len(trans_input)
            max_output_lengths.append(self.models[0].get_max_output_length(data_io.get_bucket(num_tokens, self.buckets_source)))
            source[j, :num_tokens, 0] = data_io.tokens2ids(trans_input.tokens, self.source_vocabs[0])

            factors = trans_input.factors if trans_input.factors is not None else []
            num_factors = 1 + len(factors)
            if num_factors != self.num_source_factors:
                logger.warning("Input %d factors, but model(s) expect %d", num_factors,
                               self.num_source_factors)
            for i, factor in enumerate(factors[:self.num_source_factors - 1], start=1):
                # fill in as many factors as there are tokens

                source[j, :num_tokens, i] = data_io.tokens2ids(factor, self.source_vocabs[i])[:num_tokens]

            # Check if vocabulary selection/restriction is enabled:
            # - First, see if the translator input provides a lexicon (used for multiple lexicons)
            # - If not, see if the translator itself provides a lexicon (used for single lexicon)
            # - The same lexicon must be used for all inputs in the batch.
            if trans_input.restrict_lexicon is not None:
                if restrict_lexicon is not None and restrict_lexicon is not trans_input.restrict_lexicon:
                    logger.warning("Sentence %s: different restrict_lexicon specified, will overrule previous. "
                                   "All inputs in batch must use same lexicon." % trans_input.sentence_id)
                restrict_lexicon = trans_input.restrict_lexicon
            elif self.restrict_lexicon is not None:
                if isinstance(self.restrict_lexicon, dict):
                    # This code should not be reachable since the case is checked when creating
                    # translator inputs. It is included here to guarantee that the translator can
                    # handle any valid input regardless of whether it was checked at creation time.
                    logger.warning("Sentence %s: no restrict_lexicon specified for input when using multiple lexicons, "
                                   "defaulting to first lexicon for entire batch." % trans_input.sentence_id)
                    restrict_lexicon = list(self.restrict_lexicon.values())[0]
                else:
                    restrict_lexicon = self.restrict_lexicon

            if trans_input.constraints is not None:
                raw_constraints[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                      trans_input.constraints]

            if trans_input.avoid_list is not None:
                raw_avoid_list[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                     trans_input.avoid_list]
                if any(self.unk_id in phrase for phrase in raw_avoid_list[j]):
                    logger.warning("Sentence %s: %s was found in the list of phrases to avoid; "
                                   "this may indicate improper preprocessing.", trans_input.sentence_id, C.UNK_SYMBOL)

        return source, bucket_key, restrict_lexicon, raw_constraints, raw_avoid_list, \
                mx.nd.array(max_output_lengths, ctx=self.context, dtype='int32')

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrices and scores.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation + attention and score.
        :return: TranslatorOutput.
        """
        target_ids = translation.target_ids
        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]
        target_string = C.TOKEN_SEPARATOR.join(data_io.ids2tokens(target_ids, self.vocab_target_inv, self.strip_ids))

        attention_matrix = translation.attention_matrix
        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]

        if translation.nbest_translations is None:
            return TranslatorOutput(sentence_id=trans_input.sentence_id,
                                    translation=target_string,
                                    tokens=target_tokens,
                                    attention_matrix=attention_matrix,
                                    score=translation.score,
                                    pass_through_dict=trans_input.pass_through_dict,
                                    beam_histories=translation.beam_histories)
        else:
            nbest_target_ids = translation.nbest_translations.target_ids_list
            target_tokens_list = [[self.vocab_target_inv[id] for id in ids] for ids in nbest_target_ids]
            target_strings = [C.TOKEN_SEPARATOR.join(
                                data_io.ids2tokens(target_ids,
                                                   self.vocab_target_inv,
                                                   self.strip_ids)) for target_ids in nbest_target_ids]

            attention_matrices = [matrix[:, :len(trans_input.tokens)] for matrix in
                                  translation.nbest_translations.attention_matrices]

            scores = translation.nbest_translations.scores

            return TranslatorOutput(sentence_id=trans_input.sentence_id,
                                    translation=target_string,
                                    tokens=target_tokens,
                                    attention_matrix=attention_matrix,
                                    score=translation.score,
                                    pass_through_dict=trans_input.pass_through_dict,
                                    beam_histories=translation.beam_histories,
                                    nbest_translations=target_strings,
                                    nbest_tokens=target_tokens_list,
                                    nbest_attention_matrices=attention_matrices,
                                    nbest_scores=scores)

    def _translate_nd(self,
                      source: mx.nd.NDArray,
                      source_length: int,
                      restrict_lexicon: Optional[lexicon.TopKLexicon],
                      raw_constraints: List[Optional[constrained.RawConstraintList]],
                      raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                      max_output_lengths: mx.nd.NDArray) -> List[Translation]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :param restrict_lexicon: Lexicon to use for vocabulary restriction.
        :param raw_constraints: A list of optional constraint lists.

        :return: Sequence of translations.
        """
        return self._get_best_from_beam(*self._beam_search(source,
                                                           source_length,
                                                           restrict_lexicon,
                                                           raw_constraints,
                                                           raw_avoid_list,
                                                           max_output_lengths))

    def _encode(self, sources: mx.nd.NDArray, source_length: int) -> Tuple[List[ModelState], mx.nd.NDArray]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param sources: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :return: List of ModelStates and the estimated reference length based on ratios averaged over models.
        """
        model_states = []
        ratios = []
        for model in self.models:
            state, ratio = model.run_encoder(sources, source_length)
            model_states.append(state)
            if ratio is not None:
                ratios.append(ratio)

        # num_seq takes batch_size and beam_size into account
        num_seq = model_states[0].states[0].shape[0]
        if self.constant_length_ratio > 0.0:
            # override all ratios with the constant value
            length_ratios = mx.nd.full(val=self.constant_length_ratio, shape=(num_seq, 1), ctx=self.context)
        else:
            if len(ratios) > 0:  # some model predicted a ratio?
                # average the ratios over the models that actually we able to predict them
                length_ratios = mx.nd.mean(mx.nd.stack(*ratios, axis=1), axis=1)
            else:
                length_ratios = mx.nd.zeros((num_seq, 1), ctx=self.context)

        encoded_source_length=self.models[0].encoder.get_encoded_seq_len(source_length)
        return model_states, length_ratios * encoded_source_length


    def _decode_step(self,
                     prev_word: mx.nd.NDArray,
                     step: int,
                     source_length: int,
                     states: List[ModelState],
                     models_output_layer_w: List[mx.nd.NDArray],
                     models_output_layer_b: List[mx.nd.NDArray]) \
            -> Tuple[mx.nd.NDArray, mx.nd.NDArray, List[ModelState]]:
        """
        Returns decoder predictions (combined from all models), attention scores, and updated states.

        :param prev_word: Previous words of hypotheses. Shape: (batch_size * beam_size,).
        :param step: Beam search iteration.
        :param source_length: Length of the input sequence.
        :param states: List of model states.
        :param models_output_layer_w: Custom model weights for logit computation (empty for none).
        :param models_output_layer_b: Custom model biases for logit computation (empty for none).
        :return: (scores, attention scores, list of model states)
        """
        bucket_key = (source_length, step)

        model_outs, model_attention_probs, model_states = [], [], []
        # We use zip_longest here since we'll have empty lists when not using restrict_lexicon
        for model, out_w, out_b, state in itertools.zip_longest(
                self.models, models_output_layer_w, models_output_layer_b, states):
            decoder_out, attention_probs, state = model.run_decoder(prev_word, bucket_key, state)
            # Compute logits and softmax with restricted vocabulary
            if self.restrict_lexicon:
                # Apply output layer outside decoder module.
                logits = model.output_layer(decoder_out, out_w, out_b)
                if model.skip_softmax:
                    model_out = logits  # raw logits
                else:
                    model_out = mx.nd.softmax(logits)  # normalized probabilities
            else:
                # Output layer is applied inside decoder module.
                # if model.skip_softmax decoder_out represents logits, normalized probabilities else.
                model_out = decoder_out
            model_outs.append(model_out)
            model_attention_probs.append(attention_probs)
            model_states.append(state)
        scores, attention_probs = self._combine_predictions(model_outs, model_attention_probs)
        return scores, attention_probs, model_states

    def _combine_predictions(self,
                             model_outputs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models and averaged attention prob scores.
        If model_outputs are probabilities, they are converted to negative log probabilities before combination.
        If model_outputs are logits (and no ensembling is used),
        no combination is applied and logits are converted to negative logits.

        :param model_outputs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined scores, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            if self.models[0].skip_softmax:
                scores = -model_outputs[0]
            else:
                scores = -mx.nd.log(model_outputs[0])  # pylint: disable=invalid-unary-operand-type
        else:
            scores = self.interpolation_func(model_outputs)
        return scores, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     restrict_lexicon: Optional[lexicon.TopKLexicon],
                     raw_constraint_list: List[Optional[constrained.RawConstraintList]],
                     raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                     max_output_lengths: mx.nd.NDArray) -> Tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 List[Optional[np.ndarray]],
                                                                 List[Optional[constrained.ConstrainedHypothesis]],
                                                                 Optional[List[BeamHistory]]]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Max source length.
        :param restrict_lexicon: Lexicon to use for vocabulary restriction.
        :param raw_constraint_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must appear in each output.
        :param raw_avoid_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must NOT appear in each output.
        :return List of best hypotheses indices, list of best word indices, list of attentions,
                array of accumulated length-normalized negative log-probs, hypotheses lengths,
                predicted lengths of references (if any), constraints (if any), beam histories (if any).
        """
        batch_size = source.shape[0]
        logger.debug("_beam_search batch size: %d", batch_size)

        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(source_length)
        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(source_length) for model in self.models),
                              "Models must agree on encoded sequence length")
        # Maximum output length
        max_output_length = self.models[0].get_max_output_length(source_length)

        # General data structure: batch_size * beam_size blocks in total;
        # a full beam for each sentence, folloed by the next beam-block for the next sentence and so on

        best_word_indices = mx.nd.full((batch_size * self.beam_size,), val=self.start_id, ctx=self.context,
                                       dtype='int32')

        # offset for hypothesis indices in batch decoding
        offset = mx.nd.repeat(mx.nd.arange(0, batch_size * self.beam_size, self.beam_size,
                                           dtype='int32', ctx=self.context), self.beam_size)

        # locations of each batch item when first dimension is (batch * beam)
        batch_indices = mx.nd.arange(0, batch_size * self.beam_size, self.beam_size, dtype='int32', ctx=self.context)
        first_step_mask = mx.nd.full((batch_size * self.beam_size, 1), val=np.inf, ctx=self.context)
        first_step_mask[batch_indices] = 1.0
        pad_dist = mx.nd.full((batch_size * self.beam_size, len(self.vocab_target) - 1), val=np.inf,
                              ctx=self.context)

        # Best word and hypotheses indices across beam search steps from topk operation.
        best_hyp_indices_list = []  # type: List[mx.nd.NDArray]
        best_word_indices_list = []  # type: List[mx.nd.NDArray]

        # Beam history
        beam_histories = None  # type: Optional[List[BeamHistory]]
        if self.store_beam:
            beam_histories = [defaultdict(list) for _ in range(batch_size)]

        lengths = mx.nd.zeros((batch_size * self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # Extending max_output_lengths to shape (batch_size * beam_size,)
        max_output_lengths = mx.nd.repeat(max_output_lengths, self.beam_size)

        # Attention distributions across beam search steps
        attentions = []  # type: List[mx.nd.NDArray]

        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((batch_size * self.beam_size, 1), ctx=self.context)

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        models_output_layer_w = list()
        models_output_layer_b = list()
        vocab_slice_ids = None  # type: mx.nd.NDArray
        if restrict_lexicon:
            source_words = utils.split(source, num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            # TODO: See note in method about migrating to pure MXNet when set operations are supported.
            #       We currently convert source to NumPy and target ids back to NDArray.
            vocab_slice_ids = restrict_lexicon.get_trg_ids(source_words.astype("int32").asnumpy())
            if any(raw_constraint_list):
                # Add the constraint IDs to the list of permissibled IDs, and then project them into the reduced space
                constraint_ids = np.array([word_id for sent in raw_constraint_list for phr in sent for word_id in phr])
                vocab_slice_ids = np.lib.arraysetops.union1d(vocab_slice_ids, constraint_ids)
                full_to_reduced = dict((val, i) for i, val in enumerate(vocab_slice_ids))
                raw_constraint_list = [[[full_to_reduced[x] for x in phr] for phr in sent] for sent in
                                       raw_constraint_list]

            vocab_slice_ids = mx.nd.array(vocab_slice_ids, ctx=self.context, dtype='int32')

            if vocab_slice_ids.shape[0] < self.beam_size + 1:
                # This fixes an edge case for toy models, where the number of vocab ids from the lexicon is
                # smaller than the beam size.
                logger.warning("Padding vocab_slice_ids (%d) with EOS to have at least %d+1 elements to expand",
                               vocab_slice_ids.shape[0], self.beam_size)
                n = self.beam_size - vocab_slice_ids.shape[0] + 1
                vocab_slice_ids = mx.nd.concat(vocab_slice_ids,
                                               mx.nd.full((n,), val=self.vocab_target[C.EOS_SYMBOL],
                                                          ctx=self.context, dtype='int32'),
                                               dim=0)

            pad_dist = mx.nd.full((batch_size * self.beam_size, vocab_slice_ids.shape[0] - 1),
                                  val=np.inf, ctx=self.context)
            for m in self.models:
                models_output_layer_w.append(m.output_layer_w.take(vocab_slice_ids))
                models_output_layer_b.append(m.output_layer_b.take(vocab_slice_ids))

        # (0) encode source sentence, returns a list
        model_states, estimated_reference_lengths = self._encode(source, source_length)

        # Initialize the beam to track constraint sets, where target-side lexical constraints are present
        constraints = constrained.init_batch(raw_constraint_list, self.beam_size, self.start_id,
                                             self.vocab_target[C.EOS_SYMBOL])

        if self.global_avoid_trie or any(raw_avoid_list):
            avoid_states = constrained.AvoidBatch(batch_size, self.beam_size,
                                                  avoid_list=raw_avoid_list,
                                                  global_avoid_trie=self.global_avoid_trie)
            avoid_states.consume(best_word_indices)

        # Records items in the beam that are inactive. At the beginning (t==1), there is only one valid or active
        # item on the beam for each sentence
        inactive = mx.nd.zeros((batch_size * self.beam_size), dtype='int32', ctx=self.context)
        t = 1
        for t in range(1, max_output_length):
            # (1) obtain next predictions and advance models' state
            # target_dists: (batch_size * beam_size, target_vocab_size)
            # attention_scores: (batch_size * beam_size, bucket_key)
            target_dists, attention_scores, model_states = self._decode_step(prev_word=best_word_indices,
                                                                             step=t,
                                                                             source_length=source_length,
                                                                             states=model_states,
                                                                             models_output_layer_w=models_output_layer_w,
                                                                             models_output_layer_b=models_output_layer_b)

            # (2) Produces the accumulated cost of target words in each row.
            # There is special treatment for finished and inactive rows: inactive rows are inf everywhere;
            # finished rows are inf everywhere except column zero, which holds the accumulated model score
            scores = self._update_scores.forward(target_dists, finished, inactive, scores_accumulated, pad_dist)

            # Mark entries that should be blocked as having a score of np.inf
            if self.global_avoid_trie or any(raw_avoid_list):
                block_indices = avoid_states.avoid()
                if len(block_indices) > 0:
                    scores[block_indices] = np.inf
                    if self.sample is not None:
                        target_dists[block_indices] = np.inf

            # (3) Get beam_size winning hypotheses for each sentence block separately. Only look as
            # far as the active beam size for each sentence.

            if self.sample is not None:
                best_hyp_indices, best_word_indices, scores_accumulated = self._top(scores, target_dists, finished)
            else:
                # On the first timestep, all hypotheses have identical histories, so force topk() to choose extensions
                # of the first row only by setting all other rows to inf
                if t == 1 and not self.skip_topk:
                    scores *= first_step_mask

                best_hyp_indices, best_word_indices, scores_accumulated = self._top(scores, offset)

            # Constraints for constrained decoding are processed sentence by sentence
            if any(raw_constraint_list):
                best_hyp_indices, best_word_indices, scores_accumulated, constraints, inactive = constrained.topk(
                    t,
                    batch_size,
                    self.beam_size,
                    inactive,
                    scores,
                    constraints,
                    best_hyp_indices,
                    best_word_indices,
                    scores_accumulated)

            # Map from restricted to full vocab ids if needed
            if restrict_lexicon:
                best_word_indices = vocab_slice_ids.take(best_word_indices)

            # (4) Reorder fixed-size beam data according to best_hyp_indices (ascending)
            finished, lengths, attention_scores, estimated_reference_lengths \
                                                = self._sort_by_index.forward(best_hyp_indices,
                                                                              finished,
                                                                              lengths,
                                                                              attention_scores,
                                                                              estimated_reference_lengths)

            # (5) Normalize the scores of newly finished hypotheses. Note that after this until the
            # next call to topk(), hypotheses may not be in sorted order.
            finished, scores_accumulated, lengths = self._update_finished.forward(best_word_indices,
                                                                                  max_output_lengths,
                                                                                  finished,
                                                                                  scores_accumulated,
                                                                                  lengths,
                                                                                  estimated_reference_lengths)

            # (6) Prune out low-probability hypotheses. Pruning works by setting entries `inactive`.
            if self.beam_prune > 0.0:
                inactive, best_word_indices, scores_accumulated = self._prune_hyps.forward(best_word_indices,
                                                                                           scores_accumulated,
                                                                                           finished)

            # (7) update negative constraints
            if self.global_avoid_trie or any(raw_avoid_list):
                avoid_states.reorder(best_hyp_indices)
                avoid_states.consume(best_word_indices)

            # (8) optionally save beam history
            if self.store_beam:
                finished_or_inactive = mx.nd.clip(data=finished + inactive, a_min=0, a_max=1)
                unnormalized_scores = mx.nd.where(finished_or_inactive,
                                                  scores_accumulated * self.length_penalty(lengths),
                                                  scores_accumulated)
                normalized_scores = mx.nd.where(finished_or_inactive,
                                                scores_accumulated,
                                                scores_accumulated / self.length_penalty(lengths))
                for sent in range(batch_size):
                    rows = slice(sent * self.beam_size, (sent + 1) * self.beam_size)

                    best_word_indices_sent = best_word_indices[rows].asnumpy().tolist()
                    # avoid adding columns for finished sentences
                    if any(x for x in best_word_indices_sent if x != C.PAD_ID):
                        beam_histories[sent]["predicted_ids"].append(best_word_indices_sent)
                        beam_histories[sent]["predicted_tokens"].append([self.vocab_target_inv[x] for x in
                                                                         best_word_indices_sent])
                        # for later sentences in the matrix, shift from e.g. [5, 6, 7, 8, 6] to [0, 1, 3, 4, 1]
                        shifted_parents = best_hyp_indices[rows] - (sent * self.beam_size)
                        beam_histories[sent]["parent_ids"].append(shifted_parents.asnumpy().tolist())

                        beam_histories[sent]["scores"].append(unnormalized_scores[rows].asnumpy().flatten().tolist())
                        beam_histories[sent]["normalized_scores"].append(
                            normalized_scores[rows].asnumpy().flatten().tolist())

            # Collect best hypotheses, best word indices, and attention scores
            best_hyp_indices_list.append(best_hyp_indices)
            best_word_indices_list.append(best_word_indices)
            attentions.append(attention_scores)

            if self.beam_search_stop == C.BEAM_SEARCH_STOP_FIRST:
                at_least_one_finished = finished.reshape((batch_size, self.beam_size)).sum(axis=1) > 0
                if at_least_one_finished.sum().asscalar() == batch_size:
                    break
            else:
                if finished.sum().asscalar() == batch_size * self.beam_size:  # all finished
                    break

            # (9) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices)

        logger.debug("Finished after %d / %d steps.", t + 1, max_output_length)

        # (9) Sort the hypotheses within each sentence (normalization for finished hyps may have unsorted them).
        folded_accumulated_scores = scores_accumulated.reshape((batch_size,
                                                                self.beam_size * scores_accumulated.shape[-1]))
        indices = mx.nd.cast(mx.nd.argsort(folded_accumulated_scores, axis=1), dtype='int32').reshape((-1,))
        best_hyp_indices, _ = mx.nd.unravel_index(indices, scores_accumulated.shape) + offset
        best_hyp_indices_list.append(best_hyp_indices)
        lengths = lengths.take(best_hyp_indices)
        scores_accumulated = scores_accumulated.take(best_hyp_indices)
        constraints = [constraints[x] for x in best_hyp_indices.asnumpy()]

        all_best_hyp_indices = mx.nd.stack(*best_hyp_indices_list, axis=1)
        all_best_word_indices = mx.nd.stack(*best_word_indices_list, axis=1)
        all_attentions = mx.nd.stack(*attentions, axis=1)

        return all_best_hyp_indices.asnumpy(), \
               all_best_word_indices.asnumpy(), \
               all_attentions.asnumpy(), \
               scores_accumulated.asnumpy(), \
               lengths.asnumpy().astype('int32'), \
               estimated_reference_lengths.asnumpy(), \
               constraints, \
               beam_histories

    def _get_best_from_beam(self,
                            best_hyp_indices: np.ndarray,
                            best_word_indices: np.ndarray,
                            attentions: np.ndarray,
                            seq_scores: np.ndarray,
                            lengths: np.ndarray,
                            estimated_reference_lengths: Optional[mx.nd.NDArray],
                            constraints: List[Optional[constrained.ConstrainedHypothesis]],
                            beam_histories: Optional[List[BeamHistory]] = None) -> List[Translation]:
        """
        Return the nbest (aka n top) entries from the n-best list.

        :param best_hyp_indices: Array of best hypotheses indices ids. Shape: (batch * beam, num_beam_search_steps + 1).
        :param best_word_indices: Array of best hypotheses indices ids. Shape: (batch * beam, num_beam_search_steps).
        :param attentions: Array of attentions over source words.
                           Shape: (batch * beam, num_beam_search_steps, encoded_source_length).
        :param seq_scores: Array of length-normalized negative log-probs. Shape: (batch * beam, 1)
        :param lengths: The lengths of all items in the beam. Shape: (batch * beam). Dtype: int32.
        :param estimated_reference_lengths: Predicted reference lengths.
        :param constraints: The constraints for all items in the beam. Shape: (batch * beam).
        :param beam_histories: The beam histories for each sentence in the batch.
        :return: List of Translation objects containing all relevant information.
        """
        batch_size = best_hyp_indices.shape[0] // self.beam_size
        nbest_translations = []  # type: List[List[Translation]]
        histories = beam_histories if beam_histories is not None else [None] * self.batch_size  # type: List
        reference_lengths = estimated_reference_lengths if estimated_reference_lengths is not None \
                                                        else np.full(self.batch_size * self.beam_size, None)
        for n in range(0, self.nbest_size):

            # Initialize the best_ids to the first item in each batch, plus current nbest index
            best_ids = np.arange(n, batch_size * self.beam_size, self.beam_size, dtype='int32')

            # only check for constraints for 1-best translation for each sequence in batch
            if n == 0 and any(constraints):
                # For constrained decoding, select from items that have met all constraints (might not be finished)
                unmet = np.array([c.num_needed() if c is not None else 0 for c in constraints])
                filtered = np.where(unmet == 0, seq_scores.flatten(), np.inf)
                filtered = filtered.reshape((batch_size, self.beam_size))
                best_ids += np.argmin(filtered, axis=1).astype('int32')

            # Obtain sequences for all best hypotheses in the batch
            indices = self._get_best_word_indices_for_kth_hypotheses(best_ids, best_hyp_indices)
            nbest_translations.append([self._assemble_translation(*x) for x in zip(best_word_indices[indices, np.arange(indices.shape[1])],
                                                                                   lengths[best_ids],
                                                                                   attentions[best_ids],
                                                                                   seq_scores[best_ids],
                                                                                   histories,
                                                                                   reference_lengths[best_ids])])
        # reorder and regroup lists
        reduced_translations = [_reduce_nbest_translations(grouped_nbest) for grouped_nbest in zip(*nbest_translations)]
        return reduced_translations

    @staticmethod
    def _get_best_word_indices_for_kth_hypotheses(ks: np.ndarray, all_hyp_indices: np.ndarray) -> np.ndarray:
        """
        Traverses the matrix of best hypotheses indices collected during beam search in reversed order by
        using the kth hypotheses index as a backpointer.
        Returns an array containing the indices into the best_word_indices collected during beam search to extract
        the kth hypotheses.

        :param ks: The kth-best hypotheses to extract. Supports multiple for batch_size > 1. Shape: (batch,).
        :param all_hyp_indices: All best hypotheses indices list collected in beam search. Shape: (batch * beam, steps).
        :return: Array of indices into the best_word_indices collected in beam search
            that extract the kth-best hypothesis. Shape: (batch,).
        """
        batch_size = ks.shape[0]
        num_steps = all_hyp_indices.shape[1]
        result = np.zeros((batch_size, num_steps - 1), dtype=all_hyp_indices.dtype)
        # first index into the history of the desired hypotheses.
        pointer = all_hyp_indices[ks, -1]
        # for each column/step follow the pointer, starting from the penultimate column/step
        num_steps = all_hyp_indices.shape[1]
        for step in range(num_steps - 2, -1, -1):
            result[:, step] = pointer
            pointer = all_hyp_indices[pointer, step]
        return result

    @staticmethod
    def _assemble_translation(sequence: np.ndarray,
                              length: np.ndarray,
                              attention_lists: np.ndarray,
                              seq_score: np.ndarray,
                              beam_history: Optional[BeamHistory],
                              estimated_reference_length: Optional[float]) -> Translation:
        """
        Takes a set of data pertaining to a single translated item, performs slightly different
        processing on each, and merges it into a Translation object.
        :param sequence: Array of word ids. Shape: (batch_size, bucket_key).
        :param length: The length of the translated segment.
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch_size * self.beam_size, max_output_length, encoded_source_length).
        :param seq_score: Array of length-normalized negative log-probs.
        :param estimated_reference_length: Estimated reference length (if any).
        :param beam_history: The optional beam histories for each sentence in the batch.
        :return: A Translation object.
        """
        length = int(length)
        sequence = sequence[:length].tolist()
        attention_matrix = attention_lists[:length, :]
        score = float(seq_score)
        estimated_reference_length=float(estimated_reference_length) if estimated_reference_length else None
        beam_history_list = [beam_history] if beam_history is not None else []
        return Translation(sequence, attention_matrix, score, beam_history_list,
                           nbest_translations=None,
                           estimated_reference_length=estimated_reference_length)

    def _print_beam(self,
                    sequences: mx.nd.NDArray,
                    accumulated_scores: mx.nd.NDArray,
                    finished: mx.nd.NDArray,
                    inactive: mx.nd.NDArray,
                    constraints: List[Optional[constrained.ConstrainedHypothesis]],
                    timestep: int) -> None:
        """
        Prints the beam for debugging purposes.

        :param sequences: The beam histories (shape: batch_size * beam_size, max_output_len).
        :param accumulated_scores: The accumulated scores for each item in the beam.
               Shape: (batch_size * beam_size, target_vocab_size).
        :param finished: Indicates which items are finished (shape: batch_size * beam_size).
        :param inactive: Indicates any inactive items (shape: batch_size * beam_size).
        :param timestep: The current timestep.
        """
        logger.info('BEAM AT TIMESTEP %d', timestep)
        batch_beam_size = sequences.shape[0]
        for i in range(batch_beam_size):
            # for each hypothesis, print its entire history
            score = accumulated_scores[i].asscalar()
            word_ids = [int(x.asscalar()) for x in sequences[i]]
            unmet = constraints[i].num_needed() if constraints[i] is not None else -1
            hypothesis = '----------' if inactive[i] else ' '.join(
                [self.vocab_target_inv[x] for x in word_ids if x != 0])
            logger.info('%d %d %d %d %.2f %s', i + 1, finished[i].asscalar(), inactive[i].asscalar(), unmet, score,
                        hypothesis)


class PruneHypotheses(mx.gluon.HybridBlock):
    """
    A HybridBlock that returns an array of shape (batch*beam,) indicating which hypotheses are inactive due to pruning.

    :param threshold: Pruning threshold.
    :param beam_size: Beam size.
    """

    def __init__(self, threshold: float, beam_size: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.beam_size = beam_size
        with self.name_scope():
            self.inf = self.params.get_constant(name='inf', value=mx.nd.full((1, 1), val=np.inf))

    def hybrid_forward(self, F, best_word_indices, scores, finished, inf):
        # (batch*beam, 1) -> (batch, beam)
        scores_2d = F.reshape(scores, shape=(-1, self.beam_size))
        finished_2d = F.reshape(finished, shape=(-1, self.beam_size))
        inf_array_2d = F.broadcast_like(inf, scores_2d)
        inf_array = F.broadcast_like(inf, scores)

        # best finished scores. Shape: (batch, 1)
        best_finished_scores = F.min(F.where(finished_2d, scores_2d, inf_array_2d), axis=1, keepdims=True)
        difference = F.broadcast_minus(scores_2d, best_finished_scores)
        inactive = F.cast(difference > self.threshold, dtype='int32')
        inactive = F.reshape(inactive, shape=(-1))

        best_word_indices = F.where(inactive, F.zeros_like(best_word_indices), best_word_indices)
        scores = F.where(inactive, inf_array, scores)

        return inactive, best_word_indices, scores


class SortByIndex(mx.gluon.HybridBlock):
    """
    A HybridBlock that sorts args by the given indices.
    """

    def hybrid_forward(self, F, indices, *args):
        return [F.take(arg, indices) for arg in args]


class TopK(mx.gluon.HybridBlock):
    """
    A HybridBlock for a statically-shaped batch-wise topk operation.
    """

    def __init__(self, k: int, vocab_size: int) -> None:
        """
        :param k: The number of smallest scores to return.
        :param vocab_size: Vocabulary size.
        """
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size

    def hybrid_forward(self, F, scores, offset):
        """
        Get the lowest k elements per sentence from a `scores` matrix.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """
        # Shape: (batch size, beam_size * vocab_size)
        folded_scores = F.reshape(scores, shape=(-1, self.k * self.vocab_size))

        values, indices = F.topk(folded_scores, axis=1, k=self.k, ret_typ='both', is_ascend=True)

        # Project indices back into original shape (which is different for t==1 and t>1)
        indices = F.reshape(F.cast(indices, 'int32'), shape=(-1,))
        # TODO: we currently exploit a bug in the implementation of unravel_index to not require knowing the first shape
        # value. See https://github.com/apache/incubator-mxnet/issues/13862
        unraveled = F.unravel_index(indices, shape=(C.LARGEST_INT, self.vocab_size))

        best_hyp_indices, best_word_indices = F.split(unraveled, axis=0, num_outputs=2, squeeze_axis=True)
        best_hyp_indices = best_hyp_indices + offset
        values = F.reshape(values, shape=(-1, 1))
        return best_hyp_indices, best_word_indices, values


class SampleK(mx.gluon.HybridBlock):
    """
    A HybridBlock for selecting a random word from each hypothesis according to its distribution.
    """

    def __init__(self, k: int, n: int, max_batch_size: int) -> None:
        """
        :param k: The size of the beam.
        :param n: Sample from the top-N words in the vocab at each timestep.
        :param max_batch_size: Number of sentences being decoded at once.
        """
        super().__init__()
        self.n = n
        with self.name_scope():
            self.best_hyp_indices = self.params.get_constant(name='best_hyp_indices',
                                                             value=mx.nd.arange(0, max_batch_size * k, dtype='int32'))

    def hybrid_forward(self, F, scores, target_dists, finished, best_hyp_indices):
        """
        Choose an extension of each hypothesis from its softmax distribution.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param target_dists: The non-cumulative target distributions (ignored).
        :param finished: The list of finished hypotheses.
        :param best_hyp_indices: Best hypothesis indices constant.
        :return: The row indices, column indices, and values of the sampled words.
        """
        # Map the negative logprobs to probabilities so as to have a distribution
        target_dists = F.exp(-target_dists)

        # n == 0 means sample from the full vocabulary. Otherwise, we sample from the top n.
        if self.n != 0:
            # select the top n in each row, via a mask
            masked_items = F.topk(target_dists, k=self.n, ret_typ='mask', axis=1, is_ascend=False)
            # set unmasked items to 0
            masked_items = F.where(masked_items, target_dists, masked_items)
            # renormalize
            target_dists = F.broadcast_div(masked_items, F.sum(masked_items, axis=1, keepdims=True))

        # Sample from the target distributions over words, then get the corresponding values from the cumulative scores
        best_word_indices = F.random.multinomial(target_dists, get_prob=False)
        # Zeroes for finished hypotheses.
        best_word_indices = F.where(finished, F.zeros_like(best_word_indices), best_word_indices)
        values = F.pick(scores, best_word_indices, axis=1, keepdims=True)

        best_hyp_indices = F.slice_like(best_hyp_indices, best_word_indices, axes=(0,))

        return best_hyp_indices, best_word_indices, values


class Top1(mx.gluon.HybridBlock):
    """
    A HybridBlock for a statically-shaped batch-wise first-best operation.

    Get the single lowest element per sentence from a `scores` matrix. Expects that
    beam size is 1, for greedy decoding.

    NOTE(mathmu): The current implementation of argmin in MXNet much slower than topk with k=1.
    """

    def hybrid_forward(self, F, scores, offset):
        """
        Get the single lowest element per sentence from a `scores` matrix. Expects that
        beam size is 1, for greedy decoding.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
        :return: The row indices, column indices and values of the smallest items in matrix.
        """
        best_word_indices = F.cast(F.argmin(scores, axis=1), dtype='int32')
        values = F.pick(scores, best_word_indices, axis=1)
        values = F.reshape(values, shape=(-1, 1))

        # for top1, the best hyp indices are equal to the plain offset
        best_hyp_indices = offset

        return best_hyp_indices, best_word_indices, values


class NormalizeAndUpdateFinished(mx.gluon.HybridBlock):
    """
    A HybridBlock for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self, pad_id: int,
                 eos_id: int,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 brevity_penalty_weight: float = 0.0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        with self.name_scope():
            self.length_penalty = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)
            self.brevity_penalty = None  # type: Optional[BrevityPenalty]
            if brevity_penalty_weight > 0.0:
                self.brevity_penalty = BrevityPenalty(weight=brevity_penalty_weight)

    def hybrid_forward(self, F, best_word_indices, max_output_lengths,
                       finished, scores_accumulated, lengths, reference_lengths):
        all_finished = F.broadcast_logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        newly_finished = F.broadcast_logical_xor(all_finished, finished)
        if self.brevity_penalty is not None:
            brevity_penalty = self.brevity_penalty(lengths, reference_lengths)
        else:
            brevity_penalty = F.zeros_like(reference_lengths)
        scores_accumulated = F.where(newly_finished,
                                     scores_accumulated / self.length_penalty(lengths) - brevity_penalty,
                                     scores_accumulated)

        # Update lengths of all items, except those that were already finished. This updates
        # the lengths for inactive items, too, but that doesn't matter since they are ignored anyway.
        lengths = lengths + F.cast(1 - F.expand_dims(finished, axis=1), dtype='float32')

        # Now, recompute finished. Hypotheses are finished if they are
        # - extended with <pad>, or
        # - extended with <eos>, or
        # - at their maximum length.
        finished = F.broadcast_logical_or(F.broadcast_logical_or(best_word_indices == self.pad_id,
                                                                 best_word_indices == self.eos_id),
                                          (F.cast(F.reshape(lengths, shape=(-1,)), 'int32') >= max_output_lengths))

        return finished, scores_accumulated, lengths


class UpdateScores(mx.gluon.HybridBlock):
    """
    A HybridBlock that updates the scores from the decoder step with accumulated scores.
    Inactive hypotheses receive score inf. Finished hypotheses receive their accumulated score for C.PAD_ID.
    All other options are set to infinity.
    """

    def __init__(self):
        super().__init__()
        assert C.PAD_ID == 0, "This block only works with PAD_ID == 0"

    def hybrid_forward(self, F, target_dists, finished, inactive, scores_accumulated, pad_dist):
        # Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
        # finished rows are inf everywhere except column zero (pad_id), which holds the accumulated model score.
        # Items that are finished (but not inactive) get their previous accumulated score for the <pad> symbol,
        # infinity otherwise.
        scores = F.broadcast_add(target_dists, scores_accumulated)
        # pad_dist. Shape: (batch*beam, vocab_size-1)
        scores = F.where(F.broadcast_logical_or(finished, inactive), F.concat(scores_accumulated, pad_dist), scores)
        return scores

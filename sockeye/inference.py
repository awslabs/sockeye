# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import itertools
import json
import logging
import os
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, Set

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
    :param batch_size: Batch size.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 beam_size: int,
                 batch_size: int,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False,
                 forced_max_output_len: Optional[int] = None) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.beam_size = beam_size
        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')
        self.batch_size = batch_size
        self.softmax_temperature = softmax_temperature
        self.max_input_length, self.get_max_output_length = models_max_input_output_length([self],
                                                                                           max_output_length_num_stds,
                                                                                           forced_max_output_len=forced_max_output_len)

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

    def initialize(self, max_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
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

        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key)
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
            return mx.sym.Group(decoder_init_states), data_names, label_names

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
                    logits /= self.softmax_temperature
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

    def _get_encoder_data_shapes(self, bucket_key: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(self.batch_size, bucket_key, self.num_source_factors),
                               layout=C.BATCH_MAJOR)]

    @lru_cache(maxsize=None)
    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """
        source_max_length, target_max_length = bucket_key
        return [mx.io.DataDesc(name=C.TARGET_NAME, shape=(self.batch_size * self.beam_size,), layout="NT")] + \
               self.decoder.state_shapes(self.batch_size * self.beam_size,
                                         target_max_length,
                                         self.encoder.get_encoded_seq_len(source_max_length),
                                         self.encoder.get_num_hidden())

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int) -> 'ModelState':
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens. Shape (batch_size, source length, num_source_factors).
        :param source_max_length: Bucket key.
        :return: Initial model state.
        """
        batch = mx.io.DataBatch(data=[source],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_states = self.encoder_module.get_outputs()

        # replicate encoder/init module results beam size times
        decoder_states = [mx.nd.repeat(s, repeats=self.beam_size, axis=0) for s in decoder_states]
        return ModelState(decoder_states)

    def run_decoder(self,
                    prev_word: mx.nd.NDArray,
                    bucket_key: Tuple[int, int],
                    model_state: 'ModelState') -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState']:
        """
        Runs forward pass of the single-step decoder.

        :return: Decoder stack output (logit inputs or probability distribution), attention scores, updated model state.
        """
        batch = mx.io.DataBatch(
            data=[prev_word.as_in_context(self.context)] + model_state.states,
            label=None,
            bucket_key=bucket_key,
            provide_data=self._get_decoder_data_shapes(bucket_key))
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
                override_dtype: Optional[str] = None) -> Tuple[List[InferenceModel],
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
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    """
    logger.info("Loading %d model(s) from %s ...", len(model_folders), model_folders)
    load_time_start = time.time()
    models = []  # type: List[InferenceModel]
    source_vocabs = []  # type: List[List[vocab.Vocab]]
    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model_source_vocabs = vocab.load_source_vocabs(model_folder)
        model_target_vocab = vocab.load_target_vocab(model_folder)
        source_vocabs.append(model_source_vocabs)
        target_vocabs.append(model_target_vocab)

        model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", model_version)
        utils.check_version(model_version)
        model_config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
        if override_dtype is not None:
            model_config.config_encoder.dtype = override_dtype
            model_config.config_decoder.dtype = override_dtype

        if checkpoint is None:
            params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

        inference_model = InferenceModel(config=model_config,
                                         params_fname=params_fname,
                                         context=context,
                                         beam_size=beam_size,
                                         batch_size=batch_size,
                                         softmax_temperature=softmax_temperature,
                                         decoder_return_logit_inputs=decoder_return_logit_inputs,
                                         cache_output_layer_w_b=cache_output_layer_w_b)
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
        inference_model.initialize(max_input_len, get_max_output_length)

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


class TranslatorInput:
    """
    Object required by Translator.translate().

    :param sentence_id: Sentence id.
    :param tokens: List of input tokens.
    :param factors: Optional list of additional factor sequences.
    :param constraints: Optional list of target-side constraints.
    :param chunk_id: Chunk id. Defaults to -1.
    """

    __slots__ = ('sentence_id', 'tokens', 'factors', 'constraints', 'avoid_list', 'chunk_id')

    def __init__(self,
                 sentence_id: int,
                 tokens: Tokens,
                 factors: Optional[List[Tokens]] = None,
                 constraints: Optional[List[Tokens]] = None,
                 avoid_list: Optional[List[Tokens]] = None,
                 chunk_id: int = -1) -> None:
        self.sentence_id = sentence_id
        self.chunk_id = chunk_id
        self.tokens = tokens
        self.factors = factors
        self.constraints = constraints
        self.avoid_list = avoid_list

    def __str__(self):
        return 'TranslatorInput(%d, %s, factors=%s, constraints=%s, avoid=%s, chunk_id=%d)' \
            % (self.sentence_id, self.tokens, self.factors, self.constraints, self.avoid_list, self.chunk_id)

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
                'Input %d has length (%d) that exceeds max input length (%d), '
                'triggering internal splitting. Placing all target-side constraints '
                'with the first chunk, which is probably wrong.',
                self.sentence_id, len(self.tokens), chunk_size)

        for chunk_id, i in enumerate(range(0, len(self), chunk_size)):
            factors = [factor[i:i + chunk_size] for factor in self.factors] if self.factors is not None else None
            # Constrained decoding is not supported for chunked TranslatorInputs. As a fall-back, constraints are
            # assigned to the first chunk
            constraints = self.constraints if chunk_id == 0 else None
            yield TranslatorInput(sentence_id=self.sentence_id,
                                  tokens=self.tokens[i:i + chunk_size],
                                  factors=factors,
                                  constraints=constraints,
                                  avoid_list=self.avoid_list,
                                  chunk_id=chunk_id)

    def with_eos(self) -> 'TranslatorInput':
        """
        :return: A new translator input with EOS appended to the tokens and factors.
        """
        return TranslatorInput(sentence_id=self.sentence_id,
                               tokens=self.tokens + [C.EOS_SYMBOL],
                               factors=[factor + [C.EOS_SYMBOL] for factor in
                                        self.factors] if self.factors is not None else None,
                               constraints=self.constraints,
                               avoid_list=self.avoid_list,
                               chunk_id=self.chunk_id)


class BadTranslatorInput(TranslatorInput):

    def __init__(self, sentence_id, tokens):
        super().__init__(sentence_id=sentence_id, tokens=tokens, chunk_id=-1, factors=None)


def _bad_input(sentence_id: int, reason: str = '') -> BadTranslatorInput:
    logger.warning("Bad input (%d): '%s'. Will return empty output.", sentence_id, reason.strip())
    return BadTranslatorInput(sentence_id=sentence_id, tokens=[])


def make_input_from_plain_string(sentence_id: int, string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a plain string.

    :param sentence_id: An integer id.
    :param string: An input string.
    :return: A TranslatorInput.
    """
    return TranslatorInput(sentence_id, tokens=list(data_io.get_tokens(string)), factors=None)


def make_input_from_json_string(sentence_id: int, json_string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a JSON object, serialized as a string.

    :param sentence_id: An integer id.
    :param json_string: A JSON object serialized as a string that must contain a key "text", mapping to the input text,
           and optionally a key "factors" that maps to a list of strings, each of which representing a factor sequence
           for the input text.
    :return: A TranslatorInput.
    """
    try:
        jobj = json.loads(json_string, encoding=C.JSON_ENCODING)
        tokens = jobj[C.JSON_TEXT_KEY]
        tokens = list(data_io.get_tokens(tokens))
        factors = jobj.get(C.JSON_FACTORS_KEY)
        if isinstance(factors, list):
            factors = [list(data_io.get_tokens(factor)) for factor in factors]
            lengths = [len(f) for f in factors]
            if not all(l == len(tokens) for l in lengths):
                logger.error("Factors have different length than input text: %d vs. %s", len(tokens), str(lengths))
                return _bad_input(sentence_id, reason=json_string)
        else:
            factors = None

        # List of phrases that must appear in the output
        constraints = jobj.get(C.JSON_CONSTRAINTS_KEY)
        if constraints:
            constraints = [list(data_io.get_tokens(constraint)) for constraint in constraints]
        else:
            constraints = None

        # List of phrases to prevent from occuring in the output
        avoid_list = jobj.get(C.JSON_AVOID_KEY)
        if avoid_list:
            avoid_list = [list(data_io.get_tokens(phrase)) for phrase in avoid_list]
        else:
            avoid_list = None

        return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors, constraints=constraints, avoid_list=avoid_list)

    except Exception as e:
        logger.exception(e, exc_info=True) if not is_python34() else logger.error(e)  # type: ignore
        return _bad_input(sentence_id, reason=json_string)


def make_input_from_factored_string(sentence_id: int,
                                    factored_string: str,
                                    translator: 'Translator',
                                    delimiter: str = C.DEFAULT_FACTOR_DELIMITER) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a string with factor annotations on a token level, separated by delimiter.
    If translator does not require any source factors, the string is parsed as a plain token string.

    :param sentence_id: An integer id.
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


def make_input_from_multiple_strings(sentence_id: int, strings: List[str]) -> TranslatorInput:
    """
    Returns a TranslatorInput object from multiple strings, where the first element corresponds to the surface tokens
    and the remaining elements to additional factors. All strings must parse into token sequences of the same length.

    :param sentence_id: An integer id.
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

    :param id: Id of input sentence.
    :param translation: Translation string without sentence boundary tokens.
    :param tokens: List of translated tokens.
    :param attention_matrix: Attention matrix. Shape: (target_length, source_length).
    :param score: Negative log probability of generated translation.
    :param beam_histories: List of beam histories. The list will contain more than one
           history if it was split due to exceeding max_length.
    """
    __slots__ = ('id', 'translation', 'tokens', 'attention_matrix', 'score', 'beam_histories')

    def __init__(self,
                 id: int,
                 translation: str,
                 tokens: List[str],
                 attention_matrix: np.ndarray,
                 score: float,
                 beam_histories: Optional[List[BeamHistory]] = None) -> None:
        self.id = id
        self.translation = translation
        self.tokens = tokens
        self.attention_matrix = attention_matrix
        self.score = score
        self.beam_histories = beam_histories


TokenIds = List[int]


class Translation:
    __slots__ = ('target_ids', 'attention_matrix', 'score', 'beam_histories')

    def __init__(self,
                 target_ids: TokenIds,
                 attention_matrix: np.ndarray,
                 score: float,
                 beam_history: List[BeamHistory] = None) -> None:
        self.target_ids = target_ids
        self.attention_matrix = attention_matrix
        self.score = score
        self.beam_histories = beam_history if beam_history is not None else []


def empty_translation() -> Translation:
    return Translation(target_ids=[], attention_matrix=np.asarray([[0]]), score=-np.inf)


TranslatedChunk = NamedTuple('TranslatedChunk', [
    ('id', int),
    ('chunk_id', int),
    ('translation', Translation),
])
"""
Translation of a chunk of a sentence.

:param id: Id of the sentence.
:param chunk_id: Id of the chunk.
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


def _concat_translations(translations: List[Translation], start_id: int, stop_ids: Set[int],
                         length_penalty: LengthPenalty) -> Translation:
    """
    Combine translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol, attention_matrix), score and length.
    :param start_id: The EOS symbol.
    :param translations: The BOS symbols.
    :return: A concatenation if the translations with a score.
    """
    # Concatenation of all target ids without BOS and EOS
    target_ids = [start_id]
    attention_matrices = []
    beam_histories = []  # type: List[BeamHistory]
    for idx, translation in enumerate(translations):
        assert translation.target_ids[0] == start_id
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids[1:])
            attention_matrices.append(translation.attention_matrix[1:, :])
        else:
            if translation.target_ids[-1] in stop_ids:
                target_ids.extend(translation.target_ids[1:-1])
                attention_matrices.append(translation.attention_matrix[1:-1, :])
            else:
                target_ids.extend(translation.target_ids[1:])
                attention_matrices.append(translation.attention_matrix[1:, :])
        beam_histories.extend(translation.beam_histories)

    # Combine attention matrices:
    attention_shapes = [attention_matrix.shape for attention_matrix in attention_matrices]
    # Adding another row for the empty BOS alignment vector
    bos_align_shape = np.asarray([1, 0])
    attention_matrix_combined = np.zeros(np.sum(np.asarray(attention_shapes), axis=0) + bos_align_shape)

    # We start at position 1 as position 0 is for the BOS, which is kept zero
    pos_t, pos_s = 1, 0
    for attention_matrix, (len_t, len_s) in zip(attention_matrices, attention_shapes):
        attention_matrix_combined[pos_t:pos_t + len_t, pos_s:pos_s + len_s] = attention_matrix
        pos_t += len_t
        pos_s += len_s

    # Unnormalize + sum and renormalize the score:
    score = sum(translation.score * length_penalty.get(len(translation.target_ids))
                for translation in translations)
    score = score / length_penalty.get(len(target_ids))
    return Translation(target_ids, attention_matrix_combined, score, beam_histories)


class Translator:
    """
    Translator uses one or several models to translate input.
    The translator holds a reference to vocabularies to convert between word ids and text tokens for input and
    translation strings.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param beam_prune: Beam pruning difference threshold.
    :param beam_search_stop: The stopping criterium.
    :param models: List of models.
    :param source_vocabs: Source vocabularies.
    :param target_vocab: Target vocabulary.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary restriction.
    :param avoid_list: Global list of phrases to exclude from the output.
    :param store_beam: If True, store the beam search history and return it in the TranslatorOutput.
    :param strip_unknown_words: If True, removes any <unk> symbols from outputs.
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
                 restrict_lexicon: Optional[lexicon.TopKLexicon] = None,
                 avoid_list: Optional[str] = None,
                 store_beam: bool = False,
                 strip_unknown_words: bool = False) -> None:
        self.context = context
        self.length_penalty = length_penalty
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
        self.batch_size = self.models[0].batch_size
        # after models are loaded we ensured that they agree on max_input_length, max_output_length and batch size
        self.max_input_length = self.models[0].max_input_length
        if bucket_source_width > 0:
            self.buckets_source = data_io.define_buckets(self.max_input_length, step=bucket_source_width)
        else:
            self.buckets_source = [self.max_input_length]
        self.pad_dist = mx.nd.full((self.batch_size * self.beam_size, len(self.vocab_target) - 1), val=np.inf,
                                   ctx=self.context)
        # These are constants used for manipulation of the beam and scores (particularly for pruning)
        self.zeros_array = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        self.inf_array = mx.nd.full((self.batch_size * self.beam_size, 1), val=np.inf,
                                    ctx=self.context, dtype='float32')

        # offset for hypothesis indices in batch decoding
        self.offset = mx.nd.array(np.repeat(np.arange(0, self.batch_size * self.beam_size, self.beam_size), self.beam_size),
                                  dtype='int32', ctx=self.context)

        self._update_scores = UpdateScores()
        self._update_scores.initialize(ctx=self.context)
        self._update_scores.hybridize()

        # topk function used in beam search
        self._topk = partial(utils.topk,
                             k=self.beam_size,
                             batch_size=self.batch_size,
                             offset=self.offset,
                             use_mxnet_topk=self.context != mx.cpu())  # MXNet implementation is faster on GPUs

        self._sort_by_index = SortByIndex()
        self._sort_by_index.initialize(ctx=self.context)
        self._sort_by_index.hybridize()

        self._normalize_finished = NormalizeFinishedHypotheses(pad_id=C.PAD_ID,
                                                               eos_id=self.vocab_target[C.EOS_SYMBOL],
                                                               length_penalty_alpha=self.length_penalty.alpha,
                                                               length_penalty_beta=self.length_penalty.beta)
        self._normalize_finished.initialize(ctx=self.context)
        self._normalize_finished.hybridize()
        self._prune_hyps = PruneHypotheses(threshold=self.beam_prune, beam_size=self.beam_size)
        self._prune_hyps.initialize(ctx=self.context)
        self._prune_hyps.hybridize()

        self._update_lengths_and_finished = UpdateLengthsAndFinished(pad_id=C.PAD_ID,
                                                                     eos_id=self.vocab_target[C.EOS_SYMBOL])
        self._update_lengths_and_finished.initialize(ctx=self.context)
        self._update_lengths_and_finished.hybridize()

        self.global_avoid_trie = None
        if avoid_list is not None:
            self.global_avoid_trie = constrained.AvoidTrie()
            for phrase in data_io.read_content(avoid_list):
                phrase_ids = data_io.tokens2ids(phrase, self.vocab_target)
                if self.unk_id in phrase_ids:
                    logger.warning("Global avoid phrase '%s' contains an %s; this may indicate improper preprocessing.", ' '.join(phrase), C.UNK_SYMBOL)
                self.global_avoid_trie.add_phrase(phrase_ids)

        logger.info("Translator (%d model(s) beam_size=%d beam_prune=%s beam_search_stop=%s "
                    "ensemble_mode=%s batch_size=%d buckets_source=%s avoiding=%d)",
                    len(self.models),
                    self.beam_size,
                    'off' if not self.beam_prune else "%.2f" % self.beam_prune,
                    self.beam_search_stop,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.batch_size,
                    self.buckets_source,
                    0 if self.global_avoid_trie is None else len(self.global_avoid_trie))

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
        log_probs = utils.average_arrays([mx.nd.log(p) for p in predictions])
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(mx.nd.softmax(log_probs))

    def translate(self, trans_inputs: List[TranslatorInput]) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Splits oversized sentences to sentence chunks of size less than max_input_length.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :return: List of translation results.
        """
        translated_chunks = []  # type: List[TranslatedChunk]

        # split into chunks
        input_chunks = []  # type: List[TranslatorInput]
        for input_idx, trans_input in enumerate(trans_inputs, 1):
            # bad input
            if isinstance(trans_input, BadTranslatorInput):
                translated_chunks.append(TranslatedChunk(id=input_idx, chunk_id=0, translation=empty_translation()))

            # empty input
            elif len(trans_input.tokens) == 0:
                translated_chunks.append(TranslatedChunk(id=input_idx, chunk_id=0, translation=empty_translation()))
            else:
                # TODO(tdomhan): Remove branch without EOS with next major version bump, as future models will always be trained with source side EOS symbols
                if self.source_with_eos:
                    max_input_length_without_eos = self.max_input_length - C.SPACE_FOR_XOS
                    # oversized input
                    if len(trans_input.tokens) > max_input_length_without_eos:
                        logger.debug(
                            "Input %d has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            self.buckets_source[-1], max_input_length_without_eos)
                        input_chunks.extend([trans_input_chunk.with_eos()
                                             for trans_input_chunk in
                                             trans_input.chunks(max_input_length_without_eos)])
                    # regular input
                    else:
                        input_chunks.append(trans_input.with_eos())
                else:
                    # oversized input
                    if len(trans_input.tokens) > self.max_input_length:
                        logger.debug(
                            "Input %d has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            self.buckets_source[-1], self.max_input_length)
                        input_chunks.extend([trans_input_chunk
                                             for trans_input_chunk in
                                             trans_input.chunks(self.max_input_length)])
                    # regular input
                    else:
                        input_chunks.append(trans_input)

            if trans_input.constraints is not None:
                logger.info("Input %d has %d %s: %s", trans_input.sentence_id,
                            len(trans_input.constraints),
                            "constraint" if len(trans_input.constraints) == 1 else "constraints",
                            ", ".join(" ".join(x) for x in trans_input.constraints))

        # Sort longest to shortest (to rather fill batches of shorter than longer sequences)
        input_chunks = sorted(input_chunks, key=lambda chunk: len(chunk.tokens), reverse=True)

        # translate in batch-sized blocks over input chunks
        for batch_id, batch in enumerate(utils.grouper(input_chunks, self.batch_size)):
            logger.debug("Translating batch %d", batch_id)
            # underfilled batch will be filled to a full batch size with copies of the 1st input
            rest = self.batch_size - len(batch)
            if rest > 0:
                logger.debug("Extending the last batch to the full batch size (%d)", self.batch_size)
                batch = batch + [batch[0]] * rest
            batch_translations = self._translate_nd(*self._get_inference_input(batch))
            # truncate to remove filler translations
            if rest > 0:
                batch_translations = batch_translations[:-rest]
            for chunk, translation in zip(batch, batch_translations):
                translated_chunks.append(TranslatedChunk(chunk.sentence_id, chunk.chunk_id, translation))
        # Sort by input idx and then chunk id
        translated_chunks = sorted(translated_chunks)

        # Concatenate results
        results = []  # type: List[TranslatorOutput]
        chunks_by_input_idx = itertools.groupby(translated_chunks, key=lambda translation: translation.id)
        for trans_input, (input_idx, chunks) in zip(trans_inputs, chunks_by_input_idx):
            chunks = list(chunks)  # type: ignore
            if len(chunks) == 1:  # type: ignore
                translation = chunks[0].translation  # type: ignore
            else:
                translations_to_concat = [translated_chunk.translation for translated_chunk in chunks]
                translation = self._concat_translations(translations_to_concat)

            results.append(self._make_result(trans_input, translation))

        return results

    def _get_inference_input(self,
                             trans_inputs: List[TranslatorInput]) -> Tuple[mx.nd.NDArray,
                                                                           int,
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           mx.nd.NDArray]:
        """
        Assembles the numerical data for the batch.
        This comprises an NDArray for the source sentences, the bucket key (padded source length), and a list of
        raw constraint lists, one for each sentence in the batch, an NDArray of maximum output lengths for each sentence in the batch.
        Each raw constraint list contains phrases in the form of lists of integers in the target language vocabulary.

        :param trans_inputs: List of TranslatorInputs.
        :return NDArray of source ids (shape=(batch_size, bucket_key, num_factors)),
                bucket key, list of raw constraint lists, and list of phrases to avoid,
                and an NDArray of maximum output lengths.
        """

        bucket_key = data_io.get_bucket(max(len(inp.tokens) for inp in trans_inputs), self.buckets_source)
        source = mx.nd.zeros((len(trans_inputs), bucket_key, self.num_source_factors), ctx=self.context)
        raw_constraints = [None for x in range(self.batch_size)]  # type: List[Optional[constrained.RawConstraintList]]
        raw_avoid_list = [None for x in range(self.batch_size)]  # type: List[Optional[constrained.RawConstraintList]]

        max_output_lengths = []  # type: List[int]
        for j, trans_input in enumerate(trans_inputs):
            num_tokens = len(trans_input)
            max_output_lengths.append(self.models[0].get_max_output_length(num_tokens))
            source[j, :num_tokens, 0] = data_io.tokens2ids(trans_input.tokens, self.source_vocabs[0])

            factors = trans_input.factors if trans_input.factors is not None else []
            num_factors = 1 + len(factors)
            if num_factors != self.num_source_factors:
                logger.warning("Input %d factors, but model(s) expect %d", num_factors,
                               self.num_source_factors)
            for i, factor in enumerate(factors[:self.num_source_factors - 1], start=1):
                # fill in as many factors as there are tokens

                source[j, :num_tokens, i] = data_io.tokens2ids(factor, self.source_vocabs[i])[:num_tokens]

            if trans_input.constraints is not None:
                raw_constraints[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                      trans_input.constraints]

            if trans_input.avoid_list is not None:
                raw_avoid_list[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                     trans_input.avoid_list]
                if any([self.unk_id in phrase for phrase in raw_avoid_list[j]]):
                    logger.warning("Sentence %d: %s was found in the list of phrases to avoid; this may indicate improper preprocessing.", trans_input.sentence_id, C.UNK_SYMBOL)

        return source, bucket_key, raw_constraints, raw_avoid_list, mx.nd.array(max_output_lengths, ctx=self.context, dtype='int32')

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrix, and score.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation + attention and score.
        :return: TranslatorOutput.
        """
        # remove special sentence start symbol (<s>) from the output:
        target_ids = translation.target_ids[1:]
        attention_matrix = translation.attention_matrix[1:, :]

        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]

        target_string = C.TOKEN_SEPARATOR.join(
            tok for target_id, tok in zip(target_ids, target_tokens) if target_id not in self.strip_ids)
        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]

        return TranslatorOutput(id=trans_input.sentence_id,
                                translation=target_string,
                                tokens=target_tokens,
                                attention_matrix=attention_matrix,
                                score=translation.score,
                                beam_histories=translation.beam_histories)

    def _concat_translations(self, translations: List[Translation]) -> Translation:
        """
        Combine translations through concatenation.

        :param translations: A list of translations (sequence, attention_matrix), score and length.
        :return: A concatenation if the translations with a score.
        """
        return _concat_translations(translations, self.start_id, self.stop_ids, self.length_penalty)

    def _translate_nd(self,
                      source: mx.nd.NDArray,
                      source_length: int,
                      raw_constraints: List[Optional[constrained.RawConstraintList]],
                      raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                      max_output_lengths: mx.nd.NDArray) -> List[Translation]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :param raw_constraints: A list of optional constraint lists.

        :return: Sequence of translations.
        """

        return self._get_best_from_beam(*self._beam_search(source, source_length, raw_constraints, raw_avoid_list, max_output_lengths))

    def _encode(self, sources: mx.nd.NDArray, source_length: int) -> List[ModelState]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param sources: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :return: List of ModelStates.
        """
        return [model.run_encoder(sources, source_length) for model in self.models]

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
        :return: (probs, attention scores, list of model states)
        """
        bucket_key = (source_length, step)

        model_probs, model_attention_probs, model_states = [], [], []
        # We use zip_longest here since we'll have empty lists when not using restrict_lexicon
        for model, out_w, out_b, state in itertools.zip_longest(
                self.models, models_output_layer_w, models_output_layer_b, states):
            decoder_outputs, attention_probs, state = model.run_decoder(prev_word, bucket_key, state)
            # Compute logits and softmax with restricted vocabulary
            if self.restrict_lexicon:
                logits = model.output_layer(decoder_outputs, out_w, out_b)
                probs = mx.nd.softmax(logits)
            else:
                # Otherwise decoder outputs are already target vocab probs
                probs = decoder_outputs
            model_probs.append(probs)
            model_attention_probs.append(attention_probs)
            model_states.append(state)
        neg_logprobs, attention_probs = self._combine_predictions(model_probs, model_attention_probs)
        return neg_logprobs, attention_probs, model_states

    def _combine_predictions(self,
                             probs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined negative log probabilities, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            neg_logprobs = -mx.nd.log(probs[0])  # pylint: disable=invalid-unary-operand-type
        else:
            neg_logprobs = self.interpolation_func(probs)
        return neg_logprobs, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     raw_constraint_list: List[Optional[constrained.RawConstraintList]],
                     raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                     max_output_lengths: mx.nd.NDArray) -> Tuple[mx.nd.NDArray,
                                                                 mx.nd.NDArray,
                                                                 mx.nd.NDArray,
                                                                 mx.nd.NDArray,
                                                                 List[Optional[constrained.ConstrainedHypothesis]],
                                                                 Optional[List[BeamHistory]]]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Max source length.
        :param raw_constraint_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must appear in each output.
        :param raw_avoid_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must NOT appear in each output.
        :return List of lists of word ids, list of attentions, array of accumulated length-normalized
                negative log-probs.
        """

        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(source_length)
        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(source_length) for model in self.models),
                              "Models must agree on encoded sequence length")
        # Maximum output length
        max_output_length = self.models[0].get_max_output_length(source_length)

        # General data structure: each row has batch_size * beam blocks for the 1st sentence, with a full beam,
        # then the next block for the 2nd sentence and so on

        best_word_indices = mx.nd.full((self.batch_size * self.beam_size,), val=self.start_id, ctx=self.context,
                                       dtype='int32')
        # growing sequences: (batch_size * beam_size, 1), pre-filled with <s> symbols on index 0
        sequences = mx.nd.full((self.batch_size * self.beam_size, 1), val=self.start_id, ctx=self.context,
                               dtype='int32')

        # Beam history
        beam_histories = None  # type: Optional[List[BeamHistory]]
        if self.store_beam:
            beam_histories = [defaultdict(list) for _ in range(self.batch_size)]

        lengths = mx.nd.ones((self.batch_size * self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # Extending max_output_lengths to shape (batch_size * beam_size,)
        max_output_lengths = mx.nd.repeat(max_output_lengths, self.beam_size)

        # attentions: (batch_size * beam_size, 1, encoded_source_length)
        attentions = mx.nd.zeros((self.batch_size * self.beam_size, 1, encoded_source_length), ctx=self.context)

        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((self.batch_size * self.beam_size, 1), ctx=self.context)

        # reset all padding distribution cells to np.inf
        self.pad_dist[:] = np.inf

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        models_output_layer_w = list()
        models_output_layer_b = list()
        pad_dist = self.pad_dist
        vocab_slice_ids = None  # type: mx.nd.NDArray
        if self.restrict_lexicon:
            source_words = utils.split(source, num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            # TODO: See note in method about migrating to pure MXNet when set operations are supported.
            #       We currently convert source to NumPy and target ids back to NDArray.
            vocab_slice_ids = self.restrict_lexicon.get_trg_ids(source_words.astype("int32").asnumpy())
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

            pad_dist = mx.nd.full((self.batch_size * self.beam_size, vocab_slice_ids.shape[0] - 1),
                                  val=np.inf, ctx=self.context)
            for m in self.models:
                models_output_layer_w.append(m.output_layer_w.take(vocab_slice_ids))
                models_output_layer_b.append(m.output_layer_b.take(vocab_slice_ids))

        # (0) encode source sentence, returns a list
        model_states = self._encode(source, source_length)

        # Initialize the beam to track constraint sets, where target-side lexical constraints are present
        constraints = constrained.init_batch(raw_constraint_list, self.beam_size, self.start_id,
                                             self.vocab_target[C.EOS_SYMBOL])

        if self.global_avoid_trie or any(raw_avoid_list):
            avoid_states = constrained.AvoidBatch(self.batch_size, self.beam_size,
                                                  avoid_list=raw_avoid_list,
                                                  global_avoid_trie=self.global_avoid_trie)

        # Records items in the beam that are inactive. At the beginning (t==1), there is only one valid or active
        # item on the beam for each sentence
        inactive = mx.nd.ones((self.batch_size * self.beam_size), dtype='int32', ctx=self.context)
        inactive[::self.beam_size] = 0
        t = 1
        for t in range(1, max_output_length):
            # (1) obtain next predictions and advance models' state
            # scores: (batch_size * beam_size, target_vocab_size)
            # attention_scores: (batch_size * beam_size, bucket_key)
            scores, attention_scores, model_states = self._decode_step(prev_word=best_word_indices,
                                                                       step=t,
                                                                       source_length=source_length,
                                                                       states=model_states,
                                                                       models_output_layer_w=models_output_layer_w,
                                                                       models_output_layer_b=models_output_layer_b)

            # (2) Update scores. Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
            # finished rows are inf everywhere except column zero, which holds the accumulated model score
            scores = self._update_scores.forward(scores, finished, inactive, scores_accumulated, self.inf_array, pad_dist)

            # Mark entries that should be blocked as having a score of np.inf
            if self.global_avoid_trie or any(raw_avoid_list):
                block_indices = avoid_states.avoid()
                if any(block_indices):
                    scores[block_indices] = np.inf

            # (3) Get beam_size winning hypotheses for each sentence block separately. Only look as
            # far as the active beam size for each sentence.
            best_hyp_indices, best_word_indices, scores_accumulated = self._topk(scores)

            # Constraints for constrained decoding are processed sentence by sentence
            if any(raw_constraint_list):
                best_hyp_indices, best_word_indices, scores_accumulated, \
                constraints, inactive = constrained.topk(self.batch_size,
                                                         self.beam_size,
                                                         inactive,
                                                         scores,
                                                         constraints,
                                                         best_hyp_indices,
                                                         best_word_indices,
                                                         scores_accumulated,
                                                         self.context)

            else:
                # All rows are now active (after special treatment of start state at t=1)
                inactive[:] = 0

            # Map from restricted to full vocab ids if needed
            if self.restrict_lexicon:
                best_word_indices = vocab_slice_ids.take(best_word_indices)

            # (4) Reorder fixed-size beam data according to best_hyp_indices (ascending)
            finished, lengths, attention_scores = self._sort_by_index.forward(best_hyp_indices,
                                                                              finished,
                                                                              lengths,
                                                                              attention_scores)

            # (5) Normalize the scores of newly finished hypotheses. Note that after this until the
            # next call to topk(), hypotheses may not be in sorted order.
            finished, scores_accumulated = self._normalize_finished.forward(best_word_indices,
                                                                            finished,
                                                                            scores_accumulated,
                                                                            lengths)

            # (6) Prune out low-probability hypotheses. Pruning works by setting entries `inactive`.
            if self.beam_prune > 0.0:
                inactive, best_word_indices, scores_accumulated = self._prune_hyps.forward(best_word_indices,
                                                                                           scores_accumulated,
                                                                                           finished,
                                                                                           self.inf_array,
                                                                                           self.zeros_array)

            # Update hypotheses lengths (unless finished or inactive) and compute finished hypotheses
            finished, finished_or_inactive, lengths = self._update_lengths_and_finished.forward(finished,
                                                                                                inactive,
                                                                                                lengths,
                                                                                                max_output_lengths,
                                                                                                best_word_indices)

            # (7) Reorder sequences and attentions according to best_hyp_indices
            # and extend sequences with best_word_indices.
            # pylint: disable=unsupported-assignment-operation
            sequences = mx.nd.take(sequences, best_hyp_indices)
            attentions = mx.nd.take(attentions, best_hyp_indices)
            sequences = mx.nd.concat(sequences, best_word_indices.reshape((-1, 1)), dim=1)
            attentions = mx.nd.concat(attentions, attention_scores.reshape((-1, 1, encoded_source_length)), dim=1)

            # (7a) update negative constraints
            if self.global_avoid_trie or any(raw_avoid_list):
                avoid_states.reorder(best_hyp_indices)
                avoid_states.consume(best_word_indices)

            # (7b) optionally save beam history
            if self.store_beam:
                unnormalized_scores = mx.nd.where(finished_or_inactive,
                                                  scores_accumulated * self.length_penalty(lengths - 1),
                                                  scores_accumulated)
                normalized_scores = mx.nd.where(finished_or_inactive,
                                                scores_accumulated,
                                                scores_accumulated / self.length_penalty(lengths - 1))
                for sent in range(self.batch_size):
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

            if self.beam_search_stop == C.BEAM_SEARCH_STOP_FIRST and self.batch_size == 1:
                # TODO: extend to work with batch_size > 1 (i.e., one stopped for each sentence)
                if mx.nd.sum(finished).asscalar() > 0:
                    break
            else:
                if mx.nd.sum(finished).asscalar() == self.batch_size * self.beam_size:  # all finished
                    break

            # (8) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices)

        logger.debug("Finished after %d / %d steps.", t + 1, max_output_length)

        # (9) Sort the hypotheses within each sentence (normalization for finished hyps may have unsorted them).
        folded_accumulated_scores = scores_accumulated.reshape((self.batch_size,
                                                                self.beam_size * scores_accumulated.shape[-1]))
        indices = mx.nd.argsort(folded_accumulated_scores, axis=1)
        best_hyp_indices = mx.nd.array(np.unravel_index(indices.astype(np.int32).asnumpy().ravel(),
                                                        scores_accumulated.shape),
                                       dtype='int32',
                                       ctx=self.offset.context)[0] + self.offset
        sequences = sequences.take(best_hyp_indices)
        lengths = lengths.take(best_hyp_indices)
        attentions = attentions.take(best_hyp_indices)
        scores_accumulated = scores_accumulated.take(best_hyp_indices)
        constraints = [constraints[x] for x in best_hyp_indices.asnumpy()]

        return sequences, attentions, scores_accumulated, lengths, constraints, beam_histories

    def _get_best_from_beam(self,
                            sequences: mx.nd.NDArray,
                            attention_lists: mx.nd.NDArray,
                            seq_scores: mx.nd.NDArray,
                            lengths: mx.nd.NDArray,
                            constraints: List[Optional[constrained.ConstrainedHypothesis]],
                            beam_histories: Optional[List[BeamHistory]] = None) -> List[Translation]:
        """
        Return the best (aka top) entry from the n-best list.

        :param sequences: Array of word ids. Shape: (batch * beam, bucket_key).
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch * beam, max_output_length, encoded_source_length).
        :param seq_scores: Array of length-normalized negative log-probs. Shape: (batch * beam, 1)
        :param lengths: The lengths of all items in the beam. Shape: (batch * beam).
        :param constraints: The constraints for all items in the beam. Shape: (batch * beam).
        :param beam_histories: The beam histories for each sentence in the batch.
        :return: List of Translation objects containing all relevant information.
        """
        utils.check_condition(sequences.shape[0] == attention_lists.shape[0] == seq_scores.shape[0] == lengths.shape[0],
                              "Shape mismatch")

        # Initialize the best_ids to the first item in each batch
        best_ids = mx.nd.arange(0, self.batch_size * self.beam_size, self.beam_size, ctx=self.context, dtype='int32')

        if any(constraints):
            # For constrained decoding, select from items that have met all constraints (might not be finished)
            unmet = mx.nd.array([c.num_needed() if c is not None else 0 for c in constraints], ctx=self.context)
            filtered = mx.nd.where(unmet == 0, seq_scores, self.inf_array)
            filtered = filtered.reshape((self.batch_size, self.beam_size))
            best_ids += mx.nd.cast(mx.nd.argmin(filtered, axis=1), dtype='int32')

        histories = beam_histories if beam_histories is not None else [None] * self.batch_size  # type: List
        return [self._assemble_translation(*x) for x in zip(sequences[best_ids],
                                                            lengths[best_ids],
                                                            attention_lists[best_ids],
                                                            seq_scores[best_ids],
                                                            histories)]

    @staticmethod
    def _assemble_translation(sequence: mx.nd.NDArray,
                              length: mx.nd.NDArray,
                              attention_lists: mx.nd.NDArray,
                              seq_score: mx.nd.NDArray,
                              beam_history: Optional[BeamHistory]) -> Translation:
        """
        Takes a set of data pertaining to a single translated item, performs slightly different
        processing on each, and merges it into a Translation object.

        :param sequence: Array of word ids. Shape: (batch_size, bucket_key).
        :param length: The length of the translated segment.
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch_size * self.beam_size, max_output_length, encoded_source_length).
        :param seq_score: Array of length-normalized negative log-probs.
        :param beam_history: The optional beam histories for each sentence in the batch.
        :return: A Translation object.
        """

        length = int(length.asscalar())
        sequence = sequence[:length].asnumpy().tolist()
        attention_matrix = np.stack(attention_lists.asnumpy()[:length, :], axis=0)
        score = seq_score.asscalar()
        beam_history_list = [beam_history] if beam_history is not None else []
        return Translation(sequence, attention_matrix, score, beam_history_list)

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
        for i in range(self.batch_size * self.beam_size):
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

    def hybrid_forward(self, F, best_word_indices, scores, finished, inf_array, zeros_array):
        scores_2d = F.reshape(scores, shape=(-1, self.beam_size))
        finished_2d = F.reshape(finished, shape=(-1, self.beam_size))
        inf_array_2d = F.reshape(inf_array, shape=(-1, self.beam_size))

        # best finished scores. Shape: (batch, 1)
        best_finished_scores = F.min(F.where(finished_2d, scores_2d, inf_array_2d), axis=1, keepdims=True)
        difference = F.broadcast_minus(scores_2d, best_finished_scores)
        inactive = F.cast(difference > self.threshold, dtype='int32')
        inactive = F.reshape(inactive, shape=(-1))

        best_word_indices = F.where(inactive, zeros_array, best_word_indices)
        scores = F.where(inactive, inf_array, scores)

        return inactive, best_word_indices, scores


class SortByIndex(mx.gluon.HybridBlock):
    """
    A HybridBlock that sorts args by the given indices.
    """

    def hybrid_forward(self, F, indices, *args):
        return [F.take(arg, indices) for arg in args]


class NormalizeFinishedHypotheses(mx.gluon.HybridBlock):
    """
    A HybridBlock for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self, pad_id: int,
                 eos_id: int,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        with self.name_scope():
            self.length_penalty = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)

    def hybrid_forward(self, F, best_word_indices, finished, scores_accumulated, lengths):
        all_finished = ((best_word_indices == self.pad_id) + (best_word_indices == self.eos_id))
        newly_finished = all_finished - finished
        scores_accumulated = F.where(newly_finished,
                                     scores_accumulated / self.length_penalty(lengths),
                                     scores_accumulated)
        finished = all_finished
        return finished, scores_accumulated


class UpdateLengthsAndFinished(mx.gluon.HybridBlock):
    """
    A HybridBlock that updates the lengths of unfinished and active hypotheses and recomputes finished hypotheses.
    """

    def __init__(self, pad_id: int, eos_id: int) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id

    def hybrid_forward(self, F, finished, inactive, lengths, max_output_lengths, best_word_indices):
        finished_or_inactive = F.clip(data=finished + inactive, a_min=0, a_max=1)
        # update lengths of hypotheses unless they are finished or inactive
        lengths = lengths + F.cast(1 - F.expand_dims(finished_or_inactive, axis=1), dtype='float32')
        # recompute finished. Hypotheses are finished if they are
        # - extended with <pad>, or
        # - extended with <eos>, or
        # - at their maximum length.
        finished = F.clip(
            (best_word_indices == self.pad_id) +
            (best_word_indices == self.eos_id) +
            (F.cast(F.reshape(lengths, shape=(-1,)), 'int32') >= max_output_lengths),
            a_min=0, a_max=1)

        return finished, finished_or_inactive, lengths


class UpdateScores(mx.gluon.HybridBlock):
    """
    A HybridBlock that updates the scores from the decoder step with acumulated scores.
    Inactive hypotheses receive score inf. Finished hypotheses receive their accumulated score for C.PAD_ID.
    All other options are set to infinity.
    """

    def __init__(self):
        super().__init__()
        assert C.PAD_ID == 0, "This blocks only works with PAD_ID == 0"

    def hybrid_forward(self, F, scores, finished, inactive, scores_accumulated, inf_array, pad_dist):
        # Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
        # finished rows are inf everywhere except column zero (pad_id), which holds the accumulated model score.
        # Items that are finished (but not inactive) get their previous accumulated score for the <pad> symbol,
        # infinity otherwise.
        scores = F.broadcast_add(scores, scores_accumulated)
        # pylint: disable=invalid-sequence-index
        pad_id_scores = F.where(F.clip(finished - inactive, 0, 1),
                                scores_accumulated,
                                inf_array)
        # pad_dist. Shape: (batch*beam, vocab_size)
        pad_dist = F.concat(pad_id_scores, pad_dist)
        scores = F.where(finished + inactive, pad_dist, scores)
        return scores

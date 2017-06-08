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
Code for inference/translation
"""
import logging
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import mxnet as mx
import numpy as np

import sockeye.bleu
import sockeye.constants as C
import sockeye.data_io
import sockeye.model
import sockeye.utils
import sockeye.vocab
from sockeye.attention import AttentionState
from sockeye.decoder import DecoderState

logger = logging.getLogger(__name__)


class InferenceModel(sockeye.model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

        (1) Encoder forward call: encode source sentence and return initial decoder states, given a bucket_key.
        (2) Decoder forward call: single decoder step: predict next word.
        (3) Return decoder data shapes, given a bucket key.

    :param model_folder: Folder to load model from.
    :param context: MXNet context to bind modules to.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param checkpoint: Checkpoint to load. If None, finds best parameters in model_folder.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    """

    def __init__(self,
                 model_folder: str,
                 context: mx.context.Context,
                 fused: bool,
                 max_input_len: Optional[int],
                 beam_size: int,
                 checkpoint: Optional[int] = None,
                 softmax_temperature: Optional[float] = None):
        # load config & determine parameter file
        super().__init__(sockeye.model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME)))
        fname_params = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)

        if max_input_len is None:
            max_input_len = self.config.max_seq_len
        else:
            if max_input_len != self.config.max_seq_len:
                logger.warning("Model was trained with max_seq_len=%d, but using max_input_len=%d.",
                               self.config.max_seq_len, max_input_len)
        self.max_input_len = max_input_len

        assert beam_size < self.config.vocab_target_size, 'beam size must be smaller than target vocab size'

        self.beam_size = beam_size
        self.softmax_temperature = softmax_temperature
        self.encoder_batch_size = 1
        self.context = context

        self._build_model_components(self.max_input_len, fused)
        self.encoder_module, self.decoder_module = self._build_modules()

        self.decoder_data_shapes_cache = dict()  # bucket_key -> shape cache
        max_encoder_data_shapes = self._get_encoder_data_shapes(self.max_input_len)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.max_input_len)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(fname_params)
        self.encoder_module.init_params(arg_params=self.params, allow_missing=False)
        self.decoder_module.init_params(arg_params=self.params, allow_missing=False)

    def _build_modules(self):

        # Encoder symbol & module
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)

        def encoder_sym_gen(source_seq_len: int):
            source_encoded = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

            # initial decoder states
            decoder_hidden_init, decoder_init_states = self.decoder.compute_init_states(source_encoded,
                                                                                        source_length)
            # initial attention state
            attention_state = self.attention.get_initial_state(source_length, source_seq_len)

            data_names = [C.SOURCE_NAME, C.SOURCE_LENGTH_NAME]
            label_names = []

            symbol_group = [source_encoded_batch_major,
                            attention_state.dynamic_source,
                            decoder_hidden_init] + decoder_init_states
            return mx.sym.Group(symbol_group), data_names, label_names

        encoder_module = mx.mod.BucketingModule(sym_gen=encoder_sym_gen,
                                                default_bucket_key=self.max_input_len,
                                                context=self.context)

        # Decoder symbol & module
        source_encoded = mx.sym.Variable(C.SOURCE_ENCODED_NAME)
        dynamic_source_prev = mx.sym.Variable(C.SOURCE_DYNAMIC_PREVIOUS_NAME)
        word_id_prev = mx.sym.Variable(C.TARGET_PREVIOUS_NAME)
        hidden_prev = mx.sym.Variable(C.HIDDEN_PREVIOUS_NAME)
        layer_states, self.layer_shapes, layer_names = self.decoder.create_layer_input_variables(self.beam_size)
        state = DecoderState(hidden_prev, layer_states)
        attention_state = AttentionState(context=None, probs=None, dynamic_source=dynamic_source_prev)

        def decoder_sym_gen(source_seq_len: int):
            data_names = [C.SOURCE_ENCODED_NAME,
                          C.SOURCE_DYNAMIC_PREVIOUS_NAME,
                          C.SOURCE_LENGTH_NAME,
                          C.TARGET_PREVIOUS_NAME,
                          C.HIDDEN_PREVIOUS_NAME] + layer_names
            label_names = []

            attention_func = self.attention.on(source_encoded, source_length, source_seq_len)

            softmax_out, next_state, next_attention_state = \
                self.decoder.predict(word_id_prev,
                                     state,
                                     attention_func,
                                     attention_state,
                                     softmax_temperature=self.softmax_temperature)

            symbol_group = [softmax_out,
                            next_attention_state.probs,
                            next_attention_state.dynamic_source,
                            next_state.hidden] + next_state.layer_states
            return mx.sym.Group(symbol_group), data_names, label_names

        decoder_module = mx.mod.BucketingModule(sym_gen=decoder_sym_gen,
                                                default_bucket_key=self.max_input_len,
                                                context=self.context)

        return encoder_module, decoder_module

    @staticmethod
    def _get_encoder_data_shapes(max_input_length: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.
        Encoder batch size is always 1.
        
        Shapes:
        source: (1, max_input_len)
        length: (1,)

        :param max_input_length: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, max_input_length), layout=C.BATCH_MAJOR),
                mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR)]

    def _get_decoder_data_shapes(self, input_length) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module, given a bucket_key (source input length)
        Caches results for bucket_keys if called iteratively.
        
        Shapes:
        source_encoded: (beam_size, input_length, encoder_num_hidden)
        source_length: (beam_size,)
        prev_target_id: (beam_size,)
        prev_hidden: (beam_size, decoder_num_hidden)

        :param input_length: Input length.
        :return: List of data descriptions.
        """
        if input_length in self.decoder_data_shapes_cache:
            return self.decoder_data_shapes_cache[input_length]

        shapes = self._get_decoder_variable_shapes(input_length) + self.layer_shapes
        self.decoder_data_shapes_cache[input_length] = shapes
        return shapes

    def _get_decoder_variable_shapes(self, input_length):
        """
        Returns only the data shapes of input variables. Auxiliary method to adjust the computation graph to the 
        presence or absence of coverage vectors.
        
        :param input_length: The maximal source sentence length 
        :return: A list of input shapes
        """
        shapes = [mx.io.DataDesc(C.SOURCE_ENCODED_NAME,
                                 (self.beam_size, input_length, self.encoder.get_num_hidden()),
                                 layout=C.BATCH_MAJOR),
                  mx.io.DataDesc(C.SOURCE_DYNAMIC_PREVIOUS_NAME,
                                 (self.beam_size, input_length, self.attention.dynamic_source_num_hidden),
                                 layout=C.BATCH_MAJOR),
                  mx.io.DataDesc(C.SOURCE_LENGTH_NAME,
                                 (self.beam_size,),
                                 layout="N"),
                  mx.io.DataDesc(C.TARGET_PREVIOUS_NAME,
                                 (self.beam_size,),
                                 layout="N"),
                  mx.io.DataDesc(C.HIDDEN_PREVIOUS_NAME,
                                 (self.beam_size, self.decoder.get_num_hidden()),
                                 layout="NC")]
        return shapes

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_length: mx.nd.NDArray,
                    bucket_key: int) -> Tuple[mx.nd.NDArray, mx.nd.NDArray,
                                              mx.nd.NDArray, mx.nd.NDArray,
                                              List[mx.nd.NDArray]]:
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens.
        :param source_length: Length of input sentence.
        :param bucket_key: Bucket key.
        :return: Encoded source, source length, initial decoder hidden state, initial decoder hidden states.
        """
        batch = mx.io.DataBatch(data=[source, source_length], label=None,
                                bucket_key=bucket_key,
                                provide_data=[
                                    mx.io.DataDesc(name=C.SOURCE_NAME, shape=(self.encoder_batch_size, bucket_key),
                                                   layout=C.BATCH_MAJOR),
                                    mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(self.encoder_batch_size,),
                                                   layout=C.BATCH_MAJOR)])

        self.encoder_module.forward(data_batch=batch, is_train=False)
        encoded_source, source_dynamic_init, decoder_hidden_init, *decoder_states = self.encoder_module.get_outputs()
        # replicate encoder/init module results beam size times
        encoded_source = mx.nd.tile(encoded_source, reps=(self.beam_size, 1, 1))
        source_dynamic_init = mx.nd.tile(source_dynamic_init, reps=(self.beam_size, 1, 1))
        decoder_hidden_init = mx.nd.tile(decoder_hidden_init, reps=(self.beam_size, 1))
        decoder_states = [mx.nd.tile(state, reps=(self.beam_size, 1)) for state in decoder_states]
        source_length = mx.nd.tile(source_length, reps=(self.beam_size,))

        return encoded_source, source_dynamic_init, source_length, decoder_hidden_init, decoder_states

    def run_decoder(self,
                    encoded_source: mx.nd.NDArray,
                    dynamic_source: mx.nd.NDArray,
                    source_length: mx.nd.NDArray,
                    previous_word_id: mx.nd.NDArray,
                    previous_hidden: mx.nd.NDArray,
                    decoder_states: List[mx.nd.NDArray],
                    bucket_key: int) -> Tuple[mx.nd.NDArray, mx.nd.NDArray,
                                              mx.nd.NDArray, mx.nd.NDArray,
                                              List[mx.nd.NDArray]]:
        """
        Runs forward pass of the single-step decoder.

        :param encoded_source: Encoded source sentence.
        :param dynamic_source: Dynamic encoding of source sentence.
        :param source_length: Source length.
        :param previous_word_id: Previous predicted word id.
        :param previous_hidden: Previous hidden decoder state.
        :param decoder_states: Decoder states.
        :param bucket_key: Bucket key.
        :return: Probability distribution over next word, attention scores, dynamic source encoding,
                 next hidden state, next decoder states.
        """

        data = [encoded_source,
                dynamic_source,
                source_length,
                previous_word_id.as_in_context(self.context),
                previous_hidden] + decoder_states

        decoder_batch = mx.io.DataBatch(
            data=data,
            label=None, bucket_key=bucket_key, provide_data=self._get_decoder_data_shapes(bucket_key))
        # run forward pass
        self.decoder_module.forward(data_batch=decoder_batch, is_train=False)
        # collect outputs
        softmax_out, attention_probs, dynamic_source, next_hidden, *next_layer_states = \
            self.decoder_module.get_outputs()

        return softmax_out, attention_probs, dynamic_source, next_hidden, next_layer_states


def load_models(context: mx.context.Context,
                max_input_len: int,
                beam_size: int,
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None) \
        -> Tuple[List[InferenceModel], Dict[str, int], Dict[str, int]]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :return: List of models, source vocabulary, target vocabulary.
    """
    models, source_vocabs, target_vocabs = [], [], []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    for model_folder, checkpoint in zip(model_folders, checkpoints):

        source_vocabs.append(sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_SRC_NAME)))
        target_vocabs.append(sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_TRG_NAME)))
        model = InferenceModel(model_folder=model_folder,
                               context=context,
                               fused=False,
                               max_input_len=max_input_len,
                               beam_size=beam_size,
                               softmax_temperature=softmax_temperature,
                               checkpoint=checkpoint)
        models.append(model)

    # check vocabulary consistency
    assert all(set(vocab.items()) == set(source_vocabs[0].items()) for vocab in
               source_vocabs), "Source vocabulary ids do not match"
    assert all(set(vocab.items()) == set(target_vocabs[0].items()) for vocab in
               target_vocabs), "Target vocabulary ids do not match"

    return models, source_vocabs[0], target_vocabs[0]


TranslatorInput = NamedTuple('TranslatorInput', [
    ('id', int),
    ('sentence', str),
    ('tokens', List[str]),
])
"""
Required input for Translator.

:param id: Sentence id.
:param sentence: Input sentence.
:param tokens: List of input tokens.
"""

TranslatorOutput = NamedTuple('TranslatorOutput', [
    ('id', int),
    ('translation', str),
    ('tokens', List[str]),
    ('attention_matrix', np.ndarray),
    ('score', float),
])
"""
Output structure from Translator.

:param id: Id of input sentence.
:param translation: Translation string without sentence boundary tokens.
:param tokens: List of translated tokens.
:param attention_matrix: Attention matrix. Shape: (target_length, source_length).
:param score: Negative log probability of generated translation.
"""


class Translator:
    """
    Translator uses one or several models to translate input.
    It holds references to vocabularies to takes care of encoding input strings as word ids and conversion
    of target ids into a translation string.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param models: List of models.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    """

    def __init__(self,
                 context: mx.context.Context,
                 ensemble_mode: str,
                 models: List[InferenceModel],
                 vocab_source: Dict[str, int],
                 vocab_target: Dict[str, int]):
        self.context = context
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.vocab_target_inv = sockeye.vocab.reverse_vocab(self.vocab_target)
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}
        self.models = models
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        self.buckets = sockeye.data_io.define_buckets(self.models[0].max_input_len)
        logger.info("Translator (%d model(s) beam_size=%d ensemble_mode=%s)",
                    len(self.models), self.beam_size, "None" if len(self.models) == 1 else ensemble_mode)

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
        return -mx.nd.log(sockeye.utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = sockeye.utils.average_arrays([mx.nd.log(p) for p in predictions])
        return -mx.nd.log(mx.nd.softmax(log_probs))

    @staticmethod
    def make_input(sentence_id: int, sentence: str) -> TranslatorInput:
        """
        Returns TranslatorInput from input_string

        :param sentence_id: Input sentence id.
        :param sentence: Input sentence.
        :return: Input for translate method.
        """
        tokens = list(sockeye.data_io.get_tokens(sentence))
        return TranslatorInput(id=sentence_id, sentence=sentence.rstrip(), tokens=tokens)

    def translate(self, trans_input: TranslatorInput) -> TranslatorOutput:
        """
        Translates a TranslatorInput and returns a TranslatorOutput
        
        :param trans_input: TranslatorInput as returned by make_input().
        :return: translation result.
        """
        if not trans_input.tokens:
            TranslatorOutput(id=trans_input.id,
                             translation="",
                             tokens=[""],
                             attention_matrix=np.asarray([[0]]),
                             score=-np.inf)

        return self._make_result(trans_input, *self.translate_nd(*self._get_inference_input(trans_input.tokens)))

    def _get_inference_input(self, tokens: List[str]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, Optional[int]]:
        """
        Returns NDArray of source ids, NDArray of sentence length, and corresponding bucket_key

        :param tokens: List of input tokens.
        """
        bucket_key = sockeye.data_io.get_bucket(len(tokens), self.buckets)
        if bucket_key is None:
            logger.warning("Input (%d) exceeds max bucket size (%d). Stripping", len(tokens), self.buckets[-1])
            bucket_key = self.buckets[-1]
            tokens = tokens[:bucket_key]

        source = mx.nd.zeros((1, bucket_key))
        ids = sockeye.data_io.tokens2ids(tokens, self.vocab_source)
        for i, wid in enumerate(ids):
            source[0, i] = wid
        length = mx.nd.array([len(ids)])
        return source, length, bucket_key

    def _make_result(self,
                     trans_input: TranslatorInput,
                     target_ids: List[int],
                     attention_matrix: np.ndarray,
                     neg_logprob: float) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrix, and score.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param target_ids: List of translated ids.
        :param attention_matrix: Attention matrix.
        :return: TranslatorOutput.
        """
        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]
        target_string = C.TOKEN_SEPARATOR.join(
            target_token for target_id, target_token in zip(target_ids, target_tokens) if
            target_id not in self.stop_ids)
        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]
        return TranslatorOutput(id=trans_input.id,
                                translation=target_string,
                                tokens=target_tokens,
                                attention_matrix=attention_matrix,
                                score=neg_logprob)

    def translate_nd(self, source: mx.nd.NDArray, source_length: mx.nd.NDArray, bucket_key: int) \
            -> Tuple[List[int], np.ndarray, float]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source.
        :param source_length: Source length.
        :param bucket_key: Bucket key.
        
        :return: Sequence of translated ids, attention matrix, length-normalized negative log probability.
        """
        # allow output sentence to be at most 2 times the current bucket_key
        # TODO: max_output_length adaptive to source_length
        max_output_length = bucket_key * 2

        return self._beam_search(source, source_length, bucket_key, max_output_length)

    def _combine_predictions(self,
                             predictions: List[mx.nd.NDArray],
                             attention_prob_scores: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, np.ndarray]:
        """
        Returns combined predictions of models as negative log probabilities, as well as averaged attention prob scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = sockeye.utils.average_arrays(attention_prob_scores).asnumpy()

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            neg_logprobs = -mx.nd.log(predictions[0])
        else:
            neg_logprobs = self.interpolation_func(predictions)
        return neg_logprobs, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: mx.nd.NDArray,
                     bucket_key: int,
                     max_output_length: int) -> Tuple[List[int], np.ndarray, float]:
        """
        Translates a single sentence using beam search.
        """

        # encode source and initialize decoder states for each model
        model_encoded_source, model_dynamic_source, model_source_length, model_decoder_states = [], [], [], []
        model_prev_hidden = []

        for model in self.models:
            # encode input sentence and initialize decoder states
            # encoded_source: (self.beam_size, bucket_key, rnn_num_hidden)
            # decoder_states: [(self.beam_size, rnn_num_hidden),...]
            encoded_source, source_dynamic_init, source_length, prev_hidden, decoder_states = \
                model.run_encoder(source,
                                  source_length,
                                  bucket_key)
            model_encoded_source.append(encoded_source)
            model_dynamic_source.append(source_dynamic_init)
            model_source_length.append(source_length)
            model_decoder_states.append(decoder_states)
            model_prev_hidden.append(prev_hidden)

        # prev_target_word_id(s): (beam_size,)
        prev_target_word_id = mx.nd.zeros((self.beam_size,), ctx=self.context)
        prev_target_word_id[:] = self.start_id

        accumulated_scores = mx.nd.zeros((self.beam_size,), ctx=self.context)
        lengths = mx.nd.zeros((self.beam_size,), ctx=self.context)
        finished = [False] * self.beam_size
        sequences = [[] for _ in range(self.beam_size)]
        # one list of source word attention vectors per hypothesis
        attention_lists = [[] for _ in range(self.beam_size)]
        prev_hyp_indices = None

        for t in range(0, max_output_length):

            # decode one step for each model
            model_probs, model_attention_prob_score, model_next_hidden = [], [], []
            model_next_dynamic_source, model_next_decoder_states = [], []
            for model_index, model in enumerate(self.models):
                probs, attention_prob_score, next_dynamic_source, next_hidden, next_decoder_states = model.run_decoder(
                    model_encoded_source[model_index],
                    model_dynamic_source[model_index],
                    model_source_length[model_index],
                    prev_target_word_id,
                    model_prev_hidden[model_index],
                    model_decoder_states[model_index],
                    bucket_key)
                model_probs.append(probs)
                model_attention_prob_score.append(attention_prob_score)
                model_next_hidden.append(next_hidden)
                model_next_dynamic_source.append(next_dynamic_source)
                model_next_decoder_states.append(next_decoder_states)

            # combine predictions
            hyp_scores, attention_prob_score = self._combine_predictions(model_probs, model_attention_prob_score)

            for hyp_idx in range(self.beam_size):
                if not finished[hyp_idx]:
                    # re-normalize hypothesis score by length
                    hyp_scores[hyp_idx] = (hyp_scores[hyp_idx] + accumulated_scores[hyp_idx] * lengths[hyp_idx]) / (
                        lengths[hyp_idx] + 1)
                else:
                    hyp_scores[hyp_idx][:] = np.inf
                    hyp_scores[hyp_idx][C.PAD_ID] = accumulated_scores[hyp_idx]

            # get self.beam_size smallest hypotheses
            # prev_hyp_indices: row indices in hyp_scores.
            # next_word_ids: column indices in hyp_scores
            # accumulated_scores: chosen smallest scores in hyp_scores
            (prev_hyp_indices, next_word_ids), accumulated_scores = sockeye.utils.smallest_k(hyp_scores.asnumpy(),
                                                                                             self.beam_size, t == 0)

            # select attention according to hypothesis
            attention_prob_score = attention_prob_score[prev_hyp_indices, :]

            # list of new hypothesis that are now finished
            new_hyp_finished = [word_id in self.stop_ids for word_id in next_word_ids]
            new_sequences = [None for _ in range(self.beam_size)]
            new_attention_lists = [None for _ in range(self.beam_size)]
            for new_hyp_idx, prev_hyp_idx in enumerate(prev_hyp_indices):
                if not finished[prev_hyp_idx]:
                    new_sequences[new_hyp_idx] = sequences[prev_hyp_idx] + [next_word_ids[new_hyp_idx]]
                    new_attention_lists[new_hyp_idx] = attention_lists[prev_hyp_idx] + [
                        attention_prob_score[new_hyp_idx, :]]
                else:
                    new_sequences[new_hyp_idx] = sequences[prev_hyp_idx]
                    new_attention_lists[new_hyp_idx] = attention_lists[prev_hyp_idx]
                lengths[new_hyp_idx] = len(new_sequences[new_hyp_idx])

            finished = new_hyp_finished
            sequences = new_sequences
            attention_lists = new_attention_lists

            if all(new_hyp_finished):
                break

            # prepare new batch
            prev_hyp_indices_nd = mx.nd.array(prev_hyp_indices, ctx=self.context)
            prev_target_word_id = mx.nd.array(next_word_ids, ctx=self.context)
            model_prev_hidden = [mx.nd.take(next_hidden, prev_hyp_indices_nd) for next_hidden in model_next_hidden]
            model_dynamic_source = [mx.nd.take(next_dynamic_source, prev_hyp_indices_nd) for
                                    next_dynamic_source in model_next_dynamic_source]
            model_decoder_states = [[mx.nd.take(state, prev_hyp_indices_nd) for state in decoder_states] for
                                    decoder_states in model_next_decoder_states]

        # sequences & accumulated scores are in latest 'k-best order', thus 0th element is best
        best = 0
        # attention_matrix: (target_seq_len, source_seq_len)
        attention_matrix = np.stack(attention_lists[best], axis=0)
        return sequences[best], attention_matrix, accumulated_scores[best]

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
from typing import Dict, List, NamedTuple, Optional, Tuple, Callable

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import model
from . import utils
from . import vocab

logger = logging.getLogger(__name__)


class InferenceModel(model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

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
        self.model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", self.model_version)
        utils.check_version(self.model_version)

        # load config & determine parameter file
        config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
        if max_input_len is None:
            max_input_len = config.max_seq_len_source
        else:
            if max_input_len != config.max_seq_len_source:
                logger.warning("Model was trained with max_seq_len_source=%d, but using max_input_len=%d.",
                               config.max_seq_len_source, max_input_len)
        config.max_seq_len_source = max_input_len
        super().__init__(config)

        fname_params = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)

        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')

        self.beam_size = beam_size
        self.softmax_temperature = softmax_temperature
        self.encoder_batch_size = 1
        self.context = context

        self._build_model_components(fused)
        self.encoder_module = self._get_encoder_module()
        self.decoder_module = self._get_decoder_module()

        self.decoder_data_shapes_cache = dict()  # bucket_key -> shape cache
        max_encoder_data_shapes = self._get_encoder_data_shapes(self.config.max_seq_len_source)
        source_encoded_max_seq_len = self.encoder.get_encoded_seq_len(self.config.max_seq_len_source)
        max_decoder_data_shapes = self._get_decoder_data_shapes(source_encoded_max_seq_len)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(fname_params)
        self.encoder_module.init_params(arg_params=self.params, allow_missing=False)
        self.decoder_module.init_params(arg_params=self.params, allow_missing=False)

    def _get_encoder_module(self) -> mx.mod.BucketingModule:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Encoder BucketingModule.
        """

        def sym_gen(source_seq_len: int):
            source = mx.sym.Variable(C.SOURCE_NAME)
            source_length = utils.compute_lengths(source)

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, source_seq_len)
            # TODO(fhieber): Consider standardizing encoders to return batch-major data to avoid this line.
            source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

            # initial decoder states
            decoder_init_states = self.decoder.init_states(source_encoded,
                                                           source_encoded_length,
                                                           source_encoded_seq_len)

            data_names = [C.SOURCE_NAME]
            label_names = []
            return mx.sym.Group(decoder_init_states), data_names, label_names

        return mx.mod.BucketingModule(sym_gen=sym_gen,
                                      default_bucket_key=self.config.max_seq_len_source,
                                      context=self.context)

    def _get_decoder_module(self) -> mx.mod.BucketingModule:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the length of the ENCODED source sequence.

        :return: Decoder BucketingModule.
        """

        def sym_gen(source_encoded_seq_len: int):
            self.decoder.reset()
            prev_word_id = mx.sym.Variable(C.TARGET_PREVIOUS_NAME)
            states = self.decoder.state_variables()
            state_names = [state.name for state in states]
            logits, attention_probs, states = self.decoder.decode_step(prev_word_id,
                                                                       source_encoded_seq_len,
                                                                       *states)
            if self.softmax_temperature is not None:
                logits /= self.softmax_temperature

            softmax = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            data_names = [C.TARGET_PREVIOUS_NAME] + state_names
            label_names = []
            return mx.sym.Group([softmax, attention_probs] + states), data_names, label_names

        source_encoded_max_seq_len = self.encoder.get_encoded_seq_len(self.config.max_seq_len_source)
        return mx.mod.BucketingModule(sym_gen=sym_gen,
                                      default_bucket_key=source_encoded_max_seq_len,
                                      context=self.context)

    def _get_encoder_data_shapes(self, source_max_length: int) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param source_max_length: Maximum input length.
        :return: List of data descriptions.
        """
        return [mx.io.DataDesc(name=C.SOURCE_NAME,
                               shape=(self.encoder_batch_size, source_max_length),
                               layout=C.BATCH_MAJOR)]

    def _get_decoder_data_shapes(self, source_encoded_max_length) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module, given a bucket_key (source input length)
        Caches results for bucket_keys if called iteratively.

        :param source_encoded_max_length: Input length.
        :return: List of data descriptions.
        """
        return self.decoder_data_shapes_cache.setdefault(
            source_encoded_max_length,
            [mx.io.DataDesc(C.TARGET_PREVIOUS_NAME, (self.beam_size,), layout="N")] +
            self.decoder.state_shapes(self.beam_size, source_encoded_max_length, self.encoder.get_num_hidden()))

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    source_max_length: int) -> List[mx.nd.NDArray]:
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens.
        :param source_max_length: Bucket key.
        :return: Encoded source, source length, initial decoder hidden state, initial decoder hidden states.
        """
        batch = mx.io.DataBatch(data=[source],
                                label=None,
                                bucket_key=source_max_length,
                                provide_data=self._get_encoder_data_shapes(source_max_length))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_states = self.encoder_module.get_outputs()
        # replicate encoder/init module results beam size times
        decoder_states = [mx.nd.broadcast_axis(s, axis=0, size=self.beam_size) for s in decoder_states]
        return decoder_states

    def run_decoder(self, model_state: 'ModelState') -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState']:
        """
        Runs forward pass of the single-step decoder.

        :return: Probability distribution over next word, attention scores, updated model state.
        """
        batch = mx.io.DataBatch(
            data=[model_state.prev_target_word_id.as_in_context(self.context)] + model_state.decoder_states,
            label=None,
            bucket_key=model_state.bucket_key,
            provide_data=self._get_decoder_data_shapes(model_state.bucket_key))
        self.decoder_module.forward(data_batch=batch, is_train=False)
        probs, attention_probs, *model_state.decoder_states = self.decoder_module.get_outputs()
        return probs, attention_probs, model_state


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
        source_vocabs.append(vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_SRC_NAME)))
        target_vocabs.append(vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_TRG_NAME)))
        model = InferenceModel(model_folder=model_folder,
                               context=context,
                               fused=False,
                               max_input_len=max_input_len,
                               beam_size=beam_size,
                               softmax_temperature=softmax_temperature,
                               checkpoint=checkpoint)
        models.append(model)

    utils.check_condition(all(set(vocab.items()) == set(source_vocabs[0].items()) for vocab in source_vocabs),
                          "Source vocabulary ids do not match")
    utils.check_condition(all(set(vocab.items()) == set(target_vocabs[0].items()) for vocab in target_vocabs),
                          "Target vocabulary ids do not match")

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


class ModelState:
    """
    A ModelState encapsulates information about the decoder state of an InferenceModel.
    """

    def __init__(self,
                 bucket_key: int,
                 prev_target_word_id: mx.nd.NDArray,
                 decoder_states: List[mx.nd.NDArray]):
        self.bucket_key = bucket_key
        self.prev_target_word_id = prev_target_word_id
        self.decoder_states = decoder_states

    def sort_state(self, best_hyp_indices: mx.nd.NDArray, best_word_indices: mx.nd.NDArray):
        """
        Sorts states according to k-best order from last step in beam search.
        """
        self.prev_target_word_id = best_word_indices
        self.decoder_states = [mx.nd.take(ds, best_hyp_indices) for ds in self.decoder_states]


class LengthPenalty:
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016.

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float=1.0, beta: float=0.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1)**self.alpha

    def __call__(self, lengths: mx.nd.NDArray) -> mx.nd.NDArray:
        """
        Calculate the length penalty for the given vector of lengths.

        :param lengths: A matrix of sentence lengths of dimensionality (batch_size, 1).
        :return: The length penalty (batch_size, 1).
        """
        if self.alpha == 0.0:
            # no length penalty:
            return mx.nd.ones_like(lengths)
        else:
            # note: we avoid unnecessary addition or pow operations
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator**self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator


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
                 length_penalty: LengthPenalty,
                 models: List[InferenceModel],
                 vocab_source: Dict[str, int],
                 vocab_target: Dict[str, int]):
        self.context = context
        self.length_penalty = length_penalty
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}
        self.models = models
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        self.buckets = data_io.define_buckets(self.models[0].config.max_seq_len_source)
        self.pad_dist = mx.nd.full((self.beam_size, len(self.vocab_target)), val=np.inf, ctx=self.context)
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
        return -mx.nd.log(utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = utils.average_arrays([mx.nd.log(p) for p in predictions])
        return -mx.nd.log(mx.nd.softmax(log_probs))

    @staticmethod
    def make_input(sentence_id: int, sentence: str) -> TranslatorInput:
        """
        Returns TranslatorInput from input_string

        :param sentence_id: Input sentence id.
        :param sentence: Input sentence.
        :return: Input for translate method.
        """
        tokens = list(data_io.get_tokens(sentence))
        return TranslatorInput(id=sentence_id, sentence=sentence.rstrip(), tokens=tokens)

    def translate(self, trans_input: TranslatorInput) -> TranslatorOutput:
        """
        Translates a TranslatorInput and returns a TranslatorOutput

        :param trans_input: TranslatorInput as returned by make_input().
        :return: translation result.
        """
        if not trans_input.tokens:
            return TranslatorOutput(id=trans_input.id,
                                    translation="",
                                    tokens=[""],
                                    attention_matrix=np.asarray([[0]]),
                                    score=-np.inf)

        return self._make_result(trans_input, *self.translate_nd(*self._get_inference_input(trans_input.tokens)))

    def _get_inference_input(self, tokens: List[str]) -> Tuple[mx.nd.NDArray, int]:
        """
        Returns NDArray of source ids (shape=(1, bucket_key)) and corresponding bucket_key.

        :param tokens: List of input tokens.
        :return NDArray of source ids and bucket key.
        """
        bucket_key = data_io.get_bucket(len(tokens), self.buckets)
        if bucket_key is None:
            logger.warning("Input (%d) exceeds max bucket size (%d). Stripping", len(tokens), self.buckets[-1])
            bucket_key = self.buckets[-1]
            tokens = tokens[:bucket_key]

        utils.check_condition(C.PAD_ID == 0, "pad id should be 0")
        source = mx.nd.zeros((1, bucket_key))
        ids = data_io.tokens2ids(tokens, self.vocab_source)
        for i, wid in enumerate(ids):
            source[0, i] = wid
        return source, bucket_key

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

    def translate_nd(self,
                     source: mx.nd.NDArray,
                     bucket_key: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (1, bucket_key).
        :param bucket_key: Bucket key.

        :return: Sequence of translated ids, attention matrix, length-normalized negative log probability.
        """
        # allow output sentence to be at most 2 times the current bucket_key
        # TODO: max_output_length adaptive to source_length
        max_output_length = bucket_key * C.TARGET_MAX_LENGTH_FACTOR

        return self._get_best_from_beam(*self._beam_search(source, bucket_key, max_output_length))

    def _encode(self, source: mx.nd.NDArray, bucket_key: int) -> List[ModelState]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param source: Source ids. Shape: (1, bucket_key).
        :param bucket_key: Bucket key.
        :return: List of ModelStates.
        """
        prev_target_word_id = mx.nd.full((self.beam_size,), val=self.start_id, ctx=self.context)
        model_states = [ModelState(bucket_key=m.encoder.get_encoded_seq_len(bucket_key),
                                   prev_target_word_id=prev_target_word_id,
                                   decoder_states=m.run_encoder(source, bucket_key))
                        for m in self.models]
        return model_states

    def _decode_step(self, states: List[ModelState]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, List[ModelState]]:
        """
        Returns decoder predictions (combined from all models), attention scores, and updated states.

        :param: List of model states.
        :return: (probs, attention scores, list of model states)
        """
        model_probs, model_attention_probs, model_states = [], [], []
        for model, state in zip(self.models, states):
            probs, attention_probs, state = model.run_decoder(state)
            model_probs.append(probs)
            model_attention_probs.append(attention_probs)
            model_states.append(state)
        probs, attention_probs = self._combine_predictions(model_probs, model_attention_probs)
        return probs, attention_probs, model_states

    def _combine_predictions(self,
                             probs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined probabilities, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            neg_logprobs = -mx.nd.log(probs[0])
        else:
            neg_logprobs = self.interpolation_func(probs)
        return neg_logprobs, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     bucket_key: int,
                     max_output_length: int) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray]:
        """
        Translates a single sentence using beam search.

        :param source: Source ids. Shape: (1, bucket_key).
        :param bucket_key: Bucket key.
        :param max_output_length: Cap the output at this maximum length.
        :return List of lists of word ids, list of attentions, array of accumulated length-normalized
                negative log-probs.
        """
        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(bucket_key)
        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(bucket_key) for model in self.models),
                              "Models must agree on encoded sequence length")

        lengths = mx.nd.zeros((self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((self.beam_size,), dtype='int32', ctx=self.context)
        # sequences: (beam_size, output_length)
        sequences = mx.nd.array(np.full((self.beam_size, max_output_length), C.PAD_ID), dtype='int32', ctx=self.context)
        # attentions: (beam_size, output_length, encoded_source_length)
        attentions = mx.nd.zeros((self.beam_size, max_output_length, encoded_source_length), ctx=self.context)

        # best_hyp_indices: row indices of smallest scores (ascending).
        best_hyp_indices = mx.nd.zeros((self.beam_size,), ctx=self.context)
        # best_word_indices: column indices of smallest scores (ascending).
        best_word_indices = mx.nd.zeros((self.beam_size,), ctx=self.context, dtype='int32')
        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((self.beam_size, 1), ctx=self.context)

        # reset all padding distribution cells to np.inf
        self.pad_dist[:] = np.inf

        # (0) encode source sentence
        model_states = self._encode(source, bucket_key)

        for t in range(0, max_output_length):

            # (1) obtain next predictions and advance models' state
            # scores: (beam_size, target_vocab_size)
            # attention_scores: (beam_size, bucket_key)
            scores, attention_scores, model_states = self._decode_step(model_states)

            # (2) compute length-normalized accumulated scores in place
            if t == 0:  # only one hypothesis at t==0
                scores = scores[:1] / self.length_penalty(lengths[:1] + 1)
            else:
                # renormalize scores by length+1 ...
                scores = (scores + scores_accumulated * self.length_penalty(lengths)) / self.length_penalty(lengths + 1)
                # ... but not for finished hyps.
                # their predicted distribution is set to their accumulated scores at C.PAD_ID.
                self.pad_dist[:, C.PAD_ID] = scores_accumulated
                # this is equivalent to doing this in numpy:
                #   self.pad_dist[finished, :] = np.inf
                #   self.pad_dist[finished, C.PAD_ID] = scores_accumulated[finished]
                scores = mx.nd.where(finished, self.pad_dist, scores)

            # (3) get beam_size winning hypotheses
            # TODO(fhieber): once mx.nd.topk is sped-up no numpy conversion necessary anymore.
            (best_hyp_indices[:], best_word_indices_np), scores_accumulated_np = utils.smallest_k(scores.asnumpy(),
                                                                                                  self.beam_size)
            scores_accumulated[:] = np.expand_dims(scores_accumulated_np, axis=1)
            best_word_indices[:] = best_word_indices_np

            # (4) get hypotheses and their properties for beam_size winning hypotheses (ascending)
            sequences = mx.nd.take(sequences, best_hyp_indices)
            lengths = mx.nd.take(lengths, best_hyp_indices)
            finished = mx.nd.take(finished, best_hyp_indices)
            attention_scores = mx.nd.take(attention_scores, best_hyp_indices)
            attentions = mx.nd.take(attentions, best_hyp_indices)

            # (5) update best hypotheses, their attention lists and lengths (only for non-finished hyps)
            sequences[:, t] = mx.nd.expand_dims(best_word_indices, axis=1)
            attentions[:, t, :] = mx.nd.expand_dims(attention_scores, axis=1)
            lengths += mx.nd.cast(1 - mx.nd.expand_dims(finished, axis=1), dtype='float32')

            # (6) determine which hypotheses in the beam are now finished
            finished = ((best_word_indices == C.PAD_ID) + (best_word_indices == self.vocab_target[C.EOS_SYMBOL]))
            if mx.nd.sum(finished).asscalar() == self.beam_size:  # all finished
                break

            # (7) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices, best_word_indices)

        return sequences, attentions, scores_accumulated, lengths

    @staticmethod
    def _get_best_from_beam(sequences: mx.nd.NDArray,
                            attention_lists: mx.nd.NDArray,
                            accumulated_scores: mx.nd.NDArray,
                            lengths: mx.nd.NDArray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Return the best (aka top) entry from the n-best list.

        :param sequences: Array of word ids. Shape: (beam_size, bucket_key).
        :param attention_lists: Array of attentions over source words. Shape: (length, bucket_key).
        :param accumulated_scores: Array of length-normalized negative log-probs.
        :return: Top sequence, top attention matrix, top accumulated score (length-normalized negative log-probs).
        """
        # sequences & accumulated scores are in latest 'k-best order', thus 0th element is best
        best = 0
        length = int(lengths[best].asscalar())
        sequence = sequences[best][:length].asnumpy().tolist()
        # attention_matrix: (target_seq_len, source_seq_len)
        attention_matrix = np.stack(attention_lists[best].asnumpy()[:length, :], axis=0)
        score = accumulated_scores[best].asscalar()
        return sequence, attention_matrix, score

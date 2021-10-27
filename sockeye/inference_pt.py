# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as onp
import torch as pt

from . import constants as C
from . import data_io
from . import lexical_constraints as constrained
from . import lexicon
from . import utils
from . import vocab
from .beam_search_pt import get_search_algorithm, CandidateScorer, GreedySearch
from .inference import _concat_translations, _concat_nbest_translations, TranslatorInput, TranslatorOutput, Translation, \
    BadTranslatorInput, IndexedTranslation, IndexedTranslatorInput, empty_translation, BeamHistory, \
    _reduce_nbest_translations
from .model_pt import PyTorchSockeyeModel

logger = logging.getLogger(__name__)


def models_max_input_output_length(models: List[PyTorchSockeyeModel],
                                   num_stds: int,
                                   forced_max_input_length: Optional[int] = None,
                                   forced_max_output_length: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param forced_max_input_length: An optional overwrite of the maximum input length. Does not include eos.
    :param forced_max_output_length: An optional overwrite of the maximum output length. Does not include bos.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    max_mean = max(model.length_ratio_mean for model in models)
    max_std = max(model.length_ratio_std for model in models)
    supported_max_seq_len_source = min((model.max_supported_len_source for model in models))
    supported_max_seq_len_target = min((model.max_supported_len_target for model in models))
    return get_max_input_output_length(supported_max_seq_len_source,
                                       supported_max_seq_len_target,
                                       length_ratio_mean=max_mean,
                                       length_ratio_std=max_std,
                                       num_stds=num_stds,
                                       forced_max_input_len=forced_max_input_length,
                                       forced_max_output_len=forced_max_output_length)


def get_max_input_output_length(supported_max_seq_len_source: int,
                                supported_max_seq_len_target: int,
                                length_ratio_mean: float,
                                length_ratio_std: float,
                                num_stds: int,
                                forced_max_input_len: Optional[int] = None,
                                forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length. It takes into account optional maximum source and target lengths.

    :param supported_max_seq_len_source: The maximum source length supported by the models (includes eos).
    :param supported_max_seq_len_target: The maximum target length supported by the models (includes bos).
    :param length_ratio_mean: Length ratio mean computed on the training data (including bos/eos).
    :param length_ratio_std: The standard deviation of the length ratio.
    :param num_stds: The number of standard deviations the target length may exceed the mean target length (as long as
           the supported maximum length allows for this).
    :param forced_max_input_len: An optional overwrite of the maximum input length. Does not include eos.
    :param forced_max_output_len: An optional overwrite of the maximum output length. Does not include bos.
    :return: The maximum input length and a function to get the output length given the input length.
    """

    if num_stds < 0:
        factor = C.TARGET_MAX_LENGTH_FACTOR  # type: float
    else:
        factor = length_ratio_mean + (length_ratio_std * num_stds)

    if forced_max_input_len is not None:
        max_input_len = min(supported_max_seq_len_source, forced_max_input_len + C.SPACE_FOR_XOS)
    else:
        max_input_len = supported_max_seq_len_source

    def get_max_output_length(input_length: int):
        """
        Returns the maximum output length (including bos/eos) for inference given an input length that includes <eos>.
        """
        if forced_max_output_len is not None:
            return forced_max_output_len + C.SPACE_FOR_XOS
        return int(onp.ceil(factor * input_length))

    return max_input_len, get_max_output_length


class Translator:
    """
    Translator uses one or several models to translate input.
    The translator holds a reference to vocabularies to convert between word ids and text tokens for input and
    translation strings.

    :param device: Pytorch device to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param scorer: Hypothesis/Candidate scoring instance
    :param beam_search_stop: The stopping criterion.
    :param models: List of models.
    :param source_vocabs: Source vocabularies.
    :param target_vocabs: Target vocabularies.
    :param nbest_size: Size of nbest list of translations.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary selection. Can be a dict of
                             of named lexicons.
    :param avoid_list: Global list of phrases to exclude from the output.
    :param strip_unknown_words: If True, removes any <unk> symbols from outputs.
    :param sample: If True, sample from softmax multinomial instead of using topk.
    :param output_scores: Whether the scores will be needed as outputs. If True, scores will be normalized, negative
           log probabilities. If False, scores will be negative, raw logit activations if decoding with beam size 1
           and a single model.
    :param constant_length_ratio: If > 0, will override models' prediction of the length ratio (if any).
    :param hybridize: Whether to hybridize inference code.
    :param max_output_length_num_stds: Number of standard deviations to add as a safety margin when computing the
           maximum output length. If -1, returned maximum output lengths will always be 2 * input_length.
    :param max_input_length: Maximum input length this Translator should allow. If None, value will be taken from the
           model(s). Inputs larger than this value will be chunked and translated in sequence.
           If model(s) do not support given input length it will fall back to what the model(s) support.
    :param max_output_length: Maximum output length this Translator is allowed to decode. If None, value will be taken
           from the model(s). Decodings that do not finish within this limit, will be force-stopped.
           If model(s) do not support given input length it will fall back to what the model(s) support.
    """

    def __init__(self,
                 device: pt.device,
                 ensemble_mode: str,
                 scorer: CandidateScorer,
                 batch_size: int,
                 beam_search_stop: str,
                 models: List[PyTorchSockeyeModel],
                 source_vocabs: List[vocab.Vocab],
                 target_vocabs: List[vocab.Vocab],
                 beam_size: int = 5,
                 nbest_size: int = 1,
                 restrict_lexicon: Optional[Union[lexicon.TopKLexicon, Dict[str, lexicon.TopKLexicon]]] = None,
                 avoid_list: Optional[str] = None,
                 strip_unknown_words: bool = False,
                 sample: int = None,
                 output_scores: bool = False,
                 constant_length_ratio: float = 0.0,
                 hybridize: bool = True,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 max_input_length: Optional[int] = None,
                 max_output_length: Optional[int] = None,
                 softmax_temperature: Optional[float] = None,
                 prevent_unk: bool = False,
                 greedy: bool = False) -> None:
        self.device = device
        self.dtype = models[0].dtype
        self._scorer = scorer
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.beam_search_stop = beam_search_stop
        self.source_vocabs = source_vocabs
        self.vocab_targets = target_vocabs
        self.vocab_targets_inv = [vocab.reverse_vocab(v) for v in self.vocab_targets]
        self.restrict_lexicon = restrict_lexicon
        assert C.PAD_ID == 0, "pad id should be 0"
        self.stop_ids = {C.EOS_ID, C.PAD_ID}  # type: Set[int]
        self.strip_ids = self.stop_ids.copy()  # ids to strip from the output
        self.unk_id = C.UNK_ID
        if strip_unknown_words:
            self.strip_ids.add(self.unk_id)
        self.models = models

        # after models are loaded we ensured that they agree on max_input_length, max_output_length and batch size
        # set a common max_output length for all models.
        self._max_input_length, self._get_max_output_length = models_max_input_output_length(
            models,
            max_output_length_num_stds,
            forced_max_input_length=max_input_length,
            forced_max_output_length=max_output_length)

        self.nbest_size = nbest_size
        utils.check_condition(self.beam_size >= nbest_size, 'nbest_size must be smaller or equal to beam_size.')
        if self.nbest_size > 1:
            utils.check_condition(self.beam_search_stop == C.BEAM_SEARCH_STOP_ALL,
                                  "nbest_size > 1 requires beam_search_stop to be set to 'all'")

        self._search = get_search_algorithm(
            models=self.models,
            beam_size=self.beam_size,
            device=self.device,
            vocab_target=target_vocabs[0],  # only primary target factor used for constrained decoding.
            output_scores=output_scores,
            sample=sample,
            ensemble_mode=ensemble_mode,
            beam_search_stop=beam_search_stop,
            scorer=self._scorer,
            constant_length_ratio=constant_length_ratio,
            avoid_list=avoid_list,
            softmax_temperature=softmax_temperature,
            prevent_unk=prevent_unk,
            greedy=greedy)

        self._concat_translations = partial(_concat_nbest_translations if self.nbest_size > 1 else _concat_translations,
                                            stop_ids=self.stop_ids,
                                            scorer=self._scorer)  # type: Callable

        logger.info("PyTorchTranslator (%d model(s) beam_size=%d algorithm=%s, beam_search_stop=%s max_input_length=%s "
                    "nbest_size=%s ensemble_mode=%s max_batch_size=%d avoiding=%d dtype=%s softmax_temperature=%s)",
                    len(self.models),
                    self.beam_size,
                    "GreedySearch" if isinstance(self._search, GreedySearch) else "BeamSearch",
                    self.beam_search_stop,
                    self.max_input_length,
                    self.nbest_size,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.max_batch_size,
                    0 if self._search.global_avoid_trie is None else len(self._search.global_avoid_trie),
                    self.dtype,
                    softmax_temperature)

    @property
    def max_input_length(self) -> int:
        """
        Returns maximum input length for TranslatorInput objects passed to translate()
        """
        return self._max_input_length - C.SPACE_FOR_XOS

    @property
    def max_batch_size(self) -> int:
        """
        Returns the maximum batch size allowed for this Translator.
        """
        return self.batch_size

    @property
    def num_source_factors(self) -> int:
        return self.models[0].num_source_factors

    @property
    def num_target_factors(self) -> int:
        return self.models[0].num_target_factors

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
                if len(trans_input.tokens) > self.max_input_length:
                    # oversized input
                    logger.debug(
                        "Input %s has length (%d) that exceeds max input length (%d). "
                        "Splitting into chunks of size %d.",
                        trans_input.sentence_id, len(trans_input.tokens),
                        self.max_input_length, self.max_input_length)
                    chunks = [trans_input_chunk.with_eos()
                              for trans_input_chunk in
                              trans_input.chunks(self.max_input_length)]
                    input_chunks.extend([IndexedTranslatorInput(trans_input_idx, chunk_idx, chunk_input)
                                         for chunk_idx, chunk_input in enumerate(chunks)])
                else:
                    # regular input
                    input_chunks.append(IndexedTranslatorInput(trans_input_idx,
                                                               chunk_idx=0,
                                                               translator_input=trans_input.with_eos()))

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
            with pt.inference_mode():
                batch_translations = self._translate_np(*self._get_inference_input(translator_inputs))

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
                             trans_inputs: List[TranslatorInput]) -> Tuple[pt.tensor,
                                                                           int,
                                                                           Optional[lexicon.TopKLexicon],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           pt.tensor]:
        """
        Assembles the numerical data for the batch. This comprises an NDArray for the source sentences,
        the bucket key (padded source length), and a list of raw constraint lists, one for each sentence in the batch,
        an NDArray of maximum output lengths for each sentence in the batch.
        Each raw constraint list contains phrases in the form of lists of integers in the target language vocabulary.

        :param trans_inputs: List of TranslatorInputs.
        :return tensor of source ids (shape=(batch_size, bucket_key, num_factors)),
                tensor of valid source lengths, lexicon for vocabulary restriction, list of raw constraint
                lists, and list of phrases to avoid, and an ndarray of maximum output
                lengths.
        """
        batch_size = len(trans_inputs)
        lengths = [len(inp) for inp in trans_inputs]

        max_length = max(len(inp) for inp in trans_inputs)
        # assembling source ids on cpu array (faster) and copy to Translator.context (potentially GPU) in one go below.
        source = onp.zeros((batch_size, max_length, self.num_source_factors), dtype='int32')

        restrict_lexicon = None  # type: Optional[lexicon.TopKLexicon]
        raw_constraints = [None] * batch_size  # type: List[Optional[constrained.RawConstraintList]]
        raw_avoid_list = [None] * batch_size  # type: List[Optional[constrained.RawConstraintList]]

        max_output_lengths = []  # type: List[int]
        for j, trans_input in enumerate(trans_inputs):
            num_tokens = len(trans_input)  # includes eos
            max_output_lengths.append(self._get_max_output_length(num_tokens))
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
                raw_constraints[j] = [data_io.tokens2ids(phrase, self.vocab_targets[0]) for phrase in
                                      trans_input.constraints]

            if trans_input.avoid_list is not None:
                raw_avoid_list[j] = [data_io.tokens2ids(phrase, self.vocab_targets[0]) for phrase in
                                     trans_input.avoid_list]
                if any(self.unk_id in phrase for phrase in raw_avoid_list[j]):
                    logger.warning("Sentence %s: %s was found in the list of phrases to avoid; "
                                   "this may indicate improper preprocessing.", trans_input.sentence_id, C.UNK_SYMBOL)

        source = pt.tensor(source, device=self.device, dtype=pt.int32)
        source_length = pt.tensor(lengths, device=self.device, dtype=pt.int32)  # shape: (batch_size,)
        max_output_lengths = pt.tensor(max_output_lengths, device=self.device, dtype=pt.int32)
        return source, source_length, restrict_lexicon, raw_constraints, raw_avoid_list, max_output_lengths

    def _get_translation_tokens_and_factors(self, target_ids: List[List[int]]) -> Tuple[List[str],
                                                                                        str,
                                                                                        List[List[str]],
                                                                                        List[str]]:
        """
        Separates surface translation from factors. Input is a nested list of target ids.
        Creates tokens and output string for surface translation and for each factor, using the inverted target-side
        vocabularies. Ensures that factor strings are of the same length as the translation string.

        :param target_ids: Nested list of target ids.
        """
        all_target_tokens = []  # type: List[List[str]]
        all_target_strings = []  # type: List[str]
        # Strip any position where primary factor token is to be stripped
        pruned_target_ids = (tokens for tokens in target_ids if not tokens[0] in self.strip_ids)
        for factor_index, factor_sequence in enumerate(zip(*pruned_target_ids)):
            vocab_target_inv = self.vocab_targets_inv[factor_index]
            target_tokens = [vocab_target_inv[target_id] for target_id in factor_sequence]
            target_string = C.TOKEN_SEPARATOR.join(target_tokens)
            all_target_tokens.append(target_tokens)
            all_target_strings.append(target_string)

        if not all_target_strings:
            all_target_tokens = [[] for _ in range(len(self.vocab_targets_inv))]
            all_target_strings = ['' for _ in range(len(self.vocab_targets_inv))]

        tokens, *factor_tokens = all_target_tokens
        translation, *factor_translations = all_target_strings

        return tokens, translation, factor_tokens, factor_translations

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids and scores.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation and score.
        :return: TranslatorOutput.
        """
        primary_tokens, primary_translation, factor_tokens, factor_translations = \
            self._get_translation_tokens_and_factors(translation.target_ids)

        if translation.nbest_translations is None:
            nbest_translations = None
            nbest_tokens = None
            nbest_scores = None
            nbest_factor_translations = None
            nbest_factor_tokens = None
        else:
            nbest_tokens, nbest_translations, nbest_factor_tokens, nbest_factor_translations = [], [], [], []
            for nbest_target_ids in translation.nbest_translations.target_ids_list:
                ith_target_tokens, ith_primary_translation, ith_nbest_factor_tokens, ith_nbest_factor_translations = \
                    self._get_translation_tokens_and_factors(nbest_target_ids)
                nbest_tokens.append(ith_target_tokens)
                nbest_translations.append(ith_primary_translation)
                nbest_factor_tokens.append(ith_nbest_factor_tokens)
                nbest_factor_translations.append(ith_nbest_factor_translations)
            nbest_scores = translation.nbest_translations.scores

        return TranslatorOutput(sentence_id=trans_input.sentence_id,
                                translation=primary_translation,
                                tokens=primary_tokens,
                                score=translation.score,
                                pass_through_dict=trans_input.pass_through_dict,
                                beam_histories=translation.beam_histories,
                                nbest_translations=nbest_translations,
                                nbest_tokens=nbest_tokens,
                                nbest_scores=nbest_scores,
                                factor_translations=factor_translations,
                                factor_tokens=factor_tokens,
                                nbest_factor_translations=nbest_factor_translations,
                                nbest_factor_tokens=nbest_factor_tokens)

    def _translate_np(self,
                      source: pt.tensor,
                      source_length: pt.tensor,
                      restrict_lexicon: Optional[lexicon.TopKLexicon],
                      raw_constraints: List[Optional[constrained.RawConstraintList]],
                      raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                      max_output_lengths: pt.tensor) -> List[Translation]:
        """
        Translates source of source_length and returns list of Translations.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Valid source lengths.
        :param restrict_lexicon: Lexicon to use for vocabulary restriction.
        :param raw_constraints: A list of optional constraint lists.

        :return: List of translations.
        """
        return self._get_best_translations(*self._search(source,
                                                         source_length,
                                                         restrict_lexicon,
                                                         raw_constraints,
                                                         raw_avoid_list,
                                                         max_output_lengths))

    def _get_best_translations(self,
                               best_hyp_indices: pt.tensor,
                               best_word_indices: pt.tensor,
                               seq_scores: pt.tensor,
                               lengths: pt.tensor,
                               estimated_reference_lengths: Optional[pt.tensor] = None,
                               constraints: List[Optional[constrained.ConstrainedHypothesis]] = [],
                               beam_histories: Optional[List[BeamHistory]] = None) -> List[Translation]:
        """
        Return the nbest (aka n top) entries from the n-best list.

        :param best_hyp_indices: Array of best hypotheses indices ids. Shape: (batch * beam, num_beam_search_steps + 1).
        :param best_word_indices: Array of best hypotheses indices ids.
                                  Shape: (batch * beam, num_target_factors, num_beam_search_steps).
        :param seq_scores: Array of length-normalized negative log-probs. Shape: (batch * beam, 1)
        :param lengths: The lengths of all items in the beam. Shape: (batch * beam). Dtype: int32.
        :param estimated_reference_lengths: Predicted reference lengths.
        :param constraints: The constraints for all items in the beam. Shape: (batch * beam).
        :param beam_histories: The beam histories for each sentence in the batch.
        :return: List of Translation objects containing all relevant information.
        """
        best_hyp_indices = best_hyp_indices.cpu().numpy()
        best_word_indices = best_word_indices.cpu().numpy()
        seq_scores = seq_scores.cpu().numpy()
        lengths = lengths.cpu().numpy()
        estimated_reference_lengths = estimated_reference_lengths.cpu().numpy() if estimated_reference_lengths is not None else None
        batch_size = best_hyp_indices.shape[0] // self.beam_size
        nbest_translations = []  # type: List[List[Translation]]
        histories = beam_histories if beam_histories is not None else [None] * self.batch_size  # type: List
        reference_lengths = estimated_reference_lengths if estimated_reference_lengths is not None \
                                                        else onp.zeros((self.batch_size * self.beam_size, 1))
        for n in range(0, self.nbest_size):

            # Initialize the best_ids to the first item in each batch, plus current nbest index
            best_ids = onp.arange(n, batch_size * self.beam_size, self.beam_size, dtype='int32')

            # only check for constraints for 1-best translation for each sequence in batch
            if n == 0 and any(constraints):
                # For constrained decoding, select from items that have met all constraints (might not be finished)
                unmet = onp.array([c.num_needed() if c is not None else 0 for c in constraints])
                filtered = onp.where(unmet == 0, seq_scores.flatten(), onp.inf)
                filtered = filtered.reshape((batch_size, self.beam_size))
                best_ids += onp.argmin(filtered, axis=1).astype('int32', copy=False)

            # Obtain sequences for all best hypotheses in the batch. Shape: (batch, length)
            indices = self._get_best_word_indices_for_kth_hypotheses(best_ids, best_hyp_indices)
            indices_shape_1 = indices.shape[1]  # pylint: disable=unsubscriptable-object
            nbest_translations.append(
                    [self._assemble_translation(*x, unshift_target_factors=C.TARGET_FACTOR_SHIFT) for x in
                     zip(best_word_indices[indices,
                                           :,  # get all factors
                                           onp.arange(indices_shape_1)],
                         lengths[best_ids],
                         seq_scores[best_ids],
                         histories,
                         reference_lengths[best_ids])])

        # reorder and regroup lists
        reduced_translations = [_reduce_nbest_translations(grouped_nbest) for grouped_nbest in zip(*nbest_translations)]  # type: ignore
        return reduced_translations

    @staticmethod
    def _get_best_word_indices_for_kth_hypotheses(ks: onp.ndarray, all_hyp_indices: onp.ndarray) -> onp.ndarray:
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
        result = onp.zeros((batch_size, num_steps - 1), dtype=all_hyp_indices.dtype)
        # first index into the history of the desired hypotheses.
        pointer = all_hyp_indices[ks, -1]
        # for each column/step follow the pointer, starting from the penultimate column/step
        for step in range(num_steps - 2, -1, -1):
            result[:, step] = pointer
            pointer = all_hyp_indices[pointer, step]
        return result

    @staticmethod
    def _assemble_translation(sequence: onp.ndarray,
                              length: onp.ndarray,
                              seq_score: onp.ndarray,
                              beam_history: Optional[BeamHistory],
                              estimated_reference_length: Optional[float],
                              unshift_target_factors: bool = False) -> Translation:
        """
        Takes a set of data pertaining to a single translated item, performs slightly different
        processing on each, and merges it into a Translation object.
        :param sequence: Array of word ids. Shape: (bucketed_length, num_target_factors).
        :param length: The length of the translated segment.
        :param seq_score: Array of length-normalized negative log-probs.
        :param estimated_reference_length: Estimated reference length (if any).
        :param beam_history: The optional beam histories for each sentence in the batch.
        :return: A Translation object.
        """
        if unshift_target_factors:
            sequence = _unshift_target_factors(sequence, fill_last_with=C.EOS_ID)
        else:
            sequence = sequence.tolist()
        length = int(length)  # type: ignore
        sequence = sequence[:length]  # type: ignore
        score = float(seq_score)
        estimated_reference_length = float(estimated_reference_length) if estimated_reference_length else None
        beam_history_list = [beam_history] if beam_history is not None else []
        return Translation(sequence, score, beam_history_list,  # type: ignore
                           nbest_translations=None,
                           estimated_reference_length=estimated_reference_length)


def _unshift_target_factors(sequence: onp.ndarray, fill_last_with: int = C.EOS_ID):
    """
    Shifts back target factors so that they re-align with the words.

    :param sequence: Array of word ids. Shape: (bucketed_length, num_target_factors).
    """
    if len(sequence.shape) == 1 or sequence.shape[1] == 1:
        return sequence.tolist()
    num_factors_to_shift = sequence.shape[1] - 1
    _fillvalue = num_factors_to_shift * [fill_last_with]
    _words = sequence[:, 0].tolist()  # tokens from t==0 onwards
    _next_factors = sequence[1:, 1:].tolist()  # factors from t==1 onwards
    sequence = [(w, *fs) for w, fs in itertools.zip_longest(_words, _next_factors, fillvalue=_fillvalue)]  # type: ignore
    return sequence

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
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

from mxnet import np, context

from . import constants as C
from . import data_io
from . import lexical_constraints as constrained
from . import lexicon
from . import utils
from . import vocab
from .beam_search import get_search_algorithm, CandidateScorer, GreedySearch
from .model import SockeyeModel

logger = logging.getLogger(__name__)


def models_max_input_output_length(models: List[SockeyeModel],
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
        return int(np.ceil(factor * input_length))

    return max_input_len, get_max_output_length


BeamHistory = Dict[str, List]
Tokens = List[str]
TokenIds = List[List[int]]  # each token id may contain multiple factors
SentenceId = Union[int, str]


@dataclass
class TranslatorInput:
    """
    Object required by Translator.translate().
    If not None, `pass_through_dict` is an arbitrary dictionary instantiated from a JSON object
    via `make_input_from_dict()`, and it contains extra fields found in an input JSON object.
    If `--output-type json` is selected, all such fields that are not fields used or changed by
    Sockeye will be included in the output JSON object. This provides a mechanism for passing
    fields through the call to Sockeye.
    """

    sentence_id: SentenceId
    tokens: Tokens
    factors: Optional[List[Tokens]] = None
    restrict_lexicon: Optional[lexicon.TopKLexicon] = None
    constraints: Optional[List[Tokens]] = None
    avoid_list: Optional[List[Tokens]] = None
    pass_through_dict: Optional[Dict] = None

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
            pass_through_dict = copy.deepcopy(self.pass_through_dict) \
                if (chunk_id == 0 and self.pass_through_dict is not None) else None
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
        logger.exception(e, exc_info=True)  # type: ignore
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
        logger.exception(e, exc_info=True)  # type: ignore
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


@dataclass
class TranslatorOutput:
    """
    Output structure from Translator.

    sentence_id: Sentence id.
    translation: Translation string without sentence boundary tokens.
    tokens: List of translated tokens.
    score: Negative log probability of generated translation.
    pass_through_dict: Dictionary of key/value pairs to pass through when working with JSON.
    beam_histories: List of beam histories. The list will contain more than one
        history if it was split due to exceeding max_length.
    nbest_translations: List of nbest translations as strings.
    nbest_tokens: List of nbest translations as lists of tokens.
    nbest_scores: List of nbest scores, one for each nbest translation.
    factor_translations: List of factor outputs.
    factor_tokens: List of list of secondary factor tokens.
    """
    sentence_id: SentenceId
    translation: str
    tokens: Tokens
    score: float
    pass_through_dict: Optional[Dict[str, Any]] = None
    beam_histories: Optional[List[BeamHistory]] = None
    nbest_translations: Optional[List[str]] = None
    nbest_tokens: Optional[List[Tokens]] = None
    nbest_scores: Optional[List[float]] = None
    factor_translations: Optional[List[str]] = None
    factor_tokens: Optional[List[Tokens]] = None
    nbest_factor_translations: Optional[List[List[str]]] = None
    nbest_factor_tokens: Optional[List[List[Tokens]]] = None

    def json(self) -> Dict:
        """
        Returns a dictionary suitable for json.dumps() representing all
        the information in the class. It is initialized with any keys
        present in the corresponding `TranslatorInput` object's pass_through_dict.
        Keys from here that are not overwritten by Sockeye will thus be passed
        through to the output.

        :return: A dictionary.
        """
        _d = copy.deepcopy(self.pass_through_dict) if self.pass_through_dict is not None else {}  # type: Dict[str, Any]
        _d['sentence_id'] = self.sentence_id
        _d['translation'] = self.translation
        _d['score'] = self.score

        if self.nbest_translations is not None and len(self.nbest_translations) > 1:
            _d['translations'] = self.nbest_translations
            _d['scores'] = self.nbest_scores

        if self.factor_translations is not None:
            for i, factor in enumerate(self.factor_translations, 1):
                _d[f'factor{i}'] = factor

        if self.nbest_factor_translations is not None and len(self.nbest_factor_translations) > 1:
            _d['translations_factors'] = []
            for factor_translations in self.nbest_factor_translations:
                _d['translations_factors'].append(
                    {f'factor{i}': factor_translation for i, factor_translation in enumerate(factor_translations, 1)})

        return _d


@dataclass
class NBestTranslations:
    target_ids_list: List[TokenIds]
    scores: List[float]


@dataclass
class Translation:
    target_ids: TokenIds
    score: float
    beam_histories: List[BeamHistory] = None
    nbest_translations: NBestTranslations = None
    estimated_reference_length: Optional[float] = None


def empty_translation(add_nbest: bool = False) -> Translation:
    """
    Return an empty translation.

    :param add_nbest: Include (empty) nbest_translations in the translation object.
    """
    return Translation(target_ids=[],
                       score=-np.inf,
                       nbest_translations=NBestTranslations([], []) if add_nbest else None)


@dataclass
class IndexedTranslatorInput:
    """
    Translation of a chunk of a sentence.

    input_idx: Internal index of translation requests to keep track of the correct order of translations.
    chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
    input: The translator input.
    """
    input_idx: int
    chunk_idx: int
    translator_input: TranslatorInput


@dataclass(order=True)
class IndexedTranslation:
    """
    Translation of a chunk of a sentence.

    input_idx: Internal index of translation requests to keep track of the correct order of translations.
    chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
    translation: The translation of the input chunk.
    """
    input_idx: int
    chunk_idx: int
    translation: Translation


def _concat_nbest_translations(translations: List[Translation],
                               stop_ids: Set[int],
                               scorer: CandidateScorer) -> Translation:
    """
    Combines nbest translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol), score and length.
    :param stop_ids: The EOS symbols.
    :param scorer: Candidate scorer for recomputing score of concatenated translations.
    :return: A concatenation of the translations with a score.
    """
    expanded_translations = (_expand_nbest_translation(translation) for translation in translations)

    concatenated_translations = []  # type: List[Translation]

    for translations_to_concat in zip(*expanded_translations):
        concatenated_translations.append(_concat_translations(translations=list(translations_to_concat),
                                                              stop_ids=stop_ids,
                                                              scorer=scorer))

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
    scores = [translation.score for translation in nbest_translations_list]

    nbest_translations = NBestTranslations(sequences, scores)

    return Translation(best_translation.target_ids,
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
    for target_ids, score in zip(translation.nbest_translations.target_ids_list, translation.nbest_translations.scores):
        nbest_list.append(Translation(target_ids, score, translation.beam_histories,
                                      estimated_reference_length=translation.estimated_reference_length))

    return nbest_list


def _concat_translations(translations: List[Translation],
                         stop_ids: Set[int],
                         scorer: CandidateScorer) -> Translation:
    """
    Combines translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol), score and length.
    :param stop_ids: The EOS symbols.
    :param scorer: Candidate scorer for recomputing score of concatenated translations.
    :return: A concatenation of the translations with a score.
    """
    if len(translations) == 1:
        return translations[0]

    # Concatenation of all target ids without BOS and EOS
    target_ids = []
    beam_histories = []  # type: List[BeamHistory]
    estimated_reference_length = None  # type: Optional[float]

    for idx, translation in enumerate(translations):
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids)
        else:
            if translation.target_ids[-1][0] in stop_ids:
                target_ids.extend(translation.target_ids[:-1])
            else:
                target_ids.extend(translation.target_ids)
        beam_histories.extend(translation.beam_histories)
        if translation.estimated_reference_length is not None:
            if estimated_reference_length is None:
                estimated_reference_length = translation.estimated_reference_length
            else:
                estimated_reference_length += translation.estimated_reference_length

    # Unnormalize + sum and renormalize the score:
    raw_score = sum(scorer.unnormalize(t.score, len(t.target_ids), t.estimated_reference_length) for t in translations)
    score = scorer(raw_score, len(target_ids), estimated_reference_length)
    return Translation(target_ids, score, beam_histories,
                       estimated_reference_length=estimated_reference_length)


class Translator:
    """
    Translator uses one or several models to translate input.
    The translator holds a reference to vocabularies to convert between word ids and text tokens for input and
    translation strings.

    :param context: MXNet context to bind modules to.
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
                 context: context.Context,
                 ensemble_mode: str,
                 scorer: CandidateScorer,
                 batch_size: int,
                 beam_search_stop: str,
                 models: List[SockeyeModel],
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
        self.context = context
        self.dtype = C.DTYPE_FP32 if models[0].dtype == C.DTYPE_INT8 else models[0].dtype
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
            context=self.context,
            vocab_target=target_vocabs[0],  # only primary target factor used for constrained decoding.
            output_scores=output_scores,
            sample=sample,
            ensemble_mode=ensemble_mode,
            beam_search_stop=beam_search_stop,
            scorer=self._scorer,
            constant_length_ratio=constant_length_ratio,
            avoid_list=avoid_list,
            hybridize=hybridize,
            softmax_temperature=softmax_temperature,
            prevent_unk=prevent_unk,
            greedy=greedy)

        self._concat_translations = partial(_concat_nbest_translations if self.nbest_size > 1 else _concat_translations,
                                            stop_ids=self.stop_ids,
                                            scorer=self._scorer)  # type: Callable

        logger.info("Translator (%d model(s) beam_size=%d algorithm=%s, beam_search_stop=%s max_input_length=%s "
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
                             trans_inputs: List[TranslatorInput]) -> Tuple[np.ndarray,
                                                                           int,
                                                                           Optional[lexicon.TopKLexicon],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           np.ndarray]:
        """
        Assembles the numerical data for the batch. This comprises an NDArray for the source sentences,
        the bucket key (padded source length), and a list of raw constraint lists, one for each sentence in the batch,
        an NDArray of maximum output lengths for each sentence in the batch.
        Each raw constraint list contains phrases in the form of lists of integers in the target language vocabulary.

        :param trans_inputs: List of TranslatorInputs.
        :return ndarray of source ids (shape=(batch_size, bucket_key, num_factors)),
                ndarray of valid source lengths, lexicon for vocabulary restriction, list of raw constraint
                lists, and list of phrases to avoid, and an ndarray of maximum output
                lengths.
        """
        batch_size = len(trans_inputs)
        lengths = [len(inp) for inp in trans_inputs]

        max_length = max(len(inp) for inp in trans_inputs)
        # assembling source ids on cpu array (faster) and copy to Translator.context (potentially GPU) in one go below.
        source = np.zeros((batch_size, max_length, self.num_source_factors), dtype=np.float32, ctx=context.cpu())

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

        source = np.array(source, ctx=self.context)
        source_length = np.array(lengths, ctx=self.context, dtype=self.dtype)  # shape: (batch_size,)
        max_output_lengths = np.array(max_output_lengths, ctx=self.context, dtype='int32')
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
                      source: np.ndarray,
                      source_length: np.ndarray,
                      restrict_lexicon: Optional[lexicon.TopKLexicon],
                      raw_constraints: List[Optional[constrained.RawConstraintList]],
                      raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                      max_output_lengths: np.ndarray) -> List[Translation]:
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
                               best_hyp_indices: np.ndarray,
                               best_word_indices: np.ndarray,
                               seq_scores: np.ndarray,
                               lengths: np.ndarray,
                               estimated_reference_lengths: Optional[np.ndarray] = None,
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
        batch_size = best_hyp_indices.shape[0] // self.beam_size
        nbest_translations = []  # type: List[List[Translation]]
        histories = beam_histories if beam_histories is not None else [None] * self.batch_size  # type: List
        reference_lengths = estimated_reference_lengths if estimated_reference_lengths is not None \
                                                        else np.zeros((self.batch_size * self.beam_size, 1))
        for n in range(0, self.nbest_size):

            # Initialize the best_ids to the first item in each batch, plus current nbest index
            best_ids = np.arange(n, batch_size * self.beam_size, self.beam_size, dtype='int32')

            # only check for constraints for 1-best translation for each sequence in batch
            if n == 0 and any(constraints):
                # For constrained decoding, select from items that have met all constraints (might not be finished)
                unmet = np.array([c.num_needed() if c is not None else 0 for c in constraints])
                filtered = np.where(unmet == 0, seq_scores.flatten(), np.inf)
                filtered = filtered.reshape((batch_size, self.beam_size))
                best_ids += np.argmin(filtered, axis=1).astype('int32', copy=False)

            # Obtain sequences for all best hypotheses in the batch. Shape: (batch, length)
            indices = self._get_best_word_indices_for_kth_hypotheses(best_ids, best_hyp_indices)
            indices_shape_1 = indices.shape[1]  # pylint: disable=unsubscriptable-object
            nbest_translations.append(
                    [self._assemble_translation(*x, unshift_target_factors=C.TARGET_FACTOR_SHIFT) for x in
                     zip(best_word_indices[indices,
                                           :,  # get all factors
                                           np.arange(indices_shape_1)],
                         lengths[best_ids],
                         seq_scores[best_ids],
                         histories,
                         reference_lengths[best_ids])])

        # reorder and regroup lists
        reduced_translations = [_reduce_nbest_translations(grouped_nbest) for grouped_nbest in zip(*nbest_translations)]  # type: ignore
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
        for step in range(num_steps - 2, -1, -1):
            result[:, step] = pointer
            pointer = all_hyp_indices[pointer, step]
        return result

    @staticmethod
    def _assemble_translation(sequence: np.ndarray,
                              length: np.ndarray,
                              seq_score: np.ndarray,
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


def _unshift_target_factors(sequence: np.ndarray, fill_last_with: int = C.EOS_ID):
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

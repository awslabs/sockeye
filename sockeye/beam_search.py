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

import logging
import functools
import operator
from abc import abstractmethod, ABC
from typing import Tuple, Optional, List, Union

import mxnet as mx
import numpy as np

from . import constants as C
from . import lexical_constraints as constrained
from . import lexicon
from . import utils
from . import vocab
from .model import SockeyeModel

logger = logging.getLogger(__name__)


class _Inference(ABC):

    @abstractmethod
    def state_structure(self):
        raise NotImplementedError()

    @abstractmethod
    def encode_and_initialize(self,
                              inputs: mx.nd.NDArray,
                              valid_length: Optional[mx.nd.NDArray] = None):
        raise NotImplementedError()

    @abstractmethod
    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        raise NotImplementedError()


class _SingleModelInference(_Inference):

    def __init__(self,
                 model: SockeyeModel,
                 skip_softmax: bool = False,
                 constant_length_ratio: float = 0.0) -> None:
        self._model = model
        self._skip_softmax = skip_softmax
        self._const_lr = constant_length_ratio

    def state_structure(self) -> List:
        return [self._model.state_structure()]

    def encode_and_initialize(self, inputs: mx.nd.NDArray, valid_length: Optional[mx.nd.NDArray] = None):
        states, predicted_output_length = self._model.encode_and_initialize(inputs, valid_length, self._const_lr)
        predicted_output_length = predicted_output_length.expand_dims(axis=1)
        return states, predicted_output_length

    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        logits, states, _ = self._model.decode_step(step_input, states, vocab_slice_ids)
        if not self._skip_softmax:
            logits = logits.log_softmax(axis=-1)
        scores = -logits
        return scores, states


class _EnsembleInference(_Inference):

    def __init__(self,
                 models: List[SockeyeModel],
                 ensemble_mode: str = 'linear',
                 constant_length_ratio: float = 0.0) -> None:
        self._models = models
        if ensemble_mode == 'linear':
            self._interpolation = self.linear_interpolation
        elif ensemble_mode == 'log_linear':
            self._interpolation = self.log_linear_interpolation
        else:
            raise ValueError()
        self._const_lr = constant_length_ratio

    def state_structure(self) -> List:
        structure = []
        for model in self._models:
            structure.append(model.state_structure())
        return structure

    def encode_and_initialize(self, inputs: mx.nd.NDArray, valid_length: Optional[mx.nd.NDArray] = None):
        model_states = []  # type: List[mx.nd.NDArray]
        predicted_output_lengths = []  # type: List[mx.nd.NDArray]
        for model in self._models:
            states, predicted_output_length = model.encode_and_initialize(inputs, valid_length, self._const_lr)
            predicted_output_lengths.append(predicted_output_length)
            model_states += states
        # average predicted output lengths, (batch, 1)
        predicted_output_lengths = mx.nd.mean(mx.nd.stack(*predicted_output_lengths, axis=1), axis=1, keepdims=True)
        return model_states, predicted_output_lengths

    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        outputs = []  # type: List[mx.nd.NDArray]
        new_states = []  # type: List[mx.nd.NDArray]
        state_index = 0
        for model, model_state_structure in zip(self._models, self.state_structure()):
            model_states = states[state_index:state_index+len(model_state_structure)]
            state_index += len(model_state_structure)
            logits, model_states, _ = model.decode_step(step_input, model_states, vocab_slice_ids)
            probs = logits.softmax(axis=-1)
            outputs.append(probs)
            new_states += model_states
        scores = self._interpolation(outputs)
        return scores, new_states

    @staticmethod
    def linear_interpolation(predictions):
        return -mx.nd.log(utils.average_arrays(predictions))  # pylint: disable=invalid-unary-operand-type

    @staticmethod
    def log_linear_interpolation(predictions):
        log_probs = utils.average_arrays([p.log() for p in predictions])
        return -log_probs.log_softmax()  # pylint: disable=invalid-unary-operand-type


class UpdateScores(mx.gluon.HybridBlock):
    """
    A HybridBlock that updates the scores from the decoder step with accumulated scores.
    Inactive hypotheses receive score inf. Finished hypotheses receive their accumulated score for C.PAD_ID.
    Hypotheses at maximum length are forced to produce C.EOS_ID.
    All other options are set to infinity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert C.PAD_ID == 0, "This block only works with PAD_ID == 0"

    def hybrid_forward(self, F,
                       target_dists, finished, inactive,
                       scores_accumulated, lengths, max_lengths,
                       pad_dist, eos_dist):
        # broadcast hypothesis score to each prediction.
        # scores_accumulated. Shape: (batch*beam, 1)
        # target_dists. Shape: (batch*beam, vocab_size)
        scores = F.broadcast_add(target_dists, scores_accumulated)

        # Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
        # finished rows are inf everywhere except column zero (pad_id), which holds the accumulated model score.
        # Items that are finished (but not inactive) get their previous accumulated score for the <pad> symbol,
        # infinity otherwise.
        # pad_dist. Shape: (batch*beam, vocab_size)
        pad_dist = F.concat(scores_accumulated, pad_dist)
        scores = F.where(F.broadcast_logical_or(finished, inactive), pad_dist, scores)

        # Update lengths of all items, except those that were already finished. This updates
        # the lengths for inactive items, too, but that doesn't matter since they are ignored anyway.
        lengths = lengths + (1 - finished)

        # Items that are at their maximum length and not finished now are forced to produce the <eos> symbol.
        # That is, we keep scores for hypotheses below max length or finished, and 'force-eos' the rest.
        below_max_length = lengths < max_lengths
        scores = F.where(F.broadcast_logical_or(below_max_length, finished), scores, eos_dist + scores)

        return scores, lengths


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

    def forward(self, lengths):
        if isinstance(lengths, mx.nd.NDArray) or isinstance(lengths, mx.sym.Symbol):
            return super().forward(lengths)
        else:
            return self.hybrid_forward(None, lengths)

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


class BrevityPenalty(mx.gluon.HybridBlock):
    """
    Calculates the logarithmic brevity penalty as:
      weight * log min(1, exp(1 - ref_len / hyp_len)) = weight * min(0, 1 - ref_len / hyp_len).

    :param weight: Linear weight.
    """

    def __init__(self, weight: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = weight

    def forward(self, hyp_lengths, reference_lengths):
        if isinstance(hyp_lengths, mx.nd.NDArray) or isinstance(hyp_lengths, mx.sym.Symbol):
            return super().forward(hyp_lengths, reference_lengths)
        else:
            return self.hybrid_forward(None, hyp_lengths, reference_lengths)

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


class CandidateScorer(mx.gluon.HybridBlock):

    def __init__(self,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 brevity_penalty_weight: float = 0.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        with self.name_scope():
            self._lp = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)
            self._bp = None  # type: Optional[BrevityPenalty]
            if brevity_penalty_weight > 0.0:
                self._bp = BrevityPenalty(weight=brevity_penalty_weight)

    def forward(self, scores, lengths, reference_lengths):
        if isinstance(scores, mx.nd.NDArray) or isinstance(scores, mx.sym.Symbol):
            return super().forward(scores, lengths, reference_lengths)
        else:
            return self.hybrid_forward(None, scores, lengths, reference_lengths)

    def hybrid_forward(self, F, scores, lengths, reference_lengths):
        lp = self._lp(lengths)
        if self._bp is not None:
            bp = self._bp(lengths, reference_lengths)
        else:
            if F is None:
                bp = 0.0
            else:
                # avoid warning for unused input
                bp = F.zeros_like(reference_lengths) if reference_lengths is not None else 0.0
        return scores / lp - bp

    def unnormalize(self, scores, lengths, reference_lengths):
        bp = 0.0 if self._bp is None else self._bp(lengths, reference_lengths)
        return (scores + bp) * self._lp(lengths)


class SortNormalizeAndUpdateFinished(mx.gluon.HybridBlock):
    """
    A HybridBlock for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self,
                 dtype: str,
                 pad_id: int,
                 eos_id: int,
                 scorer: CandidateScorer,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype
        self.pad_id = pad_id
        self.eos_id = eos_id
        self._scorer = scorer

    def hybrid_forward(self, F, best_hyp_indices, best_word_indices,
                       finished, scores_accumulated, lengths, reference_lengths):

        # Reorder fixed-size beam data according to best_hyp_indices (ascending)
        finished = F.take(finished, best_hyp_indices)
        lengths = F.take(lengths, best_hyp_indices)
        reference_lengths = F.take(reference_lengths, best_hyp_indices)

        # Normalize hypotheses that JUST finished
        all_finished = F.broadcast_logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        newly_finished = F.broadcast_logical_xor(all_finished, finished)
        scores_accumulated = F.where(newly_finished,
                                     self._scorer(scores_accumulated,
                                                  F.cast(F.expand_dims(lengths, axis=1), self.dtype),
                                                  reference_lengths),
                                     scores_accumulated)

        # Recompute finished. Hypotheses are finished if they are extended with <pad> or <eos>
        finished = F.broadcast_logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)

        return finished, scores_accumulated, lengths, reference_lengths


class TopK(mx.gluon.HybridBlock):
    """
    Batch-wise topk operation.
    Forward method uses imperative shape inference, since both batch_size and vocab_size are dynamic
    during translation (due to variable batch size and potential vocabulary selection).
    """

    def __init__(self, k: int, **kwargs) -> None:
        """
        :param k: The number of smallest scores to return.
        """
        super().__init__(**kwargs)
        self.k = k

    def forward(self, scores, offset):
        """
        Get the lowest k elements per sentence from a `scores` matrix.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """
        vocab_size = scores.shape[1]
        batch_size = int(offset.shape[-1] / self.k)
        # Shape: (batch size, beam_size * vocab_size)
        batchwise_scores = scores.reshape(shape=(batch_size, self.k * vocab_size))
        indices, values = super().forward(batchwise_scores)
        best_hyp_indices, best_word_indices = mx.nd.unravel_index(indices, shape=(batch_size * self.k, vocab_size))
        if batch_size > 1:
            # Offsetting the indices to match the shape of the scores matrix
            best_hyp_indices += offset
        return best_hyp_indices, best_word_indices, values

    def hybrid_forward(self, F, scores):
        values, indices = F.topk(scores, axis=1, k=self.k, ret_typ='both', is_ascend=True)
        # Project indices back into original shape (which is different for t==1 and t>1)
        return F.reshape(F.cast(indices, 'int32'), shape=(-1,)), F.reshape(values, shape=(-1, 1))


class SampleK(mx.gluon.HybridBlock):
    """
    A HybridBlock for selecting a random word from each hypothesis according to its distribution.
    """
    def __init__(self, n, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n = n

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


def _repeat_states(states: List, beam_size: int, state_structure: List) -> List:
    repeated_states = []
    flat_structure = functools.reduce(operator.add, state_structure)
    assert len(states) == len(flat_structure), "Number of states do not match the defined state structure"
    for state, state_format in zip(states, flat_structure):
        if state_format == C.STEP_STATE or state_format == C.BIAS_STATE:
            repeat_axis = 0
        elif state_format == C.DECODER_STATE or state_format == C.ENCODER_STATE:
            # TODO: Change repeat axis to 1 when interleaved multihead attention is implemented
            repeat_axis = 0
        else:
            raise ValueError("Provided state format %s not recognized." % state_format)
        repeated_state = state.repeat(repeats=beam_size, axis=repeat_axis)
        repeated_states.append(repeated_state)
    return repeated_states


class SortStates(mx.gluon.HybridBlock):

    def __init__(self, state_structure, prefix):
        mx.gluon.HybridBlock.__init__(self, prefix=prefix)
        self.flat_structure = functools.reduce(operator.add, state_structure)

    def hybrid_forward(self, F, best_hyp_indices, *states):
        sorted_states = []
        assert len(states) == len(self.flat_structure), "Number of states do not match the defined state structure"
        for state, state_format in zip(states, self.flat_structure):
            if state_format == C.STEP_STATE or state_format == C.BIAS_STATE:
                sorted_state = F.take(state, best_hyp_indices)
            elif state_format == C.DECODER_STATE:
                # TODO: Change take axis to 1 when interleaved multihead attention is implemented
                sorted_state = F.take(state, best_hyp_indices)
            elif state_format == C.ENCODER_STATE:
                # No need for takes on encoder layer states
                sorted_state = state
            else:
                raise ValueError("Provided state format %s not recognized." % state_format)
            sorted_states.append(sorted_state)
        return sorted_states


class BeamSearch(mx.gluon.Block):
    """
    Features:
    - beam search stop
    - constraints (pos & neg)
    - ensemble decoding
    - vocabulary selection
    - sampling (TODO: check if its working correctly)

    Not supported:
    - beam pruning
    - beam history
    """

    def __init__(self,
                 beam_size: int,
                 dtype: str,
                 bos_id: int,
                 eos_id: int,
                 context: Union[mx.Context, List[mx.Context]],
                 output_vocab_size: int,
                 scorer: CandidateScorer,
                 num_source_factors: int,
                 inference: _Inference,
                 beam_search_stop: str = C.BEAM_SEARCH_STOP_ALL,
                 global_avoid_trie: Optional[constrained.AvoidTrie] = None,
                 sample: Optional[int] = None) -> None:
        super().__init__(prefix='beam_search_')
        self.beam_size = beam_size
        self.dtype = dtype
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.output_vocab_size = output_vocab_size
        self.context = context
        self._inference = inference
        self.beam_search_stop = beam_search_stop
        self.num_source_factors = num_source_factors
        self.global_avoid_trie = global_avoid_trie

        with self.name_scope():
            self._sort_states = SortStates(state_structure=self._inference.state_structure(),
                                           prefix='sort_states_')
            self._update_scores = UpdateScores(prefix='update_scores_')
            self._scorer = scorer
            self._sort_norm_and_update_finished = SortNormalizeAndUpdateFinished(
                prefix='sort_norm_and_update_finished_',
                dtype=self.dtype,
                pad_id=C.PAD_ID,
                eos_id=eos_id,
                scorer=scorer)

            self._sample = None  # type: Optional[mx.gluon.HybridBlock]
            self._top = None  # type: Optional[mx.gluon.HybridBlock]
            if sample is not None:
                self._sample = SampleK(sample)
            else:
                self._top = TopK(self.beam_size)

    def forward(self,
                source: mx.nd.NDArray,
                source_length: mx.nd.NDArray,
                restrict_lexicon: Optional[lexicon.TopKLexicon],
                raw_constraint_list: List[Optional[constrained.RawConstraintList]],
                raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                max_output_lengths: mx.nd.NDArray) -> Tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            List[Optional[np.ndarray]],
                                                            List[Optional[constrained.ConstrainedHypothesis]]]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Valid source lengths. Shape: (batch_size,).
        :param restrict_lexicon: Lexicon to use for vocabulary restriction.
        :param raw_constraint_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must appear in each output.
        :param raw_avoid_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must NOT appear in each output.
        :param max_output_lengths: NDArray of maximum output lengths per input in source.
                Shape: (batch_size,). Dtype: int32.
        :return List of best hypotheses indices, list of best word indices,
                array of accumulated length-normalized negative log-probs, hypotheses lengths,
                predicted lengths of references (if any), constraints (if any).
        """
        batch_size = source.shape[0]
        logger.debug("beam_search batch size: %d", batch_size)

        # Maximum beam search iterations (determined by longest input with eos)
        max_iterations = max_output_lengths.max().asscalar()
        logger.debug("max beam search iterations: %d", max_iterations)

        sample_best_hyp_indices = None
        if self._sample is not None:
            utils.check_condition(restrict_lexicon is None,
                                  "Sampling is not available when working with a restricted lexicon.")
            sample_best_hyp_indices = mx.nd.arange(0, batch_size * self.beam_size, dtype='int32')

        # General data structure: batch_size * beam_size blocks in total;
        # a full beam for each sentence, followed by the next beam-block for the next sentence and so on

        best_word_indices = mx.nd.full((batch_size * self.beam_size,), val=self.bos_id, ctx=self.context,
                                       dtype='int32')

        # offset for hypothesis indices in batch decoding
        offset = mx.nd.repeat(mx.nd.arange(0, batch_size * self.beam_size, self.beam_size,
                                           dtype='int32', ctx=self.context), self.beam_size)

        # locations of each batch item when first dimension is (batch * beam)
        batch_indices = mx.nd.arange(0, batch_size * self.beam_size, self.beam_size, dtype='int32', ctx=self.context)
        first_step_mask = mx.nd.full((batch_size * self.beam_size, 1), val=np.inf, ctx=self.context, dtype=self.dtype)
        first_step_mask[batch_indices] = 1.0
        pad_dist = mx.nd.full((batch_size * self.beam_size, self.output_vocab_size - 1), val=np.inf,
                              ctx=self.context, dtype=self.dtype)
        eos_dist = mx.nd.full((batch_size * self.beam_size, self.output_vocab_size), val=np.inf,
                              ctx=self.context, dtype=self.dtype)
        eos_dist[:, C.EOS_ID] = 0

        # Best word and hypotheses indices across beam search steps from topk operation.
        best_hyp_indices_list = []  # type: List[mx.nd.NDArray]
        best_word_indices_list = []  # type: List[mx.nd.NDArray]

        lengths = mx.nd.zeros((batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        finished = mx.nd.zeros((batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # Extending max_output_lengths to shape (batch_size * beam_size,)
        max_output_lengths = mx.nd.repeat(max_output_lengths, self.beam_size)

        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((batch_size * self.beam_size, 1), ctx=self.context, dtype=self.dtype)

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        vocab_slice_ids = None  # type: Optional[mx.nd.NDArray]
        if restrict_lexicon:
            source_words = utils.split(source, num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            vocab_slice_ids = restrict_lexicon.get_trg_ids(source_words.astype("int32").asnumpy())
            if any(raw_constraint_list):
                # Add the constraint IDs to the list of permissibled IDs, and then project them into the reduced space
                constraint_ids = np.array([word_id for sent in raw_constraint_list for phr in sent for word_id in phr])
                vocab_slice_ids = np.lib.arraysetops.union1d(vocab_slice_ids, constraint_ids)
                full_to_reduced = dict((val, i) for i, val in enumerate(vocab_slice_ids))
                raw_constraint_list = [[[full_to_reduced[x] for x in phr] for phr in sent] for sent in
                                       raw_constraint_list]
            # Pad to a multiple of 8.
            vocab_slice_ids = np.pad(vocab_slice_ids, (0, 7 - ((len(vocab_slice_ids) - 1) % 8)),
                                     mode='constant', constant_values = self.eos_id)
            vocab_slice_ids = mx.nd.array(vocab_slice_ids, ctx=self.context, dtype='int32')

            if vocab_slice_ids.shape[0] < self.beam_size + 1:
                # This fixes an edge case for toy models, where the number of vocab ids from the lexicon is
                # smaller than the beam size.
                logger.warning("Padding vocab_slice_ids (%d) with EOS to have at least %d+1 elements to expand",
                               vocab_slice_ids.shape[0], self.beam_size)
                n = self.beam_size - vocab_slice_ids.shape[0] + 1
                vocab_slice_ids = mx.nd.concat(vocab_slice_ids,
                                               mx.nd.full((n,), val=self.eos_id, ctx=self.context, dtype='int32'),
                                               dim=0)

            pad_dist = mx.nd.full((batch_size * self.beam_size, vocab_slice_ids.shape[0] - 1),
                                  val=np.inf, ctx=self.context)
            eos_dist = mx.nd.full((batch_size * self.beam_size, vocab_slice_ids.shape[0]),
                                  val=np.inf, ctx=self.context)
            eos_dist[:, C.EOS_ID] = 0

        # Initialize the beam to track constraint sets, where target-side lexical constraints are present
        constraints = constrained.init_batch(raw_constraint_list, self.beam_size, self.bos_id, self.eos_id)

        if self.global_avoid_trie or any(raw_avoid_list):
            avoid_states = constrained.AvoidBatch(batch_size, self.beam_size,
                                                  avoid_list=raw_avoid_list,
                                                  global_avoid_trie=self.global_avoid_trie)
            avoid_states.consume(best_word_indices)

        # (0) encode source sentence, returns a list
        model_states, estimated_reference_lengths = self._inference.encode_and_initialize(source, source_length)
        # repeat states to beam_size
        model_states = _repeat_states(model_states, self.beam_size, self._inference.state_structure())

        # Records items in the beam that are inactive. At the beginning (t==1), there is only one valid or active
        # item on the beam for each sentence
        inactive = mx.nd.zeros((batch_size * self.beam_size), dtype='int32', ctx=self.context)
        t = 1
        for t in range(1, max_iterations + 1):  # TODO: max_iterations + 1 is the MINIMUM to get correct results right now
            # (1) obtain next predictions and advance models' state
            # target_dists: (batch_size * beam_size, target_vocab_size)
            target_dists, model_states = self._inference.decode_step(best_word_indices, model_states, vocab_slice_ids)

            # (2) Produces the accumulated cost of target words in each row.
            # There is special treatment for finished and inactive rows: inactive rows are inf everywhere;
            # finished rows are inf everywhere except column zero, which holds the accumulated model score
            scores, lengths = self._update_scores(target_dists,
                                                  finished,
                                                  inactive,
                                                  scores_accumulated,
                                                  lengths,
                                                  max_output_lengths,
                                                  pad_dist,
                                                  eos_dist)

            # Mark entries that should be blocked as having a score of np.inf
            if self.global_avoid_trie or any(raw_avoid_list):
                block_indices = avoid_states.avoid()
                if len(block_indices) > 0:
                    scores[block_indices] = np.inf
                    if self._sample is not None:
                        target_dists[block_indices] = np.inf

            # (3) Get beam_size winning hypotheses for each sentence block separately. Only look as
            # far as the active beam size for each sentence.
            if self._sample is not None:
                best_hyp_indices, best_word_indices, scores_accumulated = self._sample(scores,
                                                                                       target_dists,
                                                                                       finished,
                                                                                       sample_best_hyp_indices)
            else:
                # On the first timestep, all hypotheses have identical histories, so force topk() to choose extensions
                # of the first row only by setting all other rows to inf
                if t == 1:
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

            # (4) Normalize the scores of newly finished hypotheses. Note that after this until the
            # next call to topk(), hypotheses may not be in sorted order.
            finished, scores_accumulated, lengths, estimated_reference_lengths = self._sort_norm_and_update_finished(
                best_hyp_indices,
                best_word_indices,
                finished,
                scores_accumulated,
                lengths,
                estimated_reference_lengths)

            # Collect best hypotheses, best word indices
            best_hyp_indices_list.append(best_hyp_indices)
            best_word_indices_list.append(best_word_indices)

            if self._should_stop(finished, batch_size):
                break

            # (5) update models' state with winning hypotheses (ascending)
            model_states = self._sort_states(best_hyp_indices, *model_states)

        logger.debug("Finished after %d out of %d steps.", t, max_iterations)

        # (9) Sort the hypotheses within each sentence (normalization for finished hyps may have unsorted them).
        folded_accumulated_scores = scores_accumulated.reshape((batch_size,
                                                                self.beam_size * scores_accumulated.shape[-1]))
        indices = mx.nd.cast(mx.nd.argsort(folded_accumulated_scores.astype('float32'), axis=1), dtype='int32').reshape((-1,))
        best_hyp_indices, _ = mx.nd.unravel_index(indices, scores_accumulated.shape) + offset
        scores_accumulated = scores_accumulated.take(best_hyp_indices)
        best_hyp_indices_list.append(best_hyp_indices)
        lengths = lengths.take(best_hyp_indices)
        all_best_hyp_indices = mx.nd.stack(*best_hyp_indices_list, axis=1)
        all_best_word_indices = mx.nd.stack(*best_word_indices_list, axis=1)
        constraints = [constraints[x] for x in best_hyp_indices.asnumpy()]

        return all_best_hyp_indices.asnumpy(), \
               all_best_word_indices.asnumpy(), \
               scores_accumulated.asnumpy(), \
               lengths.asnumpy().astype('int32'), \
               estimated_reference_lengths.asnumpy(), \
               constraints

    def _should_stop(self, finished, batch_size):
        if self.beam_search_stop == C.BEAM_SEARCH_STOP_FIRST:
            at_least_one_finished = finished.reshape((batch_size, self.beam_size)).sum(axis=1) > 0
            return at_least_one_finished.sum().asscalar() == batch_size
        else:
            return finished.sum().asscalar() == batch_size * self.beam_size  # all finished


def get_beam_search(models: List[SockeyeModel],
                    beam_size: int,
                    context: Union[mx.Context, List[mx.Context]],
                    vocab_target: vocab.Vocab,
                    output_scores: bool,
                    scorer: CandidateScorer,
                    ensemble_mode: str = 'linear',
                    beam_search_stop: str = C.BEAM_SEARCH_STOP_ALL,
                    constant_length_ratio: float = 0.0,
                    avoid_list: Optional[str] = None,
                    sample: Optional[int] = None,
                    hybridize: bool = True) -> BeamSearch:

    inference = None  # type: Optional[_Inference]
    if len(models) == 1:
        skip_softmax = beam_size == 1 and not output_scores and not sample
        if skip_softmax:
            logger.info("Enabled skipping softmax for a single model and greedy decoding.")
        inference = _SingleModelInference(model=models[0],
                                          skip_softmax=skip_softmax, constant_length_ratio=constant_length_ratio)
    else:
        inference = _EnsembleInference(models=models,
                                       ensemble_mode=ensemble_mode,
                                       constant_length_ratio=constant_length_ratio)

    global_avoid_trie = None if avoid_list is None else constrained.get_avoid_trie(avoid_list, vocab_target)
    bs = BeamSearch(
        beam_size=beam_size,
        dtype=C.DTYPE_FP32 if models[0].dtype == C.DTYPE_INT8 else models[0].dtype,
        bos_id=C.BOS_ID,
        eos_id=C.EOS_ID,
        context=context,
        output_vocab_size=models[0].output_layer_vocab_size,
        beam_search_stop=beam_search_stop,
        scorer=scorer,
        sample=sample,
        num_source_factors=models[0].num_source_factors,
        global_avoid_trie=global_avoid_trie,
        inference=inference
    )
    bs.initialize()
    if hybridize:
        bs.hybridize(static_alloc=True)
    return bs

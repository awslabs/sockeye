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

import mxnet as mx
import numpy as np
from abc import abstractmethod, ABC

from . import lexical_constraints as constrained
from . import lexicon
from . import utils
from . import vocab
from . import constants as C
from .model import SockeyeModel
from typing import Tuple, Optional, List, Dict, Callable, Union, cast
import logging

logger = logging.getLogger(__name__)


# TODO: better name for that class?
class Inference(ABC):

    @abstractmethod
    def encode_and_initialize(self,
                              inputs: mx.nd.NDArray,
                              valid_length: Optional[mx.nd.NDArray] = None,
                              constant_length_ratio: float = 0.0):
        raise NotImplementedError()

    @abstractmethod
    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        raise NotImplementedError()


class SingleModelInference(Inference):

    def __init__(self, model: SockeyeModel, skip_softmax: bool = False) -> None:
        self._model = model
        self._skip_softmax = skip_softmax

    def encode_and_initialize(self,
                              inputs: mx.nd.NDArray,
                              valid_length: Optional[mx.nd.NDArray] = None,
                              constant_length_ratio: float = 0.0):
        states, predicted_output_length = self._model.encode_and_initialize(inputs, valid_length, constant_length_ratio)
        predicted_output_length = predicted_output_length.expand_dims(axis=1)
        return states, predicted_output_length

    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        logits, states, _ = self._model.decode_step(step_input, states, vocab_slice_ids)
        logits = logits.astype('float32', copy=False)
        scores = -logits if self._skip_softmax else -logits.log_softmax(axis=-1)
        return scores, states


class EnsembleInference(Inference):

    def __init__(self, models: List[SockeyeModel], ensemble_mode: str = 'linear') -> None:
        self._models = models
        if ensemble_mode == 'linear':
            self._interpolation = self.linear_interpolation
        elif ensemble_mode == 'log_linear':
            self._interpolation = self.log_linear_interpolation
        else:
            raise ValueError()

    def encode_and_initialize(self,
                              inputs: mx.nd.NDArray,
                              valid_length: Optional[mx.nd.NDArray] = None,
                              constant_length_ratio: float = 0.0):
        model_states = []  # type: List[List[mx.nd.NDArray]]
        predicted_output_lengths = []  # type: List[mx.nd.NDArray]
        for model in self._models:
            states, predicted_output_length = model.encode_and_initialize(inputs, valid_length, constant_length_ratio)
            predicted_output_lengths.append(predicted_output_length)
            model_states.append(states)
        # average predicted output lengths, (batch, 1)
        predicted_output_lengths = mx.nd.mean(mx.nd.stack(*predicted_output_lengths, axis=1), axis=1, keepdims=True)
        return model_states, predicted_output_lengths

    def decode_step(self,
                    step_input: mx.nd.NDArray,
                    states: List,
                    vocab_slice_ids: Optional[mx.nd.NDArray] = None):
        outputs, new_states = [], []
        for model, model_states in zip(self._models, states):
            logits, model_states, _ = model.decode_step(step_input, model_states, vocab_slice_ids)
            logits = logits.astype('float32', copy=False)
            probs = logits.softmax(axis=-1)
            outputs.append(probs)
            new_states.append(model_states)
        scores = self._interpolation(outputs)
        return scores, new_states

    @staticmethod
    def linear_interpolation(predictions):
        return -mx.nd.log(utils.average_arrays(predictions))  # pylint: disable=invalid-unary-operand-type

    @staticmethod
    def log_linear_interpolation(predictions):
        log_probs = utils.average_arrays([p.log() for p in predictions])
        return -log_probs.log_softmax()  # pylint: disable=invalid-unary-operand-type


def _repeat_states(states: List, beam_size) -> List:
    repeated_states = []
    for state in states:
        if isinstance(state, List):
            state = _repeat_states(state, beam_size)
        elif isinstance(state, mx.nd.NDArray):
            state = state.repeat(repeats=beam_size, axis=0)
        else:
            ValueError("state list can only be nested list or NDArrays")
        repeated_states.append(state)
    return repeated_states


def _sort_states(states: List, best_hyp_indices: mx.nd.NDArray) -> List:
    sorted_states = []
    for state in states:
        if isinstance(state, List):
            state = _sort_states(state, best_hyp_indices)
        elif isinstance(state, mx.nd.NDArray):
            state = mx.nd.take(state, best_hyp_indices)
        else:
            ValueError("state list can only be nested list or NDArrays")
        sorted_states.append(state)
    return sorted_states


class BeamSearch(mx.gluon.Block):
    # TODO: misses constraint decoding

    def __init__(self,
                 beam_size: int,
                 start_id: int,
                 context: Union[mx.Context, List[mx.Context]],
                 vocab_target: vocab.Vocab,
                 beam_search_stop: str,
                 num_source_factors: int,
                 get_max_output_length: Callable,
                 inference: Inference):
        super().__init__(prefix='beam_search_')
        self.beam_size = beam_size
        self.start_id = start_id
        self.context = context
        self.vocab_target = vocab_target
        self._get_max_output_length = get_max_output_length
        self._inference = inference
        self.skip_topk = False  # TODO
        self.beam_search_stop = beam_search_stop
        self.num_source_factors = num_source_factors
        self.constant_length_ratio = 0.0  # TODO
        self.global_avoid_trie = None  # TODO
        self.length_penalty = LengthPenalty(1.0, 0.0)  # TODO
        self.brevity_penalty = None #BrevityPenalty(weight=0.0)  # TODO
        brevity_penalty_weight = self.brevity_penalty.weight if self.brevity_penalty is not None else 0.0

        with self.name_scope():
            self._sort_by_index = SortByIndex(prefix='sort_by_index_')
            self._update_scores = UpdateScores(prefix='update_scores_')
            self._norm_and_update_finished = NormalizeAndUpdateFinished(prefix='norm_and_update_finished_',
                                                                        pad_id=C.PAD_ID,
                                                                        eos_id=vocab_target[C.EOS_SYMBOL],
                                                                        length_penalty_alpha=self.length_penalty.alpha,
                                                                        length_penalty_beta=self.length_penalty.beta,
                                                                        brevity_penalty_weight=brevity_penalty_weight)
            self._top = TopK(self.beam_size)  # type: mx.gluon.HybridBlock

    def forward(self,
                source: mx.nd.NDArray,
                source_length: mx.nd.NDArray,
                restrict_lexicon: Optional[lexicon.TopKLexicon],
                raw_constraint_list: List[Optional[constrained.RawConstraintList]],
                raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                 : mx.nd.NDArray) -> Tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            List[Optional[np.ndarray]]]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Valid source lengths. Shape: (batch_size,).
        :param restrict_lexicon: Lexicon to use for vocabulary restriction.
        :param raw_constraint_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must appear in each output.
        :param raw_avoid_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must NOT appear in each output.
        :return List of best hypotheses indices, list of best word indices,
                array of accumulated length-normalized negative log-probs, hypotheses lengths,
                predicted lengths of references (if any), constraints (if any), beam histories (if any).
        """
        batch_size = source.shape[0]
        logger.debug("_beam_search batch size: %d", batch_size)

        # Maximum output length
        max_output_length = self._get_max_output_length(source.shape[1])

        # General data structure: batch_size * beam_size blocks in total;
        # a full beam for each sentence, folloed by the next beam-block for the next sentence and so on

        best_word_indices = mx.nd.full((batch_size * self.beam_size,), val=self.start_id, ctx=self.context,
                                       dtype='int32')

        # offset for hypothesis indices in batch decoding
        offset = mx.nd.repeat(mx.nd.arange(0, batch_size * self.beam_size, self.beam_size,
                                           dtype='int32', ctx=self.context), self.beam_size)

        # locations of each batch item when first dimension is (batch * beam)
        batch_indices = mx.nd.arange(0, batch_size * self.beam_size, self.beam_size, dtype='int32', ctx=self.context)
        first_step_mask = mx.nd.full((batch_size * self.beam_size, 1), val=np.inf, ctx=self.context, dtype='float32')
        first_step_mask[batch_indices] = 1.0
        pad_dist = mx.nd.full((batch_size * self.beam_size, len(self.vocab_target) - 1), val=np.inf,
                              ctx=self.context, dtype='float32')

        # Best word and hypotheses indices across beam search steps from topk operation.
        best_hyp_indices_list = []  # type: List[mx.nd.NDArray]
        best_word_indices_list = []  # type: List[mx.nd.NDArray]

        lengths = mx.nd.zeros((batch_size * self.beam_size, 1), ctx=self.context, dtype='float32')
        finished = mx.nd.zeros((batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # Extending max_output_lengths to shape (batch_size * beam_size,)
        max_output_lengths = mx.nd.repeat(max_output_lengths, self.beam_size)

        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((batch_size * self.beam_size, 1), ctx=self.context, dtype='float32')

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        vocab_slice_ids = None  # type: Optional[mx.nd.NDArray]
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

        # Initialize the beam to track constraint sets, where target-side lexical constraints are present
        constraints = constrained.init_batch(raw_constraint_list, self.beam_size, self.start_id,
                                             self.vocab_target[C.EOS_SYMBOL])

        if self.global_avoid_trie or any(raw_avoid_list):
            avoid_states = constrained.AvoidBatch(batch_size, self.beam_size,
                                                  avoid_list=raw_avoid_list,
                                                  global_avoid_trie=self.global_avoid_trie)
            avoid_states.consume(best_word_indices)

        # (0) encode source sentence, returns a list
        model_states, estimated_reference_lengths = self._inference.encode_and_initialize(source,
                                                                                          source_length,
                                                                                          self.constant_length_ratio)
        # repeat states to beam_size
        model_states = _repeat_states(model_states, self.beam_size)

        # Records items in the beam that are inactive. At the beginning (t==1), there is only one valid or active
        # item on the beam for each sentence
        inactive = mx.nd.zeros((batch_size * self.beam_size), dtype='int32', ctx=self.context)
        t = 1
        for t in range(1, max_output_length):
            # (1) obtain next predictions and advance models' state
            # target_dists: (batch_size * beam_size, target_vocab_size)
            target_dists, model_states = self._inference.decode_step(best_word_indices, model_states, vocab_slice_ids)

            # (2) Produces the accumulated cost of target words in each row.
            # There is special treatment for finished and inactive rows: inactive rows are inf everywhere;
            # finished rows are inf everywhere except column zero, which holds the accumulated model score
            scores = self._update_scores(target_dists, finished, inactive, scores_accumulated, pad_dist)

            # Mark entries that should be blocked as having a score of np.inf
            if self.global_avoid_trie or any(raw_avoid_list):
                block_indices = avoid_states.avoid()
                if len(block_indices) > 0:
                    scores[block_indices] = np.inf
                    if self.sample is not None:
                        target_dists[block_indices] = np.inf

            # (3) Get beam_size winning hypotheses for each sentence block separately. Only look as
            # far as the active beam size for each sentence.
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
            finished, lengths, estimated_reference_lengths = self._sort_by_index(best_hyp_indices,
                                                                                 finished,
                                                                                 lengths,
                                                                                 estimated_reference_lengths)

            # (5) Normalize the scores of newly finished hypotheses. Note that after this until the
            # next call to topk(), hypotheses may not be in sorted order.
            finished, scores_accumulated, lengths = self._norm_and_update_finished(best_word_indices,
                                                                                   max_output_lengths,
                                                                                   finished,
                                                                                   scores_accumulated,
                                                                                   lengths,
                                                                                   estimated_reference_lengths)

            # Collect best hypotheses, best word indices
            best_hyp_indices_list.append(best_hyp_indices)
            best_word_indices_list.append(best_word_indices)

            if self.beam_search_stop == C.BEAM_SEARCH_STOP_FIRST:
                at_least_one_finished = finished.reshape((batch_size, self.beam_size)).sum(axis=1) > 0
                if at_least_one_finished.sum().asscalar() == batch_size:
                    break
            else:
                if finished.sum().asscalar() == batch_size * self.beam_size:  # all finished
                    break

            # (9) update models' state with winning hypotheses (ascending)
            _sort_states(model_states, best_hyp_indices)

        logger.debug("Finished after %d / %d steps.", t + 1, max_output_length)

        # (9) Sort the hypotheses within each sentence (normalization for finished hyps may have unsorted them).
        folded_accumulated_scores = scores_accumulated.reshape((batch_size,
                                                                self.beam_size * scores_accumulated.shape[-1]))
        indices = mx.nd.cast(mx.nd.argsort(folded_accumulated_scores, axis=1), dtype='int32').reshape((-1,))
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


class SortByIndex(mx.gluon.HybridBlock):
    """
    A HybridBlock that sorts args by the given indices.
    """
    def hybrid_forward(self, F, indices, *args):
        return [F.take(arg, indices) for arg in args]


class UpdateScores(mx.gluon.HybridBlock):
    """
    A HybridBlock that updates the scores from the decoder step with accumulated scores.
    Inactive hypotheses receive score inf. Finished hypotheses receive their accumulated score for C.PAD_ID.
    All other options are set to infinity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class NormalizeAndUpdateFinished(mx.gluon.HybridBlock):
    """
    A HybridBlock for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self,
                 pad_id: int,
                 eos_id: int,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 brevity_penalty_weight: float = 0.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
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


class TopK(mx.gluon.HybridBlock):
    """
    Batch-wise topk operation.
    Forward method uses imperative shape inference, since both batch_size and vocab_size are dynamic
    during translation (due to variable batch size and potential vocabulary selection).
    """

    def __init__(self, k: int) -> None:
        """
        :param k: The number of smallest scores to return.
        """
        super().__init__()
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

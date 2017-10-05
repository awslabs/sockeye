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
Functions to generate loss symbols for sequence-to-sequence models.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import mxnet as mx
from mxnet.metric import EvalMetric

from . import config
from . import constants as C
from . import utils


class LossConfig(config.Config):
    """
    Loss configuration.

    :param name: Loss name.
    :param vocab_size: Target vocab size.
    :param normalize: Whether to normalize loss value.
    :param smoothed_cross_entropy_alpha: Smoothing value for smoothed-cross-entropy loss.
    """
    def __init__(self,
                 name: str,
                 vocab_size: int,
                 normalize: bool,
                 smoothed_cross_entropy_alpha: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalize = normalize
        self.smoothed_cross_entropy_alpha = smoothed_cross_entropy_alpha


def get_loss(loss_config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param loss_config: Loss configuration.
    """
    if loss_config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(loss_config)
    elif loss_config.name == C.SMOOTHED_CROSS_ENTROPY:
        return SmoothedCrossEntropyLoss(loss_config)
    else:
        raise ValueError("unknown loss name: %s" % loss_config.name)


def _normalize(loss: mx.sym.Symbol, labels: mx.sym.Symbol):
    """
    Normalize loss by the number of non-PAD tokens.

    :param loss: A loss value for each label.
    :param labels: A label for each loss entry (potentially containing PAD tokens).
    :return: The normalized loss.
    """
    return mx.sym.broadcast_div(loss, mx.sym.sum(labels != C.PAD_ID))


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> EvalMetric:
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        self.loss_config = loss_config

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self.loss_config.normalize:
            normalization = "valid"
        else:
            normalization = "null"
        return [mx.sym.SoftmaxOutput(data=logits,
                                    label=labels,
                                    ignore_label=C.PAD_ID,
                                    use_ignore=True,
                                    normalization=normalization,
                                    name=C.SOFTMAX_NAME)]


    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)


class CrossEntropyMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param output_labels: Name of labels that should be used when updating with update_dict.
    """
    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            label = label.as_in_context(pred.context).reshape((label.size,))
            prob = mx.nd.pick(pred, label.astype(dtype="int32"))
            # Ignore padding
            # TODO: contribute ignoring padding for cross-entropy back to MXNet
            ignore = (label == C.PAD_ID).astype(dtype=prob.dtype)
            prob = prob * (1 - ignore) + ignore
            # Sum, normalizing if needed
            loss = -mx.nd.log(prob + 1e-8)
            if self.loss_config.normalize:
                loss = loss / mx.nd.sum(1 - ignore)
                self.num_inst += 1
            else:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size
            self.sum_metric += mx.nd.sum(loss).asscalar()


class SmoothedCrossEntropyLoss(Loss):
    """
    Computes a smoothed cross-entropy loss. Smoothing is defined by alpha which indicates the
    amount of probability mass subtracted from the true label probability (1-alpha).
    Alpha is then uniformly distributed across other labels.
    If normalize is True, the gradient by dividing by the number of non-PAD tokens.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        self.loss_config = loss_config
        utils.check_condition(self.loss_config.smoothed_cross_entropy_alpha >= 0,
                              "alpha for smoothed loss must be >= 0")

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        probs = mx.sym.softmax(data=logits)

        on_value = 1.0 - self.loss_config.smoothed_cross_entropy_alpha
        off_value = self.loss_config.smoothed_cross_entropy_alpha / (self.loss_config.vocab_size - 1.0)
        cross_entropy = mx.sym.one_hot(indices=mx.sym.cast(data=labels, dtype='int32'),
                                       depth=self.loss_config.vocab_size,
                                       on_value=on_value,
                                       off_value=off_value)

        # zero out pad symbols (0)
        cross_entropy = mx.sym.where(labels, cross_entropy, mx.sym.zeros((0, self.loss_config.vocab_size)))

        # compute cross_entropy
        cross_entropy = cross_entropy * - mx.sym.log(data=probs + 1e-10)
        cross_entropy = mx.sym.sum(data=cross_entropy, axis=1)

        if self.loss_config.normalize:
            cross_entropy = _normalize(cross_entropy, labels)

        cross_entropy = mx.sym.MakeLoss(cross_entropy, name=C.SMOOTHED_CROSS_ENTROPY)
        probs = mx.sym.BlockGrad(probs, name=C.SOFTMAX_NAME)
        return [cross_entropy, probs]

    def create_metric(self) -> "SmoothedCrossEntropyMetric":
        return SmoothedCrossEntropyMetric(self.loss_config)


class SmoothedCrossEntropyMetric(EvalMetric):
    """
    Metric wrapper for smoothed cross entropy loss.  Since SCE already returns loss values during
    training, this class simply sums the results.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param output_labels: Name of labels that should be used when updating with update_dict.
    """
    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.SMOOTHED_CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    def update(self, labels, preds):
        # Take cross entropy values only
        sces = preds[::2]
        for label, sce in zip(labels, sces):
            batch_size = label.shape[0]
            # SCE is pre-computed, so just sum
            self.sum_metric += sce.asnumpy().sum()
            # Only scale if loss is not already normalized
            if self.loss_config.normalize:
                self.num_inst += 1
            else:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size

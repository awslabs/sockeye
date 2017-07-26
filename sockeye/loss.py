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
from typing import List

import mxnet as mx

from . import config
from . import constants as C
from . import utils


class LossConfig(config.Config):
    """
    Loss configuration.

    :param type: Loss name.
    :param vocab_size: Target vocab size.
    :param normalize: Whether to normalize loss value.
    :param smoothed_cross_entropy_alpha: Smoothing value for smoothed-cross-entropy loss.
    """
    def __init__(self,
                 type: str,
                 vocab_size: int,
                 normalize: bool,
                 smoothed_cross_entropy_alpha: float = 0.0) -> None:
        super().__init__()
        self.type = type
        self.vocab_size = vocab_size
        self.normalize = normalize
        self.smoothed_cross_entropy_alpha = smoothed_cross_entropy_alpha


def get_loss(config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param config: Loss configuration.
    """
    if config.type == C.CROSS_ENTROPY:
        return CrossEntropyLoss(config.normalize)
    elif config.type == C.SMOOTHED_CROSS_ENTROPY:
        return SmoothedCrossEntropyLoss(config.smoothed_cross_entropy_alpha,
                                        config.vocab_size,
                                        config.normalize)
    else:
        raise ValueError("unknown loss name: %s" % config.type)


class Loss:
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


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param normalize: If True normalize the gradient by dividing by the number of non-PAD tokens.
    """

    def __init__(self, normalize: bool = False):
        self._normalize = normalize

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self._normalize:
            normalization = "valid"
        else:
            normalization = "null"
        return [mx.sym.SoftmaxOutput(data=logits,
                                    label=labels,
                                    ignore_label=C.PAD_ID,
                                    use_ignore=True,
                                    normalization=normalization,
                                    name=C.SOFTMAX_NAME)]


def _normalize(loss: mx.sym.Symbol, labels: mx.sym.Symbol):
    """
    Normalize loss by the number of non-PAD tokens.

    :param loss: A loss value for each label.
    :param labels: A label for each loss entry (potentially containing PAD tokens).
    :return: The normalized loss.
    """
    return mx.sym.broadcast_div(loss, mx.sym.sum(labels != C.PAD_ID))


class SmoothedCrossEntropyLoss(Loss):
    """
    Computes a smoothed cross-entropy loss. Smoothing is defined by alpha which indicates the
    amount of probability mass subtracted from the true label probability (1-alpha).
    Alpha is then uniformly distributed across other labels.

    :param alpha: Smoothing value.
    :param vocab_size: Size of the target vocabulary.
    :param normalize: If True normalize the gradient by dividing by the number of non-PAD tokens.
    """

    def __init__(self, alpha: float, vocab_size: int, normalize: bool = False):
        utils.check_condition(alpha >= 0, "alpha for smoothed loss must be >= 0")
        self._alpha = alpha
        self._vocab_size = vocab_size
        self._normalize = normalize

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        probs = mx.sym.softmax(data=logits)

        on_value = 1.0 - self._alpha
        off_value = self._alpha / (self._vocab_size - 1.0)
        cross_entropy = mx.sym.one_hot(indices=mx.sym.cast(data=labels, dtype='int32'),
                                       depth=self._vocab_size,
                                       on_value=on_value,
                                       off_value=off_value)

        # zero out pad symbols (0)
        cross_entropy = mx.sym.where(labels, cross_entropy, mx.sym.zeros((0, self._vocab_size)))

        # compute cross_entropy
        cross_entropy *= - mx.sym.log(data=probs + 1e-10)
        cross_entropy = mx.sym.sum(data=cross_entropy, axis=1)

        if self._normalize:
            cross_entropy = _normalize(cross_entropy, labels)

        cross_entropy = mx.sym.MakeLoss(cross_entropy, name=C.SMOOTHED_CROSS_ENTROPY)
        probs = mx.sym.BlockGrad(probs, name=C.SOFTMAX_NAME)
        return [cross_entropy, probs]

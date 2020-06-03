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
Functions to generate loss symbols for sequence-to-sequence models.
"""
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict

import mxnet as mx

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


class Loss(mx.gluon.HybridBlock):
    """
    Generic Loss interface.
    A loss has a name, a configuration, and stores information about the output and label it requires from the model(s),
    as well as a weight (default 1.0) and a method to create the corresponding metric.
    """

    def __init__(self,
                 name: str,
                 output_name: str,
                 label_name: str,
                 weight: float = 1.0) -> None:
        super().__init__(prefix=name)
        self._output_name = output_name
        self._label_name = label_name
        self._weight = weight
        self._metric = None
        logger.info("Loss: %s | weight=%.2f | metric: %s | output_name: '%s' | label_name: '%s'",
                    self.prefix, self.weight, self.metric.name, self.output_name, self.label_name)

    def forward(self, outputs: Dict[str, Any], labels: Dict[str, Any]):
        """
        Loss retrieves the required output and label.
        """
        utils.check_condition(self.output_name in outputs,
                              "output '%s' not found. Loss requires this output key" % self.output_name)
        utils.check_condition(self.label_name in labels,
                              "label '%s' not found. Loss requires this label key" % self.output_name)
        output = outputs[self.output_name]
        label = labels[self.label_name]
        return super().forward(output.astype(label, copy=False), label)

    def hybrid_forward(self, F, outputs, labels):
        """
        Given outputs and labels, the loss returns two scalars: the loss value and a normalizer for that loss value.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        raise NotImplementedError()

    @property
    def metric(self):
        if self._metric is None:
            self._metric = self.create_metric()
        return self._metric

    @property
    def weight(self):
        return self._weight

    @property
    def output_name(self):
        return self._output_name

    @property
    def label_name(self):
        return self._label_name


class LossMetric(ABC):
    def __init__(self, name: str) -> None:
        self._name = name
        self._sum = 0.0
        self._num_inst = 0.0

    def __repr__(self):
        return "%s(%.2f/%.2f=%.2f)" % (self.name, self._sum, self._num_inst, self.get())

    def __str__(self):
        return "%s=%f" % (self.name, self.get())

    @property
    def name(self):
        return self._name

    def update(self, loss, num_samples):
        self._sum += loss
        self._num_inst += num_samples

    def get(self) -> float:
        return self._sum / self._num_inst if self._num_inst else float('nan')

    def reset(self):
        self._sum = 0.0
        self._num_inst = 0.0


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.
    Uses F.SoftmaxOutput to efficiently backpropagate cross-entropy gradients and do label smoothing.
    """

    def __init__(self,
                 name: str = C.CROSS_ENTROPY,
                 weight: float = 1.0,
                 label_smoothing: float = 0.0,
                 dtype: str = C.DTYPE_FP32,
                 output_name: str = C.LOGITS_NAME,
                 label_name: str = C.TARGET_LABEL_NAME,
                 ignore_label: int = C.PAD_ID) -> None:
        super().__init__(name=name, output_name=output_name, label_name=label_name, weight=weight)
        self.ignore_label = ignore_label
        self._alpha = label_smoothing
        self._normalization = "valid"
        self._dtype = dtype

    def hybrid_forward(self, F, logits, labels):
        """
        Returns unnormalized cross-entropy loss of the batch.

        :param F: MXNet API namespace.
        :param logits: Logits. Shape: (batch_size, sequence_length, output_dim).
        :param labels: Sparse labels. Shape: (batch_size, sequence_length)
        :return: Cross-entropy loss (1,), and number of valid tokens for normalization.
        """
        # computes softmax over the last axis, backpropagates ce gradients. Shape: (batch, len, vocab)
        softmax_out = F.SoftmaxOutput(data=logits,
                                      label=labels,
                                      ignore_label=self.ignore_label,
                                      use_ignore=True,
                                      normalization=self._normalization,
                                      smooth_alpha=self._alpha,
                                      # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                                      grad_scale=self.weight,
                                      preserve_shape=True)
        # (batch, len)
        pred = F.log(F.pick(F.BlockGrad(softmax_out), labels, axis=-1, keepdims=False))
        # (batch, len,)
        valid_mask = labels != self.ignore_label
        # (batch, len)
        pred = pred * valid_mask
        # (1,)
        ce = -F.sum(pred)
        return ce, F.sum(valid_mask)

    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        return PerplexityMetric()


class CrossEntropyLossWithoutSoftmaxOutput(Loss):
    """ no label smoothing supported """

    def __init__(self,
                 name: str = C.CROSS_ENTROPY,
                 weight: float = 1.0,
                 label_smoothing: float = 0.0,
                 dtype: str = C.DTYPE_FP32,
                 output_name: str = C.LOGITS_NAME,
                 label_name: str = C.TARGET_LABEL_NAME,
                 ignore_label: int = C.PAD_ID) -> None:
        super().__init__(name=name, output_name=output_name, label_name=label_name, weight=weight)
        self.ls = None
        if label_smoothing > 0.0:
            with self.name_scope():
                self.ls = LabelSmoothing(epsilon=label_smoothing, units=8230)  # TODO
        self.ignore_label = ignore_label
        self._alpha = label_smoothing
        self._dtype = dtype

    def hybrid_forward(self, F, logits, labels):
        pred = F.log_softmax(logits, axis=-1)

        if self.ls is None:
            # (batch, len)
            loss = -F.pick(pred, labels, axis=-1, keepdims=False)
        else:
            loss = -F.sum(pred * self.ls(labels), axis=-1, keepdims=False)

        # (batch, len,)
        valid_mask = labels != self.ignore_label

        # (batch, len)
        loss = loss * valid_mask

        # (1,)
        ce = F.sum(loss) * self.weight
        return ce, F.sum(valid_mask)

    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        return PerplexityMetric()


class LabelSmoothing(mx.gluon.HybridBlock):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Parameters
    ----------
    axis : int, default -1
        The axis to smooth.
    epsilon : float, default 0.1
        The epsilon parameter in label smoothing
    sparse_label : bool, default True
        Whether input is an integer array instead of one hot array.
    units : int or None
        Vocabulary size. If units is not given, it will be inferred from the input.
    prefix : str or None
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, axis=-1, epsilon=0.1, units=None,
                 sparse_label=True, prefix=None, params=None):
        super(LabelSmoothing, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._epsilon = epsilon
        self._sparse_label = sparse_label
        self._units = units

    def hybrid_forward(self, F, inputs, units=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        inputs : Symbol or NDArray
            Shape (batch_size, length) or (batch_size, length, V)
        units : int or None
        Returns
        -------
        smoothed_label : Symbol or NDArray
            Shape (batch_size, length, V)
        """
        if self._sparse_label:
            assert units is not None or self._units is not None, \
                'units needs to be given in function call or ' \
                'instance initialization when sparse_label is False'
            if units is None:
                units = self._units
            inputs = F.one_hot(inputs, depth=units)
        if units is None and self._units is None:
            return F.Custom(inputs, epsilon=self._epsilon, axis=self._axis,
                            op_type='_smoothing_with_dim')
        else:
            if units is None:
                units = self._units
            return ((1 - self._epsilon) * inputs) + (self._epsilon / units)


class PerplexityMetric(LossMetric):

    def __init__(self, name=C.PERPLEXITY):
        super().__init__(name=name)

    def update(self, batch_cross_entropy: float, batch_num_valid: float):
        self._sum += batch_cross_entropy
        self._num_inst += batch_num_valid

    def get(self):
        return math.exp(super().get())


class PoissonLoss(Loss):
    """
    Computes the Poisson regression loss.
    MSEMetric for this loss will be reporting the mean
    square error between lengths, not length ratios!
    """

    def __init__(self,
                 name: str = C.LENRATIO_NAME + "_" + C.LINK_POISSON,
                 weight: float = 1.0,
                 output_name: str = C.LENRATIO_NAME,
                 label_name: str = C.LENRATIO_LABEL_NAME) -> None:
        super().__init__(name=name, output_name=output_name, label_name=label_name, weight=weight)

    def hybrid_forward(self, F, length_predictions, labels):
        """
        Returns Poisson loss and output symbol given data and expected integers as labels.

        :param length_predictions: Length predictions. Shape: (batch_size,).
        :param labels: Targets. Shape: (batch_size,).
        :return: Poisson loss of length predictions of the batch, and number of samples (batch size).
        """
        # (batch_size,)
        loss = length_predictions - labels * F.log(F.maximum(1e-10, length_predictions))
        # (1,)
        loss = F.sum(loss * self.weight)
        num_samples = F.sum(F.ones_like(length_predictions))
        return loss, num_samples

    def create_metric(self) -> 'LossMetric':
        return LossMetric(name=C.LENRATIO_MSE)


class MSELoss(Loss):
    """
    Computes the Mean Squared Error loss.
    MSEMetric for this loss will be reporting the mean square error between length ratios.
    """

    def __init__(self,
                 name: str = C.LENRATIO_NAME + "_" + C.LINK_NORMAL,
                 weight: float = 1.0,
                 output_name: str = C.LENRATIO_NAME,
                 label_name: str = C.LENRATIO_LABEL_NAME) -> None:
        super().__init__(name=name, output_name=output_name, label_name=label_name, weight=weight)

    def hybrid_forward(self, F, length_predictions, labels):
        """
        Returns MSE loss.

        :param length_predictions: Length predictions. Shape: (batch_size,).
        :param labels: Targets. Shape: (batch_size,).
        :return: MSE loss of length predictions of the batch.
        """
        # (batch_size,)
        loss = (self.weight / 2) * F.square(length_predictions - labels)
        # (1,)
        loss = F.sum(loss)
        num_samples = F.sum(F.ones_like(length_predictions))
        return loss, num_samples

    def create_metric(self) -> 'LossMetric':
        return LossMetric(name=C.LENRATIO_MSE)

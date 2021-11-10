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

import logging
from typing import Any, Dict, Optional, Tuple
from abc import ABC, abstractmethod

import torch as pt
import math

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


class Loss(pt.nn.Module):
    """
    Generic Loss interface.
    A loss has a name, a configuration, and stores information about the output and label it requires from the model(s),
    as well as a weight (default 1.0) and a method to create the corresponding metric.
    """

    def __init__(self,
                 name: str,
                 output_name: str,
                 label_name: str,
                 weight: float = 1.0,
                 metric_prefix: str = '') -> None:
        super().__init__()
        self._name = name
        self._output_name = output_name
        self._label_name = label_name
        self._weight = weight
        self._metric = None  # type: Optional[LossMetric]
        self._metric_prefix = metric_prefix
        logger.info("Loss: %s | weight=%.2f | metric: %s (%s) | output_name: '%s' | label_name: '%s'",
                    self._name, self.weight, self.metric.name, self.metric.short_name,
                    self.output_name, self.label_name)

    def __call__(self, outputs: Dict[str, Any], labels: Dict[str, Any]):
        """
        Loss retrieves the required output and label.
        """
        utils.check_condition(self.output_name in outputs,
                              "output '%s' not found. Loss requires this output key" % self.output_name)
        utils.check_condition(self.label_name in labels,
                              "label '%s' not found. Loss requires this label key" % self.output_name)
        output = outputs[self.output_name]
        label = labels[self.label_name]

        return super().__call__(output, label)

    @abstractmethod
    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        raise NotImplementedError()

    @property
    def metric(self) -> 'LossMetric':
        if self._metric is None:
            self._metric = self.create_metric()
        return self._metric

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_name(self) -> str:
        return self._output_name

    @property
    def label_name(self) -> str:
        return self._label_name


class LossMetric(ABC):

    def __init__(self, name: str, short_name: Optional[str] = None, prefix: str = '') -> None:
        self._name = prefix + name
        self._short_name = prefix + short_name if short_name else self._name
        self._sum = 0.0
        self._num_inst = 0.0

    def __repr__(self):
        return "%s(%.2f/%.2f=%.2f)" % (self.name, self._sum, self._num_inst, self.get())

    def __str__(self):
        return "%s=%f" % (self.short_name, self.get())

    @property
    def name(self):
        return self._name

    @property
    def short_name(self) -> str:
        return self._short_name

    def update(self, loss, num_samples):
        self._sum += loss
        self._num_inst += num_samples

    def get(self) -> float:
        return self._sum / self._num_inst if self._num_inst else float('nan')

    def reset(self):
        self._sum = 0.0
        self._num_inst = 0.0


# TODO(fhieber): should be scriptable/traceable
class PyTorchCrossEntropyLoss(Loss):
    """
    Computes a cross-entropy loss, normalized by the number of valid (non-pad) tokens.
    Uses an efficient implementation for label smoothing and avoids the obscure SoftmaxOutput op.
    """

    def __init__(self,
                 name: str = C.CROSS_ENTROPY,
                 weight: float = 1.0,
                 label_smoothing: float = 0.0,
                 dtype: str = C.DTYPE_FP32,
                 output_name: str = C.LOGITS_NAME,
                 label_name: str = C.TARGET_LABEL_NAME,
                 ignore_label: int = C.PAD_ID,
                 metric_prefix: str = '',
                 label_smoothing_impl: str = 'mxnet') -> None:
        super().__init__(name=name, output_name=output_name, label_name=label_name,
                         weight=weight, metric_prefix=metric_prefix)
        self.ignore_label = ignore_label
        self._alpha = label_smoothing
        self._dtype = dtype
        self._reduction = 'mean'  # TODO: consider sum reduction and normalization outside of loss for reporting
        if label_smoothing == 0 or label_smoothing_impl == 'torch':
            self._ce_impl = self._torch_cross_entropy_loss
        elif label_smoothing > 0.0 and label_smoothing_impl == 'mxnet':
            self._ce_impl = self._smoothed_loss_as_in_mxnet
        elif label_smoothing > 0.0 and label_smoothing_impl == 'fairseq':
            self._ce_impl = self._smoothed_loss_as_in_fairseq
        else:
            raise ValueError("unknown label_smoothing impl. choose from mxnet, fairseq, or torch.")

    def _smoothed_loss_as_in_mxnet(self, logits, labels):
        """
        Computes label-smoothed cross-entropy loss just like sockeye.loss.CrossEntropyLossWithoutSoftmaxOutput()
        Notable details:
        - smoothing with 1/vocab_size, not 1/(vocab_size-1) as in fairseq
        - form taken from https://github.com/dmlc/gluon-nlp/blob/b714eaccc67619d7bdcbd1574d30be87d9c73f0c/src/gluonnlp/loss.py#L4
        """
        pred = pt.log_softmax(logits, dim=-1)
        nll = -pred.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)
        all_scores = pred.sum(dim=-1)
        # (batch, len,)
        valid_mask = labels.not_equal(self.ignore_label)
        pad_mask = ~valid_mask
        nll.masked_fill_(pad_mask, 0.0)
        all_scores.masked_fill_(pad_mask, 0.0)

        nll = (1 - self._alpha) * nll - self._alpha / logits.size(-1) * all_scores
        num_valid = valid_mask.sum()
        ce = nll.sum() * self.weight / num_valid
        return ce

    def _smoothed_loss_as_in_fairseq(self, logits, labels):
        """
        Computes smoothed NLL as in fairseq, see
        # https://github.com/pytorch/fairseq/blob/db0175a882e8ae0f30d89b5a610373dbe032d528/fairseq/criterions/label_smoothed_cross_entropy.py#L33
        """
        pred = pt.log_softmax(logits, dim=-1)
        if labels.dim() == logits.dim() - 1:
            labels = labels.unsqueeze(-1)
        nll = -pred.gather(dim=-1, index=labels.long())
        smooth_loss = pred.sum(dim=-1, keepdim=True)

        pad_mask = labels.eq(self.ignore_label)
        nll.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

        nll = nll.sum()
        smooth_loss = smooth_loss.sum()

        alpha_i = self._alpha / (logits.size(-1) - 1)
        nll = (1.0 - self._alpha - alpha_i) * nll - alpha_i * smooth_loss

        num_valid = (~pad_mask).sum()
        ce = nll.sum() * self.weight / num_valid
        return ce

    def _torch_cross_entropy_loss(self, logits, labels):
        logits = logits.view(-1, logits.size()[-1])
        # Reshape due to: view size is not compatible with input tensor's size and stride
        # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        labels = labels.reshape(-1)
        _kwargs = {'weight': None, 'ignore_index': self.ignore_label, 'reduction': self._reduction}
        if self._alpha > 0.0:
            _kwargs['label_smoothing'] = self._alpha
        ce = pt.nn.functional.cross_entropy(logits, labels.long(), **_kwargs)
        ce *= self.weight
        return ce

    def forward(self, logits: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        ce = self._ce_impl(logits, labels)
        return ce, pt.ones(1, device=ce.device)

    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        return PerplexityMetric(prefix=self._metric_prefix)


class PerplexityMetric(LossMetric):

    def __init__(self, prefix: str = '', name: str = C.PERPLEXITY, short_name: str = C.PERPLEXITY_SHORT_NAME) -> None:
        super().__init__(prefix=prefix, name=name, short_name=short_name)

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
                 name: str = f'{C.LENRATIO_NAME}_{C.LINK_POISSON}',
                 weight: float = 1.0,
                 output_name: str = C.LENRATIO_NAME,
                 label_name: str = C.LENRATIO_LABEL_NAME) -> None:
        super().__init__(name=name, output_name=output_name, label_name=label_name, weight=weight)

    def forward(self, length_predictions: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Returns Poisson loss and output given data and expected integers as labels.

        :param length_predictions: Length predictions. Shape: (batch_size,).
        :param labels: Targets. Shape: (batch_size,).
        :return: Poisson loss of length predictions of the batch, and number of samples (batch size).
        """
        # (batch_size,)
        loss = length_predictions - labels * pt.log(pt.clamp(length_predictions, min=1e-10))
        # (1,)
        loss = (loss * self.weight).sum()
        num_samples = pt.ones_like(length_predictions).sum()
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

    def forward(self, length_predictions: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Returns MSE loss.

        :param length_predictions: Length predictions. Shape: (batch_size,).
        :param labels: Targets. Shape: (batch_size,).
        :return: MSE loss of length predictions of the batch.
        """
        # (batch_size,)
        loss = (self.weight / 2) * pt.square(length_predictions - labels)
        # (1,)
        loss = loss.sum()
        num_samples = pt.ones_like(length_predictions).sum()
        return loss, num_samples

    def create_metric(self) -> 'LossMetric':
        return LossMetric(name=C.LENRATIO_MSE)


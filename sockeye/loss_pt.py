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
import torch as pt
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from mxnet import gluon, np, npx
from sockeye.loss import LossMetric

from . import constants as C
# TODO: consider inheriting from 'from torch.nn.modules.loss import _Loss' ?
from . import utils

logger = logging.getLogger(__name__)


class PyTorchLoss(pt.nn.Module):
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

        return super().__call__(output.to(dtype=label.dtype), label)

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


class PyTorchCrossEntropyLoss(PyTorchLoss):
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
                 num_labels: int = 0,
                 metric_prefix: str = '') -> None:  # this is needed for label smoothing
        super().__init__(name=name, output_name=output_name, label_name=label_name,
                         weight=weight, metric_prefix=metric_prefix)
        self.ignore_label = ignore_label
        self._alpha = label_smoothing
        self._dtype = dtype
        self._num_labels = float(num_labels)

    def forward(self, logits: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        if self._alpha == 0.0:
            #pt.nn.functional.cross_entropy(logits_pt, labels.long(), weight=None, ignore_index=0, reduction='mean')
            pt.nn.functional.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        pred = npx.log_softmax(logits, axis=-1)
        pt.nn.NLLLoss
        pt.nn.CrossEntropyLoss

        # (batch, len)
        neg_log_likelihood = - npx.pick(pred,  # pylint: disable=invalid-unary-operand-type
                                        labels, axis=-1, keepdims=False)

        # label smoothing as in
        # https://github.com/dmlc/gluon-nlp/blob/b714eaccc67619d7bdcbd1574d30be87d9c73f0c/src/gluonnlp/loss.py#L4
        if self._alpha > 0:
            all_scores = np.sum(pred, axis=-1)
            neg_log_likelihood = (1 - self._alpha) * neg_log_likelihood - self._alpha / self._num_labels * all_scores

        # (batch, len,)
        valid_mask = labels != self.ignore_label

        # (batch, len)
        loss = neg_log_likelihood * valid_mask

        # (1,)
        num_valid = np.sum(valid_mask)

        # (1,)
        ce = np.sum(loss) * self.weight

        # we need to divide by num_valid here to backpropagate a 'valid' normalized loss value like in SoftmaxOutput.
        return ce / num_valid, np.ones((1,))

    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        return PerplexityMetric(prefix=self._metric_prefix)
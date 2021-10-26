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

import math

import torch as pt
import mxnet as mx
import pytest
from mxnet import np

import sockeye.constants as C
import sockeye.loss
import sockeye.loss_pt
import sockeye.model
import sockeye.utils


# Dummy loss for testing
class DummyLoss(sockeye.loss.Loss):
    def forward(self, outputs, labels):
        return (outputs + labels) * self.weight

    def create_metric(self):
        return sockeye.loss.LossMetric('test_metric')


def test_loss_block():
    b = DummyLoss(name='test', output_name='output', label_name='label', weight=2.0)
    b.initialize()
    assert b.name == 'test'
    assert b.output_name == 'output'
    assert b.label_name == 'label'
    assert b.weight == 2.0

    # check required outputs/labels not found
    with pytest.raises(sockeye.utils.SockeyeError) as _:
        b({'unknown_output': np.zeros((1,))}, {'label': np.zeros((1,))})
    with pytest.raises(sockeye.utils.SockeyeError) as _:
        b({'output': np.zeros((1,))}, {'unknown_label': np.zeros((1,))})

    metric = b.create_metric()
    assert isinstance(metric, sockeye.loss.LossMetric)
    assert metric.name == 'test_metric'

    loss_out = b({'output': np.ones((1,))}, {'label': np.ones((1,))}).item()
    assert loss_out == 4.0


def test_loss_metric():
    metric = sockeye.loss.LossMetric(name='metric')
    assert metric.name == 'metric'
    assert np.isnan(metric.get())
    metric.update(loss=2, num_samples=2)
    assert metric.get() == 1.0
    metric.update(loss=2, num_samples=6)
    assert metric.get() == 0.5
    metric.reset()
    assert np.isnan(metric.get())


def test_cross_entropy_loss_without_softmax_output():
    b = sockeye.loss.CrossEntropyLossWithoutSoftmaxOutput(ignore_label=C.PAD_ID, label_smoothing=0.0, num_labels=4)
    b.initialize()
    assert b.ignore_label == C.PAD_ID
    assert b.name == C.CROSS_ENTROPY
    assert b.weight == 1.0
    assert b._dtype == C.DTYPE_FP32
    assert b.output_name == C.LOGITS_NAME
    assert b.label_name == C.TARGET_LABEL_NAME
    assert b._alpha == 0.0

    logits = np.array([[1, 1, 1, 1],
                       [4, 2, 2, 2],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]])
    logits.attach_grad()
    labels = np.array([1, 0, 2, 3])
    labels.attach_grad()

    with mx.autograd.record():
        loss_value, loss_samples = b({C.LOGITS_NAME: logits, 'other_stuff': None},
                                     {C.TARGET_LABEL_NAME: labels, 'other_stuff': None})
    loss_value.backward()
    assert loss_samples.item() == 1  # this loss returns always 1

    expected_logits_grad = [[0.08333334, -0.25,        0.08333334,  0.08333334],
                            [0.,          0.,          0.,          0.],
                            [0.08333334,  0.08333334, -0.25,        0.08333334],
                            [0.08333334,  0.08333334,  0.08333334, -0.25]]
    num_valid = (C.PAD_ID != labels).sum().item()
    expected_loss_value = -(math.log(1/4) * 3) / num_valid  # 3 valid rows, all uniform, divided by num_valid

    assert np.isclose(loss_value.item(), expected_loss_value)
    assert np.allclose(logits.grad, expected_logits_grad)
    assert labels.grad.sum().item() == 0


@pytest.mark.parametrize("logits_mx, labels_mx, weight, alpha",
                         [(np.array([[1, 1, 1, 1],
                                     [4, 2, 2, 2],
                                     [1, 1, 1, 1],
                                     [1, 1, 1, 1]]),
                           np.array([1, 0, 2, 3]),
                           1, 0.0),
                          (np.array([[1, 1, 1, 1],
                                     [4, 2, 2, 2],
                                     [1, 1, 1, 1],
                                     [1, 1, 1, 1]]),
                           np.array([1, 0, 2, 3]),
                           1, 0.2),
                          (np.random.uniform(0, 1, (10, 32)),
                           np.random.randint(0, 32, (10,)),
                           2, 0.0),
                          (np.random.uniform(0, 5, (2, 10, 16)),
                           np.random.randint(0, 16, (2, 10)),
                           0.5, 0.1),
                          (np.random.uniform(0, 5, (8, 10, 128)),
                           np.random.randint(0, 128, (8, 10)),
                           0.5, 0.5)
                          ])
def test_mx_pt_eq_cross_entropy_loss(logits_mx, labels_mx, weight, alpha):
    logits_mx.attach_grad()
    labels_mx = labels_mx.astype('float32')
    logits_pt = pt.tensor(logits_mx.asnumpy(), requires_grad=True)
    labels_pt = pt.tensor(labels_mx.asnumpy())
    num_labels = logits_mx.shape[-1]

    loss_mx = sockeye.loss.CrossEntropyLossWithoutSoftmaxOutput(ignore_label=C.PAD_ID, label_smoothing=alpha,
                                                                num_labels=num_labels, weight=weight)
    loss_mx.initialize()
    loss_pt = sockeye.loss_pt.PyTorchCrossEntropyLoss(ignore_label=C.PAD_ID, label_smoothing=alpha, weight=weight)

    with mx.autograd.record():
        loss_value_mx, loss_samples_mx = loss_mx({C.LOGITS_NAME: logits_mx, 'other_stuff': None},
                                                 {C.TARGET_LABEL_NAME: labels_mx, 'other_stuff': None})
    loss_value_mx.backward()

    loss_value_pt, loss_samples_pt = loss_pt({C.LOGITS_NAME: logits_pt, 'other_stuff': None},
                                             {C.TARGET_LABEL_NAME: labels_pt, 'other_stuff': None})
    loss_value_pt.backward()

    assert loss_samples_mx.item() == loss_samples_pt.detach().numpy()
    assert np.allclose(logits_mx.grad.asnumpy(), logits_pt.grad.numpy())


def test_perplexity_metric():
    ppl = sockeye.loss.PerplexityMetric()
    assert ppl.name == C.PERPLEXITY
    ces = [2.0, 1.4, 5.2]
    for ce in ces:
        ppl.update(ce, 1)
    expected_ppl = math.exp(sum(ces) / len(ces))
    assert np.isclose(ppl.get(), expected_ppl)


@pytest.mark.parametrize("length_predictions_mx, labels_mx, weight, loss",
                         [(np.array([1, 1, 2, 2, 3, 6]),
                           np.array([1, 2, 3, 4, 5, 6]),
                           1.0, 'poisson'),
                          (np.array([1, 1, 2, 2, 7, 6]),
                           np.array([1, 2, 0, 4, 0, 6]),
                           0.5, 'poisson'),
                          (np.array([1, 1, 2, 2, 3, 6]),
                           np.array([1, 2, 3, 4, 5, 6]),
                           1.0, 'mse'),
                          (np.array([1, 1, 2, 2, 7, 6]),
                           np.array([1, 2, 0, 4, 0, 6]),
                           0.5, 'mse'),
                         ])
def test_mx_pt_eq_length_task_losses(length_predictions_mx, labels_mx, weight, loss):
    length_predictions_mx.attach_grad()
    length_predictions_pt = pt.tensor(length_predictions_mx.asnumpy(), requires_grad=True)
    labels_pt = pt.tensor(labels_mx.asnumpy())

    if loss == 'poisson':
        b_mx = sockeye.loss.PoissonLoss()
        b_pt = sockeye.loss_pt.PoissonLoss()
    elif loss == 'mse':
        b_mx = sockeye.loss.MSELoss()
        b_pt = sockeye.loss_pt.MSELoss()
    else:
        raise ValueError("unknown loss")

    b_mx.initialize()
    with mx.autograd.record():
        loss_value_mx, loss_samples_mx = b_mx({C.LENRATIO_NAME: length_predictions_mx, 'other_stuff': None},
                                              {C.LENRATIO_LABEL_NAME: labels_mx, 'other_stuff': None})
    loss_value_mx.backward()

    loss_value_pt, loss_samples_pt = b_pt({C.LENRATIO_NAME: length_predictions_pt, 'other_stuff': None},
                                          {C.LENRATIO_LABEL_NAME: labels_pt, 'other_stuff': None})
    loss_value_pt.backward()

    assert np.allclose(loss_value_mx.asnumpy(), loss_value_pt.detach().numpy())
    assert loss_samples_mx.item() == loss_samples_pt.detach().numpy()
    assert np.allclose(length_predictions_mx.grad.asnumpy(), length_predictions_pt.grad.numpy())




# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as onp
import pytest
import torch as pt

import sockeye.constants as C
import sockeye.loss
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
    assert b.name == 'test'
    assert b.output_name == 'output'
    assert b.label_name == 'label'
    assert b.weight == 2.0

    # check required outputs/labels not found
    with pytest.raises(sockeye.utils.SockeyeError) as _:
        b({'unknown_output': pt.zeros(1)}, {'label': pt.zeros(1)})
    with pytest.raises(sockeye.utils.SockeyeError) as _:
        b({'output': pt.zeros(1)}, {'unknown_label': pt.zeros(1)})

    metric = b.create_metric()
    assert isinstance(metric, sockeye.loss.LossMetric)
    assert metric.name == 'test_metric'

    loss_out = b({'output': pt.ones(1)}, {'label': pt.ones(1)}).item()
    assert loss_out == 4.0


def test_loss_metric():
    metric = sockeye.loss.LossMetric(name='metric')
    assert metric.name == 'metric'
    assert onp.isnan(metric.get())
    metric.update(loss=2, num_samples=2)
    assert metric.get() == 1.0
    metric.update(loss=2, num_samples=6)
    assert metric.get() == 0.5
    metric.reset()
    assert onp.isnan(metric.get())


def test_cross_entropy_loss():
    b = sockeye.loss.CrossEntropyLoss(ignore_label=C.PAD_ID, label_smoothing=0.0)
    assert b.ignore_label == C.PAD_ID
    assert b.name == C.CROSS_ENTROPY
    assert b.weight == 1.0
    assert b._dtype == C.DTYPE_FP32
    assert b.output_name == C.LOGITS_NAME
    assert b.label_name == C.TARGET_LABEL_NAME
    assert b._alpha == 0.0

    logits = pt.tensor([[1, 1, 1, 1],
                        [4, 2, 2, 2],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]], dtype=pt.float32, requires_grad=True)
    labels = pt.tensor([1, 0, 2, 3])

    loss_value, loss_samples = b({C.LOGITS_NAME: logits, 'other_stuff': None},
                                 {C.TARGET_LABEL_NAME: labels, 'other_stuff': None})
    loss_value.backward()
    assert loss_samples.item() == 1  # this loss returns always 1

    expected_logits_grad = pt.tensor([[0.08333334, -0.25, 0.08333334, 0.08333334],
                                      [0., 0., 0., 0.],
                                      [0.08333334, 0.08333334, -0.25, 0.08333334],
                                      [0.08333334, 0.08333334, 0.08333334, -0.25]])
    num_valid = (C.PAD_ID != labels).sum().item()
    expected_loss_value = pt.tensor(
        -(math.log(1 / 4) * 3) / num_valid)  # 3 valid rows, all uniform, divided by num_valid

    pt.testing.assert_close(loss_value, expected_loss_value)
    pt.testing.assert_close(logits.grad, expected_logits_grad)


def test_label_to_bow():
    labels = pt.tensor(
        [
            [1, 3],
            [0, 0],
        ]
    )
    bow = sockeye.loss._label_to_bow(labels, num_labels=4)
    expected_bow = pt.tensor([
        [0, 1, 0, 1],
        [1, 0, 0, 0],
    ], dtype=pt.float32)
    pt.testing.assert_close(bow, expected_bow)


def test_binary_cross_entropy_loss():
    vocab_size = 4
    b = sockeye.loss.BinaryCrossEntropyBowLoss(
        pos_weight=1,
        num_labels=vocab_size
    )
    assert b.name == C.BINARY_CROSS_ENTROPY
    assert b.weight == 1.0
    assert b._dtype == C.DTYPE_FP32
    assert b.output_name == C.NVS_PRED_NAME
    assert b.label_name == C.TARGET_LABEL_NAME

    # batch size x num vocab
    # 2 x 4
    # Only as single element will contribute to the loss
    # (as all other predicitons will match the labels so the loss will be ~0)
    logits = pt.tensor([[-100, 100, -100, 1],
                        [-100, 100, -100, 100]], dtype=pt.float32, requires_grad=True)

    # (batch_size, num_target_vocabs, num_vocab)
    # (2, 1, 4)
    labels = pt.tensor(
        [
            [1, 3],
            [1, 3],
        ]
    )
    batch_size = labels.shape[0]

    loss_value, loss_samples = b({C.NVS_PRED_NAME: logits, 'other_stuff': None},
                                 {C.TARGET_LABEL_NAME: labels, 'other_stuff': None})
    loss_value.backward()
    assert loss_samples.item() == 1  # this loss returns always 1
    expected_loss = -pt.log(pt.sigmoid(pt.tensor(1))) / vocab_size / batch_size
    pt.testing.assert_close(loss_value, expected_loss)
    expected_grad = - 1 / (pt.exp(pt.tensor(1)) + 1) / vocab_size / batch_size
    pt.testing.assert_close(logits.grad,
        pt.tensor([[0.0000, 0.0000, 0.0000, expected_grad],
                   [0.0000, 0.0000, 0.0000, 0.0000]])
    )



def test_perplexity_metric():
    ppl = sockeye.loss.PerplexityMetric()
    assert ppl.name == C.PERPLEXITY
    ces = [2.0, 1.4, 5.2]
    for ce in ces:
        ppl.update(ce, 1)
    expected_ppl = math.exp(sum(ces) / len(ces))
    assert onp.isclose(ppl.get(), expected_ppl)

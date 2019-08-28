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

import mxnet as mx
import numpy as np
import pytest

import sockeye.constants as C
import sockeye.loss
import sockeye.model
import sockeye.utils


# Dummy loss for testing
class DummyLoss(sockeye.loss.Loss):
    def hybrid_forward(self, F, outputs, labels):
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
        b({'unknown_output': mx.nd.zeros((1,))}, {'label': mx.nd.zeros((1,))})
    with pytest.raises(sockeye.utils.SockeyeError) as _:
        b({'output': mx.nd.zeros((1,))}, {'unknown_label': mx.nd.zeros((1,))})

    metric = b.create_metric()
    assert isinstance(metric, sockeye.loss.LossMetric)
    assert metric.name == 'test_metric'

    loss_out = b({'output': mx.nd.ones((1,))}, {'label': mx.nd.ones((1,))}).asscalar()
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


def test_cross_entropy_loss():
    b = sockeye.loss.CrossEntropyLoss()
    b.initialize()
    assert b.ignore_label == C.PAD_ID
    assert b.name == C.CROSS_ENTROPY
    assert b.weight == 1.0
    assert b._dtype == C.DTYPE_FP32
    assert b.output_name == C.LOGITS_NAME
    assert b.label_name == C.TARGET_LABEL_NAME
    assert b._alpha == 0.0

    logits = mx.nd.array([[1, 1, 1, 1],
                          [4, 2, 2, 2],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]])
    logits.attach_grad()
    labels = mx.nd.array([1, 0, 2, 3])
    labels.attach_grad()

    with mx.autograd.record():
        loss_value, loss_samples = b({C.LOGITS_NAME: logits, 'other_stuff': None},
                                     {C.TARGET_LABEL_NAME: labels, 'other_stuff': None})
    loss_value.backward()
    assert loss_samples.asscalar() == (C.PAD_ID != labels).sum().asscalar()

    expected_logits_grad = [[0.08333334, -0.25,        0.08333334,  0.08333334],
                            [0.,          0.,          0.,          0.],
                            [0.08333334,  0.08333334, -0.25,        0.08333334],
                            [0.08333334,  0.08333334,  0.08333334, -0.25]]
    expected_loss_value = -(math.log(1/4) * 3)  # 3 valid rows, all uniform

    assert np.isclose(loss_value.asscalar(), expected_loss_value)
    assert np.allclose(logits.grad.asnumpy(), expected_logits_grad)
    assert labels.grad.sum().asscalar() == 0


def test_perplexity_metric():
    ppl = sockeye.loss.PerplexityMetric()
    assert ppl.name == C.PERPLEXITY
    ces = [2.0, 1.4, 5.2]
    for ce in ces:
        ppl.update(ce, 1)
    expected_ppl = math.exp(sum(ces) / len(ces))
    assert np.isclose(ppl.get(), expected_ppl)


# TODO(fhieber): test to compare SoftmaxOutput and alternative cross entropy loss implementation

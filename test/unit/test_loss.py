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

import mxnet as mx
import numpy as np

import sockeye.constants as C
import sockeye.loss
import sockeye.model


def test_cross_entropy_loss():
    config = sockeye.loss.LossConfig(name=C.CROSS_ENTROPY, vocab_size=4, normalization_type=C.LOSS_NORM_BATCH)
    loss = sockeye.loss.get_loss(config)
    assert isinstance(loss, sockeye.loss.CrossEntropyLoss)

    logits = mx.sym.Variable("logits")
    labels = mx.sym.Variable("labels")
    sym = mx.sym.Group(loss.get_loss(logits, labels))

    assert sym.list_arguments() == ['logits', 'labels']
    assert sym.list_outputs() == [C.SOFTMAX_NAME + "_output"]

    logits_np = mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]])
    labels_np = mx.nd.array([1, 0, 2, 3])  # C.PAD_ID == 0

    expected_softmax = np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]])
    expected_grads = np.asarray([[0.0320586, -0.91285568, 0.23688284, 0.64391428],
                                 [0., 0., 0., 0.],
                                 [0.25, 0.25, -0.75, 0.25],
                                 [0.25, 0.25, 0.25, -0.75]])

    _, out_shapes, _ = (sym.infer_shape(logits=logits_np.shape, labels=labels_np.shape))
    assert out_shapes[0] == logits_np.shape

    executor = sym.simple_bind(ctx=mx.cpu(),
                               logits=logits_np.shape,
                               labels=labels_np.shape)
    executor.arg_dict["logits"][:] = logits_np
    executor.arg_dict["labels"][:] = labels_np
    softmax = executor.forward(is_train=True)[0].asnumpy()
    assert np.isclose(softmax, expected_softmax).all()

    executor.backward()
    grads = executor.grad_dict["logits"].asnumpy()
    assert np.isclose(grads, expected_grads).all()
    label_grad_sum = executor.grad_dict["labels"].asnumpy().sum()
    assert label_grad_sum == 0


def test_smoothed_cross_entropy_loss():
    config = sockeye.loss.LossConfig(name=C.CROSS_ENTROPY,
                                     vocab_size=4,
                                     normalization_type=C.LOSS_NORM_BATCH,
                                     label_smoothing=0.5)
    loss = sockeye.loss.get_loss(config)
    assert isinstance(loss, sockeye.loss.CrossEntropyLoss)

    logits = mx.sym.Variable("logits")
    labels = mx.sym.Variable("labels")
    sym = mx.sym.Group(loss.get_loss(logits, labels))

    assert sym.list_arguments() == ['logits', 'labels']
    assert sym.list_outputs() == [C.SOFTMAX_NAME + "_output"]

    logits_np = mx.nd.array([[1, 2, 3, 4],
                             [4, 2, 2, 2],
                             [3, 3, 3, 3],
                             [4, 4, 4, 4]])
    labels_np = mx.nd.array([1, 0, 2, 3])  # C.PAD_ID == 0

    expected_softmax = np.asarray([[0.0320586, 0.08714432, 0.23688284, 0.64391428],
                                   [0.71123451, 0.09625512, 0.09625512, 0.09625512],
                                   [0.25, 0.25, 0.25, 0.25],
                                   [0.25, 0.25, 0.25, 0.25]])
    expected_grads = np.asarray([[-0.13460806, -0.41285568, 0.07021617, 0.4772476],
                                 [0., 0., 0., 0.],
                                 [0.08333333, 0.08333333, -0.25, 0.08333333],
                                 [0.08333333, 0.08333333, 0.08333333, -0.25]])

    _, out_shapes, _ = (sym.infer_shape(logits=logits_np.shape, labels=labels_np.shape))
    assert out_shapes[0] == logits_np.shape

    executor = sym.simple_bind(ctx=mx.cpu(),
                               logits=logits_np.shape,
                               labels=labels_np.shape)
    executor.arg_dict["logits"][:] = logits_np
    executor.arg_dict["labels"][:] = labels_np
    outputs = executor.forward(is_train=True)
    softmax = outputs[0].asnumpy()
    assert np.isclose(softmax, expected_softmax).all()

    executor.backward()
    grads = executor.grad_dict["logits"].asnumpy()
    assert np.isclose(grads, expected_grads).all()
    label_grad_sum = executor.grad_dict["labels"].asnumpy().sum()
    assert label_grad_sum == 0

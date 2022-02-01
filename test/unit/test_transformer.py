# Copyright 2018--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
import torch as pt

import sockeye.constants as C
import sockeye.transformer


@pytest.mark.parametrize('use_glu', [(False), (True)])
def test_transformer_feed_forward(use_glu):
    block = sockeye.transformer.TransformerFeedForward(num_hidden=2,
                                                       num_model=2,
                                                       act_type=C.RELU,
                                                       dropout=0.1,
                                                       use_glu=use_glu)

    data = pt.ones(1, 10, 2)
    block(data)


@pytest.mark.parametrize('length', [1, 10, 100])
def test_pt_autoregressive_mask(length):
    x_pt = pt.zeros(2, length, 32)
    b_pt = sockeye.transformer.AutoRegressiveMask()
    result_pt = b_pt(x_pt).detach()

    assert result_pt.dtype == pt.bool
    assert result_pt.size() == (length, length)

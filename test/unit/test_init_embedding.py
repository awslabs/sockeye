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

import pytest

import numpy as np
import mxnet as mx

import sockeye.init_embedding as init_embedding


@pytest.mark.parametrize(
    "embed, vocab_in, vocab_out, expected_embed_init", [
        (np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
         {'w1': 0, 'w2': 1, 'w3': 2},
         {'w2': 0, 'w3': 1, 'w4': 2, 'w5': 3},
         mx.nd.array([[2, 2, 2], [3, 3, 3], [0, 0, 0], [0, 0, 0]]))
])
def test_init_weight(embed, vocab_in, vocab_out, expected_embed_init):
    embed_init = init_embedding.init_weight(embed, vocab_in, vocab_out)

    assert (embed_init == expected_embed_init).asnumpy().all()

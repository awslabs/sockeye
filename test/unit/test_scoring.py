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

import sockeye.scoring
from sockeye.beam_search import CandidateScorer

import mxnet as mx


def test_batch_scorer():
    # TODO: make this a useful test
    batch = 2
    seq = 4
    nh = 6
    logits = mx.nd.ones((batch, seq, nh))
    label = mx.nd.ones((batch, seq))
    length_ratio = mx.nd.ones((batch,))
    source_length = mx.nd.cast(mx.nd.random.randint(0, seq, (batch,)), 'float32')
    target_length = source_length
    b = sockeye.scoring.BatchScorer(scorer=CandidateScorer(1.0, 0.0, 0.0),
                                    score_type='neglogprob',
                                    constant_length_ratio=None)
    b.hybridize()
    scores = b(logits, label, length_ratio, source_length, target_length)
    assert scores.shape == (batch,)



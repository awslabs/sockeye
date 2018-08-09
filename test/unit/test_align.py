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

import pytest
import numpy as np

from sockeye import align
from sockeye.data_io import tokens2ids
from sockeye import constants as C

# Dummy vocab that maps strings to integers
vocab_size = 1000
vocab = {}
for i, symbol in enumerate(C.VOCAB_SYMBOLS):
    vocab[symbol] = i
bos_id = vocab[C.BOS_SYMBOL]
eos_id = vocab[C.EOS_SYMBOL]
for x in range(len(vocab), vocab_size):
    vocab[str(x)] = x

@pytest.mark.parametrize("src, trg, expected_labels",
                         [
                             ('12 13 14', '12 11 21', [vocab_size + 0, 11, 21, eos_id])
                         ])
def test_generate_pointer_labels(src, trg, expected_labels):
    aligner = align.Aligner(vocab, vocab)

    # TODO: create test fixtures
    source = tokens2ids(list(src.split()) + [eos_id], vocab, use_pointer_nets=False, max_oov_words=1,
                        point_nets_type=C.POINTER_NET_SUMMARY)
    target = tokens2ids([bos_id] + list(trg.split()), vocab, use_pointer_nets=False, max_oov_words=1,
                        point_nets_type=C.POINTER_NET_SUMMARY)
    labels = target[1:] + [eos_id]

    new_labels = aligner.get_labels(source, target, labels)

    assert new_labels == expected_labels

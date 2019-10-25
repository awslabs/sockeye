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

import os
from tempfile import TemporaryDirectory

import mxnet as mx
import numpy as np

import sockeye.constants as C
import sockeye.lexicon


def test_topk_lexicon():
    lexicon = ["a\ta\t-0.6931471805599453",
               "a\tb\t-1.2039728043259361",
               "a\tc\t-1.6094379124341003",
               "b\tb\t0.0"]
    vocab_list = ["a", "b", "c"]
    vocab = dict((y, x) for (x, y) in enumerate(C.VOCAB_SYMBOLS + vocab_list))
    k = 2
    lex = sockeye.lexicon.TopKLexicon(vocab, vocab)

    # Create from known lexicon
    with TemporaryDirectory(prefix="test_topk_lexicon.") as work_dir:
        # Write fast_align format lex table
        input_lex_path = os.path.join(work_dir, "input.lex")
        with open(input_lex_path, "w") as out:
            for line in lexicon:
                print(line, file=out)
        # Use fast_align lex table to build top-k lexicon
        lex.create(input_lex_path, k)

        # Test against known lexicon
        expected = mx.nd.zeros((len(C.VOCAB_SYMBOLS) + len(vocab_list), k), dtype='int32')
        # a -> special + a b
        expected[len(C.VOCAB_SYMBOLS), :2] = [len(C.VOCAB_SYMBOLS), len(C.VOCAB_SYMBOLS) + 1]
        # b -> special + b
        expected[len(C.VOCAB_SYMBOLS) + 1, :1] = [len(C.VOCAB_SYMBOLS) + 1]
        assert np.all(lex.lex.asnumpy() == expected.asnumpy())

        # Test save/load
        expected_sorted = mx.nd.sort(expected, axis=1)
        json_lex_path = os.path.join(work_dir, "lex.npy")
        lex.save(json_lex_path)
        lex.load(json_lex_path)
        assert np.all(lex.lex.asnumpy() == expected_sorted.asnumpy())

        # Test lookup
        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["a"], vocab["c"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS + ["a", "b"]], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["a"], vocab["a"], vocab["c"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS + ["a", "b"]], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["a"], vocab["a"], vocab["c"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS + ["a", "b"]], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["b"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS + ["b"]], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["c"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

        # Test load with smaller k
        small_k = k - 1
        lex.load(json_lex_path, k=small_k)
        assert lex.lex.shape[1] == small_k
        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["a"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS + ["a"]], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

        # Test load with larger k
        large_k = k + 1
        lex.load(json_lex_path, k=large_k)
        assert lex.lex.shape[1] == k
        trg_ids = lex.get_trg_ids(mx.nd.array([[vocab["a"], vocab["c"]]], dtype='int32'))
        expected = mx.nd.array([vocab[symbol] for symbol in C.VOCAB_SYMBOLS + ["a", "b"]], dtype='int32')
        assert np.all(trg_ids.asnumpy() == expected.asnumpy())

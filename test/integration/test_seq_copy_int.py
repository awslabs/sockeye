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

import os
from tempfile import TemporaryDirectory

import pytest

from test.common import generate_digits_file, run_train_translate

_TRAIN_LINE_COUNT = 100
_DEV_LINE_COUNT = 10
_LINE_MAX_LENGTH = 9

@pytest.mark.parametrize("train_params, translate_params", [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --rnn-num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 16 --num-embed 8 --attention-type mlp"
     " --attention-num-hidden 16 --batch-size 8 --loss cross-entropy --optimized-metric perplexity --max-updates 10"
     " --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2"),
    # "Kitchen sink" LSTM encoder-decoder with attention
    ("--encoder rnn --rnn-num-layers 4 --rnn-cell-type lstm --rnn-num-hidden 16 --rnn-residual-connections"
     " --num-embed 16 --attention-type coverage --attention-num-hidden 16 --weight-tying --attention-use-prev-word"
     " --context-gating --layer-normalization --batch-size 8 --loss smoothed-cross-entropy"
     " --smoothed-cross-entropy-alpha 0.1 --normalize-loss --optimized-metric perplexity --max-updates 10"
     " --checkpoint-frequency 10 --dropout 0.1 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2"),
    # Convolutional embedding encoder + LSTM encoder-decoder with attention
    ("--encoder rnn-with-conv-embed --conv-embed-max-filter-width 3 --conv-embed-num-filters 4 4 8"
     " --conv-embed-pool-stride 2 --conv-embed-num-highway-layers 1 --rnn-num-layers 1 --rnn-cell-type lstm"
     " --rnn-num-hidden 16 --num-embed 8 --attention-num-hidden 16 --batch-size 8 --loss cross-entropy"
     " --optimized-metric perplexity --max-updates 10 --checkpoint-frequency 10 --optimizer adam"
     " --initial-learning-rate 0.01",
     "--beam-size 2"),
])

def test_seq_copy(train_params, translate_params):
    """Task: copy short sequences of digits"""
    with TemporaryDirectory(prefix="test_seq_copy") as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        generate_digits_file(train_source_path, train_target_path, _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH)
        generate_digits_file(dev_source_path, dev_target_path, _DEV_LINE_COUNT, _LINE_MAX_LENGTH)
        # Test model configuration
        # Ignore return values (perplexity and BLEU) for integration test
        run_train_translate(train_params,
                            translate_params,
                            train_source_path,
                            train_target_path,
                            dev_source_path,
                            dev_target_path,
                            max_seq_len=_LINE_MAX_LENGTH + 1,
                            work_dir=work_dir)

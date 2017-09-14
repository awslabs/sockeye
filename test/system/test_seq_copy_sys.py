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

_TRAIN_LINE_COUNT = 10000
_DEV_LINE_COUNT = 100
_LINE_MAX_LENGTH = 9
_SEED_TRAIN = 13
_SEED_DEV = 17


@pytest.mark.parametrize("train_params, translate_params, perplexity_thresh, bleu_thresh", [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 16 --loss cross-entropy --optimized-metric perplexity"
     " --max-updates 10000 --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001"
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0 --max-updates 5000",
     "--beam-size 5",
     1.01,
     0.98),
    # 2-layer transformer encoder, LSTM decoder with attention
    ("--encoder transformer --num-layers 2:1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mhdot --rnn-attention-num-hidden 32 --batch-size 16 --rnn-attention-mhdot-heads 1"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 10000"
     " --transformer-attention-heads 4 --transformer-model-size 64"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.01,
     0.99),
    # LSTM encoder, 1-layer transformer decoder
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --decoder transformer --batch-size 16"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 3000"
     " --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.01,
     0.98),
    # 2-layer transformer
    ("--encoder transformer --decoder transformer"
     " --batch-size 16 --max-updates 3000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001"
     " --layer-normalization",
     "--beam-size 1",
     1.01,
     0.999),
    # 3-layer cnn
    ("--encoder cnn --decoder cnn "
     " --batch-size 16 --num-layers 3 --max-updates 3000"
     " --cnn-num-hidden 32 --cnn-positional-embedding-type fixed"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 1",
     1.01,
     0.999)
])
def test_seq_copy(train_params, translate_params, perplexity_thresh, bleu_thresh):
    """Task: copy short sequences of digits"""
    with TemporaryDirectory(prefix="test_seq_copy.") as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        generate_digits_file(train_source_path, train_target_path, _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH, seed=_SEED_TRAIN)
        generate_digits_file(dev_source_path, dev_target_path, _DEV_LINE_COUNT, _LINE_MAX_LENGTH, seed=_SEED_DEV)
        # Test model configuration
        perplexity, bleu = run_train_translate(train_params,
                                               translate_params,
                                               train_source_path,
                                               train_target_path,
                                               dev_source_path,
                                               dev_target_path,
                                               max_seq_len=_LINE_MAX_LENGTH + 1,
                                               work_dir=work_dir)

        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh


@pytest.mark.parametrize("train_params, translate_params, perplexity_thresh, bleu_thresh", [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 16 --loss cross-entropy --optimized-metric perplexity "
     "--max-updates 10000 --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.03,
     0.98),
    # 1-layer transformer encoder, LSTM decoder with attention
    ("--encoder transformer --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mhdot --rnn-attention-num-hidden 32 --batch-size 16 --rnn-attention-mhdot-heads 2"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 8000"
     " --transformer-attention-heads 4 --transformer-model-size 64"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.01,
     0.99),
    # LSTM encoder, 2-layer transformer decoder
    ("--encoder rnn --num-layers 1:2 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --decoder transformer --batch-size 16"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 10000"
     " --transformer-attention-heads 4 --transformer-model-size 64"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.01,
     0.98),
    # 2-layer transformer
    ("--encoder transformer --decoder transformer"
     " --batch-size 16 --max-updates 6000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001"
     " --layer-normalization",
     "--beam-size 1",
     1.07,
     0.98),
    # 3-layer cnn
    ("--encoder cnn --decoder cnn "
     " --batch-size 16 --num-layers 3 --max-updates 10000"
     " --cnn-num-hidden 32 --cnn-positional-embedding-type fixed"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 1",
     1.15,
     0.93)
])
def test_seq_sort(train_params, translate_params, perplexity_thresh, bleu_thresh):
    """Task: sort short sequences of digits"""
    with TemporaryDirectory(prefix="test_seq_sort.") as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        generate_digits_file(train_source_path, train_target_path, _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH,
                             sort_target=True, seed=_SEED_TRAIN)
        generate_digits_file(dev_source_path, dev_target_path, _DEV_LINE_COUNT, _LINE_MAX_LENGTH,
                             sort_target=True, seed=_SEED_DEV)
        # Test model configuration
        perplexity, bleu = run_train_translate(train_params,
                                               translate_params,
                                               train_source_path,
                                               train_target_path,
                                               dev_source_path,
                                               dev_target_path,
                                               max_seq_len=_LINE_MAX_LENGTH + 1,
                                               work_dir=work_dir)
        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh

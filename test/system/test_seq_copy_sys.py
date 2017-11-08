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

from test.common import tmp_digits_dataset, run_train_translate

_TRAIN_LINE_COUNT = 10000
_DEV_LINE_COUNT = 100
_LINE_MAX_LENGTH = 10
_SEED_TRAIN = 13
_SEED_DEV = 17


@pytest.mark.parametrize("train_params, translate_params, perplexity_thresh, bleu_thresh", [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 16 --loss cross-entropy --optimized-metric perplexity"
     " --max-updates 10000 --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001"
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0 --max-updates 5000 --weight-normalization",
     "--beam-size 5",
     1.01,
     0.99),
    # "Vanilla" LSTM encoder-decoder translating in chunks (due to lower --max-input-len)
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 16 --loss cross-entropy --optimized-metric perplexity"
     " --max-updates 10000 --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001"
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0 --max-updates 5000",
     "--beam-size 5 --max-input-len 4",
     1.01,
     0.99),
    # "Vanilla" LSTM encoder-decoder with attention (word-based batching)
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 80 --batch-type word --loss cross-entropy "
     " --optimized-metric perplexity --max-updates 10000 --checkpoint-frequency 1000 --optimizer adam "
     " --initial-learning-rate 0.001 --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0",
     "--beam-size 5",
     1.01,
     0.99),
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
    with tmp_digits_dataset("test_seq_copy.", _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH, _DEV_LINE_COUNT,
                            _LINE_MAX_LENGTH, seed_train=_SEED_TRAIN, seed_dev=_SEED_DEV) as data:
        # Test model configuration
        perplexity, bleu = run_train_translate(train_params,
                                               translate_params,
                                               None,  # no second set of parameters
                                               data['source'],
                                               data['target'],
                                               data['validation_source'],
                                               data['validation_target'],
                                               max_seq_len=_LINE_MAX_LENGTH + 1,
                                               work_dir=data['work_dir'])

        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh


@pytest.mark.parametrize("train_params, translate_params, perplexity_thresh, bleu_thresh", [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 16 --loss cross-entropy --optimized-metric perplexity"
     " --max-updates 10000 --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.04,
     0.98),
    # "Vanilla" LSTM encoder-decoder with attention (word-based batching)
    ("--encoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32 --batch-size 80 --batch-type word --loss cross-entropy"
     " --optimized-metric perplexity --max-updates 10000 --checkpoint-frequency 1000 --optimizer adam "
     " --initial-learning-rate 0.001 --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0",
     "--beam-size 5",
     1.04,
     0.98),
    # 1-layer transformer encoder, LSTM decoder with attention
    ("--encoder transformer --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mhdot --rnn-attention-num-hidden 32 --batch-size 16 --rnn-attention-mhdot-heads 2"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 8000"
     " --transformer-attention-heads 4 --transformer-model-size 64"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.03,
     0.98),
    # LSTM encoder, 2-layer transformer decoder
    ("--encoder rnn --num-layers 1:2 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --decoder transformer --batch-size 16"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 10000"
     " --transformer-attention-heads 4 --transformer-model-size 64"
     " --transformer-feed-forward-num-hidden 64"
     " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001",
     "--beam-size 5",
     1.03,
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
    with tmp_digits_dataset("test_seq_sort.", _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH, _DEV_LINE_COUNT,
                            _LINE_MAX_LENGTH, sort_target=True, seed_train=_SEED_TRAIN, seed_dev=_SEED_DEV) as data:
        # Test model configuration
        perplexity, bleu = run_train_translate(train_params,
                                               translate_params,
                                               None,
                                               data['source'],
                                               data['target'],
                                               data['validation_source'],
                                               data['validation_target'],
                                               max_seq_len=_LINE_MAX_LENGTH + 1,
                                               work_dir=data['work_dir'])
        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh

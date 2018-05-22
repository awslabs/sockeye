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

import logging
import random

import pytest

import sockeye.constants as C
from test.common import tmp_digits_dataset, run_train_translate

logger = logging.getLogger(__name__)

_TRAIN_LINE_COUNT = 10000
_DEV_LINE_COUNT = 100
_LINE_MAX_LENGTH = 10
_TEST_LINE_COUNT = 110
_TEST_LINE_COUNT_EMPTY = 10
_TEST_MAX_LENGTH = 11
_SEED_TRAIN_DATA = 13
_SEED_DEV_DATA = 17

# Training on different systems, as it happens in the Travis system tests, always results in different training curves
# We expect the system tests to be within the bounds of such variations. In order to simulate this variation locally
# the seed is changed every time we run the tests. This way we can hopefully find bounds, which are not optimized for
# the curve observed on a specific system for a fixed seed.
seed = random.randint(0, 1000)


COMMON_TRAINING_PARAMS = " --checkpoint-frequency 1000 --optimizer adam --initial-learning-rate 0.001" \
                         " --decode-and-evaluate 0 --label-smoothing 0.0" \
                         " --optimized-metric perplexity --loss cross-entropy"


@pytest.mark.parametrize("name, train_params, translate_params, use_prepared_data, perplexity_thresh, bleu_thresh", [
    ("Copy:lstm:lstm",
     "--encoder rnn --decoder rnn "
     " --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mlp --rnn-attention-num-hidden 32"
     " --batch-size 16 --batch-type sentence"
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0 --weight-normalization"
     " --max-updates 4000"
     " --gradient-clipping-type norm --gradient-clipping-threshold 10" + COMMON_TRAINING_PARAMS,
     "--beam-size 5 ",
     True,
     1.03,
     0.98),
    ("Copy:chunking",
     "--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mlp --rnn-attention-num-hidden 32"
     " --batch-size 16 --batch-type sentence"
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0"
     " --max-updates 5000" + COMMON_TRAINING_PARAMS,
     "--beam-size 5 --max-input-len 4",
     False,
     1.01,
     0.99),
    ("Copy:word-based-batching:pruning",
     "--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 "
     " --rnn-attention-type mlp --rnn-attention-num-hidden 32 "
     " --batch-size 80 --batch-type word "
     " --max-updates 5000  "
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0 --layer-normalization" + COMMON_TRAINING_PARAMS,
     "--beam-size 5 --batch-size 2 --beam-prune 1",
     True,
     1.01,
     0.99),
    ("Copy:transformer:lstm",
     "--encoder transformer --decoder rnn --num-layers 2:1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mhdot --rnn-attention-num-hidden 32 --rnn-attention-mhdot-heads 1"
     " --max-updates 6000"
     " --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --transformer-activation-type gelu"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 5",
     False,
     1.01,
     0.99),
    ("Copy:lstm:transformer",
     "--encoder rnn --decoder transformer --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-decoder-hidden-dropout 0.0"
     " --batch-size 16 --batch-type sentence"
     " --max-updates 4000"
     " --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --transformer-activation-type swish1" + COMMON_TRAINING_PARAMS,
     "--beam-size 5",
     True,
     1.01,
     0.98),
    ("Copy:transformer:transformer",
     "--encoder transformer --decoder transformer"
     " --max-updates 4000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --num-embed 32"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     False,
     1.02,
     0.98),
    ("Copy:cnn:cnn",
     "--encoder cnn --decoder cnn "
     " --batch-size 16 --num-layers 3 --max-updates 4000"
     " --cnn-num-hidden 32 --cnn-positional-embedding-type fixed --cnn-project-qkv"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     True,
     1.04,
     0.98)
])
def test_seq_copy(name, train_params, translate_params, use_prepared_data, perplexity_thresh, bleu_thresh):
    """Task: copy short sequences of digits"""
    with tmp_digits_dataset("test_seq_copy.", _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH, _DEV_LINE_COUNT,
                            _LINE_MAX_LENGTH, _TEST_LINE_COUNT, _TEST_LINE_COUNT_EMPTY, _TEST_MAX_LENGTH,
                            seed_train=_SEED_TRAIN_DATA, seed_dev=_SEED_DEV_DATA) as data:
        # Test model configuration
        perplexity, bleu, bleu_restrict, chrf = run_train_translate(train_params=train_params,
                                                                    translate_params=translate_params,
                                                                    translate_params_equiv=None,
                                                                    train_source_path=data['source'],
                                                                    train_target_path=data['target'],
                                                                    dev_source_path=data['validation_source'],
                                                                    dev_target_path=data['validation_target'],
                                                                    test_source_path=data['test_source'],
                                                                    test_target_path=data['test_target'],
                                                                    use_prepared_data=use_prepared_data,
                                                                    max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS,
                                                                    restrict_lexicon=True,
                                                                    work_dir=data['work_dir'],
                                                                    seed=seed)
        logger.info("test: %s", name)
        logger.info("perplexity=%f, bleu=%f, bleu_restrict=%f chrf=%f", perplexity, bleu, bleu_restrict, chrf)
        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh
        assert bleu_restrict >= bleu_thresh


@pytest.mark.parametrize(
    "name, train_params, translate_params, use_prepared_data, use_source_factor, perplexity_thresh, bleu_thresh", [
    ("Sort:lstm:lstm",
     "--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 32"
     " --max-updates 7000 "
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 5",
     True, False,
     1.03,
     0.97),
    ("Sort:word-based-batching",
     "--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32 "
     " --rnn-attention-type mlp --rnn-attention-num-hidden 32 "
     " --batch-size 80 --batch-type word"
     " --max-updates 6000"
     " --rnn-dropout-states 0.0:0.1 --embed-dropout 0.1:0.0" + COMMON_TRAINING_PARAMS,
     "--beam-size 5",
     False, False,
     1.03,
     0.97),
    ("Sort:transformer:lstm",
     "--encoder transformer --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --rnn-attention-type mhdot --rnn-attention-num-hidden 32"
     " --batch-size 16 --batch-type sentence"
     " --rnn-attention-mhdot-heads 2"
     " --max-updates 6000"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --transformer-activation-type gelu" + COMMON_TRAINING_PARAMS,
     "--beam-size 5",
     True, False,
     1.03,
     0.97),
    ("Sort:lstm:transformer",
     "--encoder rnn --num-layers 1:2 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 32"
     " --decoder transformer --transformer-model-size 32"
     " --max-updates 7000"
     " --transformer-attention-heads 4"
     " --transformer-feed-forward-num-hidden 64 --transformer-activation-type swish1"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 5",
     False, False,
     1.03,
     0.97),
    ("Sort:transformer:transformer",
     "--encoder transformer --decoder transformer"
     " --batch-size 16 --batch-type sentence"
     " --max-updates 6000"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 32 --num-embed 32"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --transformer-feed-forward-num-hidden 64" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     True, False,
     1.03,
     0.97),
    ("Sort:transformer_with_source_factor",
     "--encoder transformer --decoder transformer"
     " --batch-size 16 --batch-type sentence"
     " --max-updates 6000"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 32 --num-embed 32"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --transformer-feed-forward-num-hidden 64"
     " --source-factors-num-embed 2" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     True, True,
     1.03,
     0.96),
    ("Sort:cnn:cnn",
     "--encoder cnn --decoder cnn"
     " --batch-size 16 --batch-type sentence"
     " --max-updates 6000"
     " --num-layers 3 --cnn-num-hidden 32 --cnn-positional-embedding-type fixed" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     False, False,
     1.05,
     0.94)
])
def test_seq_sort(name, train_params, translate_params, use_prepared_data,
                  use_source_factor, perplexity_thresh, bleu_thresh):
    """Task: sort short sequences of digits"""
    with tmp_digits_dataset("test_seq_sort.", _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH, _DEV_LINE_COUNT, _LINE_MAX_LENGTH,
                            _TEST_LINE_COUNT, _TEST_LINE_COUNT_EMPTY, _TEST_MAX_LENGTH,
                            sort_target=True, seed_train=_SEED_TRAIN_DATA, seed_dev=_SEED_DEV_DATA,
                            with_source_factors=use_source_factor) as data:
        # Test model configuration
        perplexity, bleu, bleu_restrict, chrf = run_train_translate(train_params=train_params,
                                                                    translate_params=translate_params,
                                                                    translate_params_equiv=None,
                                                                    train_source_path=data['source'],
                                                                    train_target_path=data['target'],
                                                                    dev_source_path=data['validation_source'],
                                                                    dev_target_path=data['validation_target'],
                                                                    test_source_path=data['test_source'],
                                                                    test_target_path=data['test_target'],
                                                                    train_source_factor_paths=data.get(
                                                                        'train_source_factors'),
                                                                    dev_source_factor_paths=data.get(
                                                                        'dev_source_factors'),
                                                                    test_source_factor_paths=data.get(
                                                                        'test_source_factors'),
                                                                    use_prepared_data=use_prepared_data,
                                                                    max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS,
                                                                    restrict_lexicon=True,
                                                                    work_dir=data['work_dir'],
                                                                    seed=seed)
        logger.info("test: %s", name)
        logger.info("perplexity=%f, bleu=%f, bleu_restrict=%f chrf=%f", perplexity, bleu, bleu_restrict, chrf)
        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh
        assert bleu_restrict >= bleu_thresh

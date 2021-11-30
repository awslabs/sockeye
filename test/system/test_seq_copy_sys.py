# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import random

import pytest

import sockeye.constants as C
import sockeye.evaluate
import sockeye.utils
from sockeye.test_utils import tmp_digits_dataset
from test.common import check_train_translate

try:
    import mxnet
    # run integration tests with both MXNet and Pytorch
    test_both_backends = [False, True]
except ImportError:
    # only run PyTorch-based tests
    test_both_backends = [True]


logger = logging.getLogger(__name__)

_TRAIN_LINE_COUNT = 10000
_TRAIN_LINE_COUNT_EMPTY = 100
_DEV_LINE_COUNT = 100
_LINE_MAX_LENGTH = 9
_TEST_LINE_COUNT = 110
_TEST_LINE_COUNT_EMPTY = 10
_TEST_MAX_LENGTH = 9
_SEED_TRAIN_DATA = 13
_SEED_DEV_DATA = 17

# Training on different systems, as it happens in the Travis system tests, always results in different training curves
# We expect the system tests to be within the bounds of such variations. In order to simulate this variation locally
# the seed is changed every time we run the tests. This way we can hopefully find bounds, which are not optimized for
# the curve observed on a specific system for a fixed seed.
seed = random.randint(0, 1000)


COMMON_TRAINING_PARAMS = " --checkpoint-interval 1000 --optimizer adam --initial-learning-rate 0.001" \
                         " --decode-and-evaluate 0 --label-smoothing 0.0" \
                         " --optimized-metric perplexity --loss cross-entropy --weight-tying-type src_trg_softmax"


COPY_CASES = [
    ("Copy:transformer:transformer",
     "--encoder transformer --decoder transformer"
     " --max-updates 4000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --num-embed 32"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 1 --prevent-unk",
     False,
     1.02,
     0.98),
    ("greedy",
     "--encoder transformer --decoder transformer"
     " --max-updates 4000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --num-embed 32"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 1 --greedy",
     False,
     1.02,
     0.98),
    ("Copy:transformer:transformer:length_task_learned",
     "--encoder transformer --decoder transformer"
     " --max-updates 4000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --num-embed 32"
     " --length-task length --length-task-weight 1.5 --length-task-layers 1"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 5 --batch-size 2 --brevity-penalty-type learned"
     " --brevity-penalty-weight 0.9 --max-input-length %s" % _TEST_MAX_LENGTH,
     True,
     1.02,
     0.96),
    ("Copy:transformer:transformer:length_task_constant",
     "--encoder transformer --decoder transformer"
     " --max-updates 4000"
     " --num-layers 2 --transformer-attention-heads 4 --transformer-model-size 32"
     " --transformer-feed-forward-num-hidden 64 --num-embed 32"
     " --length-task ratio --length-task-weight 0.1 --length-task-layers 1"
     " --batch-size 16 --batch-type sentence" + COMMON_TRAINING_PARAMS,
     "--beam-size 5 --batch-size 2 --brevity-penalty-type constant"
     " --brevity-penalty-weight 1.0 --brevity-penalty-constant-length-ratio 1 --max-input-length %s" % _TEST_MAX_LENGTH,
     False,
     1.02,
     0.94)
]

COPY_CASES = [(use_pytorch, *other_params) for use_pytorch in test_both_backends for other_params in COPY_CASES]


@pytest.mark.parametrize("use_pytorch, name, train_params, translate_params, use_prepared_data, "
                         "perplexity_thresh, bleu_thresh", COPY_CASES)
def test_seq_copy(use_pytorch, name, train_params, translate_params, use_prepared_data, perplexity_thresh, bleu_thresh):
    """Task: copy short sequences of digits"""
    with tmp_digits_dataset(prefix="test_seq_copy",
                            train_line_count=_TRAIN_LINE_COUNT,
                            train_line_count_empty=_TRAIN_LINE_COUNT_EMPTY,
                            train_max_length=_LINE_MAX_LENGTH,
                            dev_line_count=_DEV_LINE_COUNT,
                            dev_max_length=_LINE_MAX_LENGTH,
                            test_line_count=_TEST_LINE_COUNT,
                            test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
                            test_max_length=_TEST_MAX_LENGTH,
                            sort_target=False,
                            with_n_source_factors=0) as data:
        data = check_train_translate(train_params=train_params,
                                     translate_params=translate_params,
                                     data=data,
                                     use_prepared_data=use_prepared_data,
                                     max_seq_len=_LINE_MAX_LENGTH,
                                     compare_output=True,
                                     seed=seed,
                                     use_pytorch=use_pytorch)

        # get best validation perplexity
        metrics = sockeye.utils.read_metrics_file(os.path.join(data['model'], C.METRICS_NAME))
        perplexity = min(m[C.PERPLEXITY + '-val'] for m in metrics)

        # compute metrics
        hypotheses = [json['translation'] for json in data['test_outputs']]
        hypotheses_restricted = [json['translation'] for json in data['test_outputs_restricted']]
        bleu = sockeye.evaluate.raw_corpus_bleu(hypotheses=hypotheses, references=data['test_targets'])
        chrf = sockeye.evaluate.raw_corpus_chrf(hypotheses=hypotheses, references=data['test_targets'])
        bleu_restrict = sockeye.evaluate.raw_corpus_bleu(hypotheses=hypotheses_restricted,
                                                         references=data['test_targets'])

        logger.info("================")
        logger.info("test results: %s", name)
        logger.info("perplexity=%f, bleu=%f, bleu_restrict=%f chrf=%f", perplexity, bleu, bleu_restrict, chrf)
        logger.info("================\n")
        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh
        assert bleu_restrict >= bleu_thresh


SORT_CASES = [
    ("Sort:transformer:transformer:batch_word",
     "--encoder transformer --decoder transformer"
     " --max-seq-len 10 --batch-size 90 --update-interval 1 --batch-type word --batch-sentences-multiple-of 1"
     " --max-updates 6000"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 32 --num-embed 32"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --transformer-feed-forward-num-hidden 64" + COMMON_TRAINING_PARAMS,
     "--beam-size 1 --prevent-unk",
     True, 0, 0,
     1.03,
     0.97),
    ("Sort:transformer:transformer:source_factors:target_factors:batch_max_word",
     "--encoder transformer --decoder transformer"
     " --max-seq-len 10 --batch-size 70 --update-interval 2 --batch-type max-word --batch-sentences-multiple-of 1"
     " --max-updates 6000"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 32 --num-embed 32"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --transformer-feed-forward-num-hidden 64 --target-factors-weight 1.0"
     " --target-factors-num-embed 32 --target-factors-combine sum"
     " --source-factors-num-embed 2 2 2" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     True, 3, 1,
     1.03,
     0.96),
    ("Sort:transformer:ssru_transformer:batch_word",
     "--encoder transformer --decoder ssru_transformer"
     " --max-seq-len 10 --batch-size 90 --update-interval 1 --batch-type word --batch-sentences-multiple-of 1"
     " --max-updates 6000"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 32 --num-embed 32"
     " --transformer-dropout-attention 0.0 --transformer-dropout-act 0.0 --transformer-dropout-prepost 0.0"
     " --transformer-feed-forward-num-hidden 64" + COMMON_TRAINING_PARAMS,
     "--beam-size 1",
     True, 0, 0,
     1.03,
     0.97)
]


SORT_CASES = [(use_pytorch, *other_params) for use_pytorch in test_both_backends for other_params in SORT_CASES]


@pytest.mark.parametrize("use_pytorch, name, train_params, translate_params, use_prepared_data, n_source_factors, "
                         "n_target_factors, perplexity_thresh, bleu_thresh", SORT_CASES)
def test_seq_sort(use_pytorch, name, train_params, translate_params, use_prepared_data,
                  n_source_factors, n_target_factors, perplexity_thresh, bleu_thresh):
    """Task: sort short sequences of digits"""
    with tmp_digits_dataset("test_seq_sort.",
                            _TRAIN_LINE_COUNT, _TRAIN_LINE_COUNT_EMPTY, _LINE_MAX_LENGTH,
                            _DEV_LINE_COUNT, _LINE_MAX_LENGTH,
                            _TEST_LINE_COUNT, _TEST_LINE_COUNT_EMPTY, _TEST_MAX_LENGTH,
                            sort_target=True, seed_train=_SEED_TRAIN_DATA, seed_dev=_SEED_DEV_DATA,
                            with_n_source_factors=n_source_factors,
                            with_n_target_factors=n_target_factors) as data:
        data = check_train_translate(train_params=train_params,
                                     translate_params=translate_params,
                                     data=data,
                                     use_prepared_data=use_prepared_data,
                                     max_seq_len=_LINE_MAX_LENGTH,
                                     compare_output=True,
                                     seed=seed,
                                     use_pytorch=use_pytorch)

        # get best validation perplexity
        metrics = sockeye.utils.read_metrics_file(os.path.join(data['model'], C.METRICS_NAME))
        perplexity = min(m[C.PERPLEXITY + '-val'] for m in metrics)

        # compute metrics
        hypotheses = [json['translation'] for json in data['test_outputs']]
        hypotheses_restricted = [json['translation'] for json in data['test_outputs_restricted']]
        bleu = sockeye.evaluate.raw_corpus_bleu(hypotheses=hypotheses, references=data['test_targets'])
        chrf = sockeye.evaluate.raw_corpus_chrf(hypotheses=hypotheses, references=data['test_targets'])
        bleu_restrict = sockeye.evaluate.raw_corpus_bleu(hypotheses=hypotheses_restricted,
                                                         references=data['test_targets'])

        logger.info("================")
        logger.info("test results: %s", name)
        logger.info("perplexity=%f, bleu=%f, bleu_restrict=%f chrf=%f", perplexity, bleu, bleu_restrict, chrf)
        logger.info("================\n")
        assert perplexity <= perplexity_thresh
        assert bleu >= bleu_thresh
        assert bleu_restrict >= bleu_thresh

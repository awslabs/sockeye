# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
Tests constraints in many forms.
"""

import pytest
import random

import sockeye.constants as C
from test.common import run_train_translate, tmp_digits_dataset

_TRAIN_LINE_COUNT = 20
_DEV_LINE_COUNT = 5
_TEST_LINE_COUNT = 5
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20

ENCODER_DECODER_SETTINGS = [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 4 "
     " --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 8 --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01 --batch-type sentence "
     " --decode-and-evaluate 0",
     2, 10),
    # Full transformer
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01",
     1, 10)]


@pytest.mark.parametrize(
    "train_params, batch_size, beam_size",
    ENCODER_DECODER_SETTINGS)
def test_constraints(train_params: str,
                     beam_size: int,
                     batch_size: int):
    """Task: copy short sequences of digits"""

    with tmp_digits_dataset(prefix="test_constraints",
                            train_line_count=_TRAIN_LINE_COUNT,
                            train_max_length=_LINE_MAX_LENGTH,
                            dev_line_count=_DEV_LINE_COUNT,
                            dev_max_length=_LINE_MAX_LENGTH,
                            test_line_count=_TEST_LINE_COUNT,
                            test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
                            test_max_length=_TEST_MAX_LENGTH,
                            sort_target=False) as data:

        translate_params = " --batch-size {} --beam-size {}".format(batch_size, beam_size)

        # Ignore return values (perplexity and BLEU) for integration test
        run_train_translate(train_params=train_params,
                            translate_params=translate_params,
                            translate_params_equiv=None,
                            train_source_path=data['source'],
                            train_target_path=data['target'],
                            dev_source_path=data['validation_source'],
                            dev_target_path=data['validation_target'],
                            test_source_path=data['test_source'],
                            test_target_path=data['test_target'],
                            max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS,
                            work_dir=data['work_dir'],
                            use_prepared_data=False,
                            restrict_lexicon=False,
                            use_target_constraints=True)

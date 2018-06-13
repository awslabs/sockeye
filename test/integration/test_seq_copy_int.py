# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
     " --rnn-attention-num-hidden 8 --batch-size 2 --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01 --batch-type sentence "
     " --decode-and-evaluate 0",
     "--beam-size 2",
     True, False, False, True),
    # "Kitchen sink" LSTM encoder-decoder with attention
    ("--encoder rnn --decoder rnn --num-layers 3:2 --rnn-cell-type lstm --rnn-num-hidden 8"
     " --rnn-residual-connections"
     " --num-embed 8 --rnn-attention-type coverage --rnn-attention-num-hidden 8 --weight-tying "
     "--rnn-attention-use-prev-word --rnn-context-gating --layer-normalization --batch-size 2 "
     "--loss cross-entropy --label-smoothing 0.1 --loss-normalization-type batch --optimized-metric perplexity"
     " --max-updates 2 --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01"
     " --rnn-dropout-inputs 0.5:0.1 --rnn-dropout-states 0.5:0.1 --embed-dropout 0.1 --rnn-decoder-hidden-dropout 0.01"
     " --rnn-decoder-state-init avg --rnn-encoder-reverse-input --rnn-dropout-recurrent 0.1:0.0"
     " --rnn-h2h-init orthogonal_stacked --batch-type sentence --decode-and-evaluate 0"
     " --learning-rate-decay-param-reset --weight-normalization --source-factors-num-embed 5",
     "--beam-size 2",
     False, True, True, False),
    # Convolutional embedding encoder + LSTM encoder-decoder with attention
    ("--encoder rnn-with-conv-embed --decoder rnn --conv-embed-max-filter-width 3 --conv-embed-num-filters 4:4:8"
     " --conv-embed-pool-stride 2 --conv-embed-num-highway-layers 1 --num-layers 1 --rnn-cell-type lstm"
     " --rnn-num-hidden 8 --num-embed 4 --rnn-attention-num-hidden 8 --batch-size 2 --loss cross-entropy"
     " --optimized-metric perplexity --max-updates 2 --checkpoint-frequency 2 --optimizer adam --batch-type sentence"
     " --initial-learning-rate 0.01 --decode-and-evaluate 0",
     "--beam-size 2",
     False, False, False, False),
    # Transformer encoder, GRU decoder, mhdot attention
    ("--encoder transformer --decoder rnn --num-layers 2:1 --rnn-cell-type gru --rnn-num-hidden 8 --num-embed 4:8"
     " --transformer-attention-heads 2 --transformer-model-size 4"
     " --transformer-feed-forward-num-hidden 16 --transformer-activation-type gelu"
     " --rnn-attention-type mhdot --rnn-attention-mhdot-heads 4 --rnn-attention-num-hidden 8 --batch-size 2 "
     " --max-updates 2 --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01"
     " --weight-init-xavier-factor-type avg --weight-init-scale 3.0 --embed-weight-init normal --batch-type sentence"
     " --decode-and-evaluate 0",
     "--beam-size 2",
     False, True, False, False),
    # LSTM encoder, Transformer decoder
    ("--encoder rnn --decoder transformer --num-layers 2:2 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 8"
     " --transformer-attention-heads 2 --transformer-model-size 8"
     " --transformer-feed-forward-num-hidden 16 --transformer-activation-type swish1"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 3",
     False, True, False, False),
    # Full transformer
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2",
     True, False, False, False),
    # Full transformer with source factor
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01 --source-factors-num-embed 4",
     "--beam-size 2",
     True, False, True, False),
    # 2-layer cnn
    ("--encoder cnn --decoder cnn "
     " --batch-size 2 --num-layers 2 --max-updates 2 --checkpoint-frequency 2"
     " --cnn-num-hidden 32 --cnn-positional-embedding-type fixed"
     " --optimizer adam --initial-learning-rate 0.001 --batch-type sentence --decode-and-evaluate 0",
     "--beam-size 2",
     True, False, False, False),
    # Vanilla LSTM like above but activating LHUC. In the normal case you would
    # start with a trained system instead of a random initialized one like here.
    (
     "--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 4 "
     " --rnn-attention-num-hidden 8 --rnn-attention-type mlp"
     " --batch-size 2 --batch-type sentence"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01 --lhuc all",
     "--beam-size 2",
    True, False, False, False),
    # Full transformer with LHUC
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence  --decode-and-evaluate 0"
     " --checkpoint-frequency 2 --optimizer adam --initial-learning-rate 0.01 --lhuc all",
     "--beam-size 2",
     True, False, False, False)]


@pytest.mark.parametrize(
    "train_params, translate_params, restrict_lexicon, use_prepared_data, use_source_factors, use_constrained_decoding",
    ENCODER_DECODER_SETTINGS)
def test_seq_copy(train_params: str,
                  translate_params: str,
                  restrict_lexicon: bool,
                  use_prepared_data: bool,
                  use_source_factors: bool,
                  use_constrained_decoding: bool):
    """Task: copy short sequences of digits"""

    with tmp_digits_dataset(prefix="test_seq_copy",
                            train_line_count=_TRAIN_LINE_COUNT,
                            train_max_length=_LINE_MAX_LENGTH,
                            dev_line_count=_DEV_LINE_COUNT,
                            dev_max_length=_LINE_MAX_LENGTH,
                            test_line_count=_TEST_LINE_COUNT,
                            test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
                            test_max_length=_TEST_MAX_LENGTH,
                            sort_target=False,
                            with_source_factors=use_source_factors,
                            with_target_constraints=use_constrained_decoding) as data:

        # Only one of these is supported at a time in the tests
        assert not (use_source_factors and use_constrained_decoding)

        # When using source factors
        train_source_factor_paths, dev_source_factor_paths, test_source_factor_paths = None, None, None
        if use_source_factors:
            train_source_factor_paths = [data['source']]
            dev_source_factor_paths = [data['validation_source']]
            test_source_factor_paths = [data['test_source']]

        if use_constrained_decoding:
            translate_params += " --json-input"

        # Test model configuration, including the output equivalence of batch and no-batch decoding
        translate_params_batch = translate_params + " --batch-size 2"

        # Ignore return values (perplexity and BLEU) for integration test
        run_train_translate(train_params=train_params,
                            translate_params=translate_params,
                            translate_params_equiv=translate_params_batch,
                            train_source_path=data['source'],
                            train_target_path=data['target'],
                            dev_source_path=data['validation_source'],
                            dev_target_path=data['validation_target'],
                            test_source_path=data['test_source'],
                            test_target_path=data['test_target'],
                            train_source_factor_paths=train_source_factor_paths,
                            dev_source_factor_paths=dev_source_factor_paths,
                            test_source_factor_paths=test_source_factor_paths,
                            max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS,
                            restrict_lexicon=restrict_lexicon,
                            work_dir=data['work_dir'],
                            use_prepared_data=use_prepared_data)

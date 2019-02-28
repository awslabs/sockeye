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
import logging
import os
import sys
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch

import mxnet as mx
import numpy as np
import pytest

import sockeye.average
import sockeye.checkpoint_decoder
import sockeye.evaluate
import sockeye.extract_parameters
from sockeye import constants as C
from test.common import check_train_translate, run_train_translate, tmp_digits_dataset

logger = logging.getLogger(__name__)

_TRAIN_LINE_COUNT = 20
_TRAIN_LINE_COUNT_EMPTY = 1
_DEV_LINE_COUNT = 5
_TEST_LINE_COUNT = 5
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20


# tuple format: (train_params, translate_params, use_prepared_data, use_source_factors)
ENCODER_DECODER_SETTINGS = [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 4 "
     " --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 8 --batch-size 2 --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --batch-type sentence "
     " --decode-and-evaluate 0",
     "--beam-size 2 --softmax-temperature 0.01",
     False, False),
    # "Vanilla" LSTM encoder-decoder with attention, greedy and skip topk
    ("--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 4 "
     " --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 8 --batch-size 2 --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --batch-type sentence "
     " --decode-and-evaluate 0",
     "--beam-size 1 --softmax-temperature 0.01 --skip-topk",
     False, False),
    # "Vanilla" LSTM encoder-decoder with attention, higher nbest size
    ("--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 4 "
     " --rnn-attention-type mlp"
     " --rnn-attention-num-hidden 8 --batch-size 2 --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --batch-type sentence "
     " --decode-and-evaluate 0",
     "--beam-size 2 --softmax-temperature 0.01 --nbest-size 2",
     False, False),
    # "Kitchen sink" LSTM encoder-decoder with attention
    ("--encoder rnn --decoder rnn --num-layers 3:2 --rnn-cell-type lstm --rnn-num-hidden 8"
     " --rnn-residual-connections"
     " --num-embed 8 --rnn-attention-type coverage --rnn-attention-num-hidden 8 --weight-tying "
     "--rnn-attention-use-prev-word --rnn-context-gating --layer-normalization --batch-size 2 "
     "--loss cross-entropy --label-smoothing 0.1 --loss-normalization-type batch --optimized-metric perplexity"
     " --max-updates 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01"
     " --rnn-dropout-inputs 0.5:0.1 --rnn-dropout-states 0.5:0.1 --embed-dropout 0.1 --rnn-decoder-hidden-dropout 0.01"
     " --rnn-decoder-state-init avg --rnn-encoder-reverse-input --rnn-dropout-recurrent 0.1:0.0"
     " --rnn-h2h-init orthogonal_stacked --batch-type sentence --decode-and-evaluate 0"
     " --learning-rate-decay-param-reset --weight-normalization --source-factors-num-embed 5 --source-factors-combine concat",
     "--beam-size 2 --beam-search-stop first",
     True, True),
    # Convolutional embedding encoder + LSTM encoder-decoder with attention
    ("--encoder rnn-with-conv-embed --decoder rnn --conv-embed-max-filter-width 3 --conv-embed-num-filters 4:4:8"
     " --conv-embed-pool-stride 2 --conv-embed-num-highway-layers 1 --num-layers 1 --rnn-cell-type lstm"
     " --rnn-num-hidden 8 --num-embed 4 --rnn-attention-num-hidden 8 --batch-size 2 --loss cross-entropy"
     " --optimized-metric perplexity --max-updates 2 --checkpoint-interval 2 --optimizer adam --batch-type sentence"
     " --initial-learning-rate 0.01 --decode-and-evaluate 0",
     "--beam-size 2",
     False, False),
    # Transformer encoder, GRU decoder, mhdot attention
    ("--encoder transformer --decoder rnn --num-layers 2:1 --rnn-cell-type gru --rnn-num-hidden 8 --num-embed 4:8"
     " --transformer-attention-heads 2 --transformer-model-size 4"
     " --transformer-feed-forward-num-hidden 16 --transformer-activation-type gelu"
     " --rnn-attention-type mhdot --rnn-attention-mhdot-heads 4 --rnn-attention-num-hidden 8 --batch-size 2 "
     " --max-updates 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01"
     " --weight-init-xavier-factor-type avg --weight-init-scale 3.0 --embed-weight-init normal --batch-type sentence"
     " --decode-and-evaluate 0",
     "--beam-size 2",
     True, False),
    # LSTM encoder, Transformer decoder
    ("--encoder rnn --decoder transformer --num-layers 2:2 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 8"
     " --transformer-attention-heads 2 --transformer-model-size 8"
     " --transformer-feed-forward-num-hidden 16 --transformer-activation-type swish1"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 3",
     True, False),
    # Full transformer
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2 --nbest-size 2",
     False, False),
    # Full transformer with source factor
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --source-factors-combine sum",
     "--beam-size 2",
     False, True),
    # 2-layer cnn
    ("--encoder cnn --decoder cnn "
     " --batch-size 2 --num-layers 2 --max-updates 2 --checkpoint-interval 2"
     " --cnn-num-hidden 32 --cnn-positional-embedding-type fixed"
     " --optimizer adam --initial-learning-rate 0.001 --batch-type sentence --decode-and-evaluate 0",
     "--beam-size 2",
     False, False),
    # Vanilla LSTM like above but activating LHUC. In the normal case you would
    # start with a trained system instead of a random initialized one like here.
    ("--encoder rnn --decoder rnn --num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 8 --num-embed 4 "
     " --rnn-attention-num-hidden 8 --rnn-attention-type mlp"
     " --batch-size 2 --batch-type sentence"
     " --loss cross-entropy --optimized-metric perplexity --max-updates 2"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --lhuc all",
     "--beam-size 2 --nbest-size 2",
     False, False),
    # Full transformer with LHUC
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg --embed-weight-init=normal"
     " --batch-size 2 --max-updates 2 --batch-type sentence  --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --lhuc all",
     "--beam-size 2 --beam-prune 1",
     False, False)]


@pytest.mark.parametrize("train_params, translate_params, use_prepared_data, use_source_factors",
                         ENCODER_DECODER_SETTINGS)
def test_seq_copy(train_params: str,
                  translate_params: str,
                  use_prepared_data: bool,
                  use_source_factors: bool):
    """
    Task: copy short sequences of digits
    """

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
                            with_source_factors=use_source_factors) as data:

        # TODO: Here we temporarily switch off comparing translation and scoring scores, which
        # sometimes produces inconsistent results for --batch-size > 1 (see issue #639 on github).
        check_train_translate(train_params=train_params,
                              translate_params=translate_params,
                              data=data,
                              use_prepared_data=use_prepared_data,
                              max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS,
                              compare_translate_vs_scoring_scores=False)


TINY_TEST_MODEL = [(" --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 4 --num-embed 4"
                    " --transformer-feed-forward-num-hidden 4 --weight-tying --weight-tying-type src_trg_softmax"
                    " --batch-size 2 --batch-type sentence --max-updates 4 --decode-and-evaluate 0"
                    " --checkpoint-frequency 4",
                    "--beam-size 1")]


@pytest.mark.parametrize("train_params, translate_params", TINY_TEST_MODEL)
def test_other_clis(train_params: str, translate_params: str):
    """
    Task: test CLIs and core features other than train & translate.
    """
    with tmp_digits_dataset(prefix="test_other_clis",
                            train_line_count=_TRAIN_LINE_COUNT,
                            train_line_count_empty=_TRAIN_LINE_COUNT_EMPTY,
                            train_max_length=_LINE_MAX_LENGTH,
                            dev_line_count=_DEV_LINE_COUNT,
                            dev_max_length=_LINE_MAX_LENGTH,
                            test_line_count=_TEST_LINE_COUNT,
                            test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
                            test_max_length=_TEST_MAX_LENGTH) as data:
        # train a minimal default model
        data = run_train_translate(train_params=train_params,
                                   translate_params=translate_params,
                                   data=data,
                                   max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS)

        _test_checkpoint_decoder(data['dev_source'], data['dev_target'], data['model'])
        _test_parameter_averaging(data['model'])
        _test_extract_parameters_cli(data['model'])
        _test_evaluate_cli(data['test_outputs'], data['test_target'])


def _test_evaluate_cli(test_outputs: List[str], test_target_path: str):
    """
    Runs sockeye-evaluate CLI with translations and a reference file.
    """
    with TemporaryDirectory(prefix="test_evaluate") as work_dir:
        # write temporary output file
        out_path = os.path.join(work_dir, 'hypotheses')
        with open(out_path, 'w') as fd:
            for output in test_outputs:
                print(output, file=fd)
        # Run evaluate cli
        eval_params = "{} --hypotheses {hypotheses} --references {references} --metrics {metrics}".format(
            sockeye.evaluate.__file__,
            hypotheses=out_path,
            references=test_target_path,
            metrics="bleu chrf rouge1")
        with patch.object(sys, "argv", eval_params.split()):
            sockeye.evaluate.main()


def _test_extract_parameters_cli(model_path: str):
    """
    Runs parameter extraction CLI and asserts that the resulting numpy serialization contains a parameter key.
    """
    extract_params = "--input {input} --names target_output_bias --list-all --output {output}".format(
        output=os.path.join(model_path, "params.extracted"), input=model_path)
    with patch.object(sys, "argv", extract_params.split()):
        sockeye.extract_parameters.main()
    with np.load(os.path.join(model_path, "params.extracted.npz")) as data:
        assert "target_output_bias" in data


def _test_parameter_averaging(model_path: str):
    """
    Runs parameter averaging with all available strategies
    """
    for strategy in C.AVERAGE_CHOICES:
        points = sockeye.average.find_checkpoints(model_path=model_path,
                                                  size=4,
                                                  strategy=strategy,
                                                  metric=C.PERPLEXITY)
        assert len(points) > 0
        averaged_params = sockeye.average.average(points)
        assert averaged_params


def _test_checkpoint_decoder(dev_source_path: str, dev_target_path: str, model_path: str):
    """
    Runs checkpoint decoder on 10% of the dev data and checks whether metric keys are present in the result dict.
    """
    with open(dev_source_path) as dev_fd:
        num_dev_sent = sum(1 for _ in dev_fd)
    sample_size = min(1, int(num_dev_sent * 0.1))
    cp_decoder = sockeye.checkpoint_decoder.CheckpointDecoder(context=mx.cpu(),
                                                              inputs=[dev_source_path],
                                                              references=dev_target_path,
                                                              model=model_path,
                                                              sample_size=sample_size,
                                                              batch_size=2,
                                                              beam_size=2)
    cp_metrics = cp_decoder.decode_and_evaluate()
    logger.info("Checkpoint decoder metrics: %s", cp_metrics)
    assert 'bleu-val' in cp_metrics
    assert 'chrf-val' in cp_metrics
    assert 'decode-walltime-val' in cp_metrics

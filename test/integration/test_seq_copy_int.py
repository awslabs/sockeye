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
from itertools import product
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch

import pytest
import torch as pt

try:
    import mxnet
    # run integration tests with both MXNet and Pytorch
    test_both_backends = [False, True]
except ImportError:
    # only run PyTorch-based tests
    test_both_backends = [True]

import sockeye.average
import sockeye.checkpoint_decoder_pt
import sockeye.evaluate
from sockeye import constants as C
from sockeye.config import Config
from sockeye.model_pt import load_model
from sockeye.test_utils import run_train_translate, tmp_digits_dataset
from test.common import check_train_translate

logger = logging.getLogger(__name__)

_TRAIN_LINE_COUNT = 20
_TRAIN_LINE_COUNT_EMPTY = 1
_DEV_LINE_COUNT = 5
_TEST_LINE_COUNT = 5
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20

# tuple format: (train_params, translate_params, use_prepared_data, use_source_factors)
ENCODER_DECODER_SETTINGS_TEMPLATE = [
    # Basic transformer, nbest=2 decoding
    ("--encoder transformer --decoder {decoder}"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     # Note: We set the checkpoint interval > max updates in order to make sure we create a checkpoint when reaching
     # max updates independent of the checkpoint interval
     " --checkpoint-interval 20 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 2 --nbest-size 2",
     False, 0, 0),
    # Basic transformer w/ prepared data & greedy decoding
    ("--encoder transformer --decoder {decoder}"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 1 --greedy",
     True, 0, 0),
    # Basic transformer with source and target factors, beam-search-stop first decoding
    ("--encoder transformer --decoder {decoder}"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type trg_softmax"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01"
     " --source-factors-combine sum concat average --source-factors-share-embedding true false true"
     " --source-factors-num-embed 8 2 8"
     " --target-factors-combine sum --target-factors-share-embedding false"
     " --target-factors-num-embed 8",
     "--beam-size 2 --beam-search-stop first",
     True, 3, 1),
    # Basic transformer with LHUC DISABLE FOR MX2 FOR NOW (UNKNOWN FAILURE)
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence  --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --lhuc all",
     "--beam-size 2",
     False, 0, 0),
    # Basic transformer and length ratio prediction, and learned brevity penalty during inference
    ("--encoder transformer --decoder {decoder}"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01"
     " --length-task ratio --length-task-weight 1.0 --length-task-layers 1",
     "--beam-size 2"
     " --brevity-penalty-type learned --brevity-penalty-weight 1.0",
     True, 0, 0),
    # Basic transformer and absolute length prediction, and constant brevity penalty during inference
    ("--encoder transformer --decoder {decoder}"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01"
     " --length-task length --length-task-weight 1.0 --length-task-layers 1",
     "--beam-size 2"
     " --brevity-penalty-type constant --brevity-penalty-weight 2.0 --brevity-penalty-constant-length-ratio 1.5",
     False, 0, 0),
    # Basic transformer, training only the decoder
    ("--encoder transformer --decoder {decoder}"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01"
     " --fixed-param-strategy " + C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER,
     "--beam-size 2",
     False, 0, 0),
]

# expand test cases across transformer & ssru, as well as use_pytorch true/false
TEST_CASES = [(use_pytorch, train_params.format(decoder=decoder), *other_params)
              for decoder, use_pytorch in product(C.DECODERS, test_both_backends)
              for (train_params, *other_params) in ENCODER_DECODER_SETTINGS_TEMPLATE]


@pytest.mark.parametrize("use_pytorch, train_params, translate_params, use_prepared_data,"
                         "n_source_factors, n_target_factors", TEST_CASES)
def test_seq_copy(use_pytorch: bool,
                  train_params: str,
                  translate_params: str,
                  use_prepared_data: bool,
                  n_source_factors: int,
                  n_target_factors: int):
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
                            with_n_source_factors=n_source_factors,
                            with_n_target_factors=n_target_factors) as data:

        # TODO: Here we temporarily switch off comparing translation and scoring scores, which
        # sometimes produces inconsistent results for --batch-size > 1 (see issue #639 on github).
        check_train_translate(train_params=train_params,
                              translate_params=translate_params,
                              data=data,
                              use_prepared_data=use_prepared_data,
                              max_seq_len=_LINE_MAX_LENGTH,
                              compare_output=False,
                              use_pytorch=use_pytorch)


TINY_TEST_MODEL = [(" --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 4 --num-embed 4"
                    " --transformer-feed-forward-num-hidden 4 --weight-tying-type src_trg_softmax"
                    " --batch-size 2 --batch-type sentence --max-updates 4 --decode-and-evaluate 0"
                    " --checkpoint-interval 4",
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
                            test_line_count_empty=0,
                            test_max_length=_TEST_MAX_LENGTH) as data:
        # train a minimal default model
        data = run_train_translate(train_params=train_params,
                                   translate_params=translate_params,
                                   data=data,
                                   max_seq_len=_LINE_MAX_LENGTH,
                                   use_pytorch=True)

        _test_checkpoint_decoder(data['dev_source'], data['dev_target'], data['model'])
        _test_mc_dropout(data['model'])
        _test_parameter_averaging(data['model'])
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
                print(output['translation'], file=fd)
        # Run evaluate cli
        eval_params = "{} --hypotheses {hypotheses} --references {references} --metrics {metrics}".format(
            sockeye.evaluate.__file__,
            hypotheses=out_path,
            references=test_target_path,
            metrics="bleu chrf rouge1")
        with patch.object(sys, "argv", eval_params.split()):
            sockeye.evaluate.main()


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

    model, source_vocabs, target_vocabs = load_model(model_folder=model_path, device=pt.device('cpu'))

    cp_decoder = sockeye.checkpoint_decoder_pt.CheckpointDecoder(device=pt.device('cpu'),
                                                                 inputs=[dev_source_path],
                                                                 references=[dev_target_path],
                                                                 source_vocabs=source_vocabs,
                                                                 target_vocabs=target_vocabs,
                                                                 model=model,
                                                                 model_folder=model_path,
                                                                 sample_size=sample_size,
                                                                 batch_size=2,
                                                                 beam_size=2)
    cp_metrics = cp_decoder.decode_and_evaluate()
    logger.info("Checkpoint decoder metrics: %s", cp_metrics)
    assert 'bleu' in cp_metrics
    assert 'chrf' in cp_metrics
    assert 'decode-walltime' in cp_metrics


def _test_mc_dropout(model_path: str):
    """
    Check that loading a model with MC Dropoout returns a model with dropout layers.
    """
    model, _, _ = load_model(model_folder=model_path, device=pt.device('cpu'), mc_dropout=True, inference_only=True)

    # Ensure the model has some dropout turned on
    config_blocks = [block for _, block in model.config.__dict__.items() if isinstance(block, Config)]
    dropout_settings = {setting: val for block in config_blocks for setting, val in block.__dict__.items()
            if "dropout" in setting}
    assert any(s > 0.0 for s in dropout_settings.values())

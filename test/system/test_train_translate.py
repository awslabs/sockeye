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

from tempfile import TemporaryDirectory

import os
import random
import sys

from unittest.mock import patch

import pytest

import sockeye.bleu
import sockeye.constants as C
import sockeye.train
import sockeye.translate
import sockeye.utils

_DIGITS = "0123456789"
_TRAIN_LINE_COUNT = 10000
_DEV_LINE_COUNT = 100
_LINE_MAX_LENGTH = 9

_TRAIN_PARAMS_COMMON = "--use-cpu --max-seq-len {max_len} --source {train_corpus} --target {train_corpus}" \
                       " --validation-source {dev_corpus} --validation-target {dev_corpus} --output {model}"
_TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output}"

def _generate_digits_file(path: str, line_count: int = 100, line_length: int = 9):
    with open(path, "w") as out:
        for _ in range(line_count):
            print(" ".join(random.choice(_DIGITS) for _ in range(random.randint(1, line_length))), file=out)


@pytest.mark.parametrize("train_params, translate_params, perplexity_thresh, bleu_thresh", [
    # "Vanilla" LSTM encoder-decoder with attention
    ("--encoder rnn --rnn-num-layers 1 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 8 --attention-type mlp"
     " --attention-num-hidden 32 --batch-size 8 --loss cross-entropy --optimized-metric perplexity --max-updates 2500"
     " --checkpoint-frequency 2500 --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 5",
     1.1,
     0.95),
    # "Kitchen sink" LSTM encoder-decoder with attention
    ("--encoder rnn --rnn-num-layers 2 --rnn-cell-type lstm --rnn-num-hidden 64 --num-embed 64"
     " --attention-type coverage --attention-num-hidden 16 --weight-tying --attention-use-prev-word --context-gating"
     " --layer-normalization --batch-size 8 --loss smoothed-cross-entropy --smoothed-cross-entropy-alpha 0.1"
     " --normalize-loss --optimized-metric perplexity --max-updates 2500 --checkpoint-frequency 2500 --dropout 0.1"
     " --optimizer adam --initial-learning-rate 0.01",
     "--beam-size 5",
     1.5,
     0.9),
    # Convolutional embedding encoder + LSTM encoder-decoder with attention
    ("--encoder rnn-with-conv-embed --conv-embed-max-filter-width 3 --conv-embed-num-filters 8 8 16"
     " --conv-embed-pool-stride 2 --conv-embed-num-highway-layers 1 --rnn-num-layers 1 --rnn-cell-type lstm"
     " --rnn-num-hidden 64 --num-embed 8 --attention-num-hidden 32 --batch-size 8 --loss cross-entropy"
     " --optimized-metric perplexity --max-updates 2500 --checkpoint-frequency 2500 --optimizer adam"
     " --initial-learning-rate 0.01",
     "--beam-size 5",
     2.0,
     0.65),
])

def test_train_translate(train_params, translate_params, perplexity_thresh, bleu_thresh):
    """
    Train a model and translate a dev set.  Verify perplexity and BLEU.

    :param train_params: Command line args for model training.
    :param translate_params: Command line args for translation.
    :param perplexity_thresh: Maximum perplexity for success
    :param bleu_thresh: Minimum BLEU score for success
    """
    with TemporaryDirectory(prefix="sockeye.") as work_dir:

        # Simple digits files for train/dev data
        train_path = os.path.join(work_dir, "train.txt")
        dev_path = os.path.join(work_dir, "dev.txt")
        _generate_digits_file(train_path, _TRAIN_LINE_COUNT, _LINE_MAX_LENGTH)
        _generate_digits_file(dev_path, _DEV_LINE_COUNT, _LINE_MAX_LENGTH)

        # Train model
        model_path = os.path.join(work_dir, "model")
        params = "{} {} {}".format(sockeye.train.__file__,
                                   _TRAIN_PARAMS_COMMON.format(train_corpus=train_path,
                                                               dev_corpus=dev_path,
                                                               model=model_path,
                                                               max_len=_LINE_MAX_LENGTH + 1),
                                   train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()

        # Translate corpus
        out_path = os.path.join(work_dir, "out.txt")
        params = "{} {} {}".format(sockeye.translate.__file__,
                                   _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                   input=dev_path,
                                                                   output=out_path),
                                   translate_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        # Measure perplexity
        checkpoints = sockeye.utils.read_metrics_points(path=os.path.join(model_path, C.METRICS_NAME),
                                                        model_path=model_path,
                                                        metric=C.PERPLEXITY)
        perplexity = checkpoints[-1][0]
        assert perplexity <= perplexity_thresh

        # Measure BLEU
        bleu = sockeye.bleu.corpus_bleu(open(out_path, "r").readlines(),
                                        open(dev_path, "r").readlines())
        assert bleu >= bleu_thresh

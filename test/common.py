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
import random
import sys
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
from unittest.mock import patch

import mxnet as mx
import numpy as np

import sockeye.average
import sockeye.bleu
import sockeye.constants as C
import sockeye.evaluate
import sockeye.train
import sockeye.translate
import sockeye.utils


def gaussian_vector(shape, return_symbol=False):
    """
    Generates random normal tensors (diagonal covariance)

    :param shape: shape of the tensor.
    :param return_symbol: True if the result should be a Symbol, False if it should be an Numpy array.
    :return: A gaussian tensor.
    """
    return mx.sym.random_normal(shape=shape) if return_symbol else np.random.normal(size=shape)


def integer_vector(shape, max_value, return_symbol=False):
    """
    Generates a random positive integer tensor

    :param shape: shape of the tensor.
    :param max_value: maximum integer value.
    :param return_symbol: True if the result should be a Symbol, False if it should be an Numpy array.
    :return: A random integer tensor.
    """
    return mx.sym.round(mx.sym.random_uniform(shape=shape) * max_value) if return_symbol \
        else np.round(np.random.uniform(size=shape) * max_value)


def uniform_vector(shape, min_value=0, max_value=1, return_symbol=False):
    """
    Generates a uniformly random tensor

    :param shape: shape of the tensor
    :param min_value: minimum possible value
    :param max_value: maximum possible value (exclusive)
    :param return_symbol: True if the result should be a mx.sym.Symbol, False if it should be a Numpy array
    :return:
    """
    return mx.sym.random_uniform(low=min_value, high=max_value, shape=shape) if return_symbol \
        else np.random.uniform(low=min_value, high=max_value, size=shape)


def generate_random_sentence(vocab_size, max_len):
    """
    Generates a random "sentence" as a list of integers.

    :param vocab_size: Number of words in the "vocabulary". Note that due to
                       the inclusion of special words (BOS, EOS, UNK) this does *not*
                       correspond to the maximum possible value.
    :param max_len: maximum sentence length.
    """
    length = random.randint(1, max_len)
    # Due to the special words, the actual words start at index 3 and go up to vocab_size+2
    return [random.randint(3, vocab_size + 2) for _ in range(length)]


_DIGITS = "0123456789"


def generate_digits_file(source_path: str,
                         target_path: str,
                         line_count: int = 100,
                         line_length: int = 9,
                         sort_target: bool = False,
                         seed=13):
    random_gen = random.Random(seed)
    with open(source_path, "w") as source_out, open(target_path, "w") as target_out:
        for _ in range(line_count):
            digits = [random_gen.choice(_DIGITS) for _ in range(random_gen.randint(1, line_length))]
            print(" ".join(digits), file=source_out)
            if sort_target:
                digits.sort()
            print(" ".join(digits), file=target_out)


_TRAIN_PARAMS_COMMON = "--use-cpu --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --validation-source {dev_source} --validation-target {dev_target} --output {model}"

_TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output}"

_EVAL_PARAMS_COMMON = "--hypotheses {hypotheses} --references {references}"


def run_train_translate(train_params: str,
                        translate_params: str,
                        train_source_path: str,
                        train_target_path: str,
                        dev_source_path: str,
                        dev_target_path: str,
                        max_seq_len: int = 10,
                        work_dir: Optional[str] = None) -> Tuple[float, float]:
    """
    Train a model and translate a dev set.  Report validation perplexity and BLEU.

    :param train_params: Command line args for model training.
    :param translate_params: Command line args for translation.
    :param train_source_path: Path to the source file.
    :param train_target_path: Path to the target file.
    :param dev_source_path: Path to the development source file.
    :param dev_target_path: Path to the development target file.
    :param max_seq_len: The maximum sequence length.
    :param work_dir: The directory to store the model and other outputs in.
    :return: A tuple containing perplexity and bleu.
    """
    with TemporaryDirectory(dir=work_dir, prefix="test_train_translate.") as work_dir:
        # Train model
        model_path = os.path.join(work_dir, "model")
        params = "{} {} {}".format(sockeye.train.__file__,
                                   _TRAIN_PARAMS_COMMON.format(train_source=train_source_path,
                                                               train_target=train_target_path,
                                                               dev_source=dev_source_path,
                                                               dev_target=dev_target_path,
                                                               model=model_path,
                                                               max_len=max_seq_len),
                                   train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()

        # Translate corpus
        out_path = os.path.join(work_dir, "out.txt")
        params = "{} {} {}".format(sockeye.translate.__file__,
                                   _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                   input=dev_source_path,
                                                                   output=out_path),
                                   translate_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        # test averaging
        points = sockeye.average.find_checkpoints(model_path=model_path,
                                                  size=1,
                                                  strategy='best',
                                                  metric=C.PERPLEXITY)
        assert len(points) > 0
        averaged_params = sockeye.average.average(points)
        assert averaged_params

        # get last validation perplexity
        metrics = sockeye.utils.read_metrics_file(path=os.path.join(model_path, C.METRICS_NAME))
        perplexity = metrics[-1][C.PERPLEXITY + '-val']

        # Measure BLEU
        bleu = sockeye.bleu.corpus_bleu(open(out_path, "r").readlines(),
                                        open(dev_target_path, "r").readlines())

        # Run BLEU cli
        eval_params = "{} {} ".format(sockeye.evaluate.__file__,
                                      _EVAL_PARAMS_COMMON.format(hypotheses=out_path, references=dev_target_path), )
        with patch.object(sys, "argv", eval_params.split()):
            sockeye.evaluate.main()

        return perplexity, bleu

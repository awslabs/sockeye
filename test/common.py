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
import logging
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
from unittest.mock import patch

import mxnet as mx
import numpy as np

import sockeye.average
import sockeye.constants as C
import sockeye.evaluate
import sockeye.lexicon
import sockeye.train
import sockeye.translate
import sockeye.utils

from sockeye.evaluate import raw_corpus_bleu
from sockeye.chrf import corpus_chrf

logger = logging.getLogger(__name__)


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


def generate_fast_align_lex(lex_path: str):
    """
    Generate a fast_align format lex table for digits.

    :param lex_path: Path to write lex table.
    """
    with open(lex_path, "w") as lex_out:
        for digit in _DIGITS:
            print("{0}\t{0}\t0".format(digit), file=lex_out)


_LEXICON_PARAMS_COMMON = "-i {input} -m {model} -k 1 -o {json} {quiet}"


@contextmanager
def tmp_digits_dataset(prefix: str,
                       train_line_count: int, train_max_length: int,
                       dev_line_count: int, dev_max_length: int,
                       sort_target: bool = False,
                       seed_train: int = 13, seed_dev: int = 13):
    with TemporaryDirectory(prefix=prefix) as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        generate_digits_file(train_source_path, train_target_path, train_line_count, train_max_length,
                             sort_target=sort_target, seed=seed_train)
        generate_digits_file(dev_source_path, dev_target_path, dev_line_count, dev_max_length, sort_target=sort_target,
                             seed=seed_dev)
        data = {'work_dir': work_dir,
                'source': train_source_path,
                'target': train_target_path,
                'validation_source': dev_source_path,
                'validation_target': dev_target_path}
        yield data


_TRAIN_PARAMS_COMMON = "--use-cpu --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --validation-source {dev_source} --validation-target {dev_target} --output {model} {quiet}"

_TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output} {quiet}"

_TRANSLATE_PARAMS_RESTRICT = "--restrict-lexicon {json}"

_EVAL_PARAMS_COMMON = "--hypotheses {hypotheses} --references {references} --metrics {metrics} {quiet}"


def run_train_translate(train_params: str,
                        translate_params: str,
                        translate_params_equiv: Optional[str],
                        train_source_path: str,
                        train_target_path: str,
                        dev_source_path: str,
                        dev_target_path: str,
                        max_seq_len: int = 10,
                        restrict_lexicon: bool = False,
                        work_dir: Optional[str] = None,
                        quiet: bool = False) -> Tuple[float, float, float, float]:
    """
    Train a model and translate a dev set.  Report validation perplexity and BLEU.

    :param train_params: Command line args for model training.
    :param translate_params: First command line args for translation.
    :param translate_params_equiv: Second command line args for translation. Should produce the same outputs
    :param train_source_path: Path to the source file.
    :param train_target_path: Path to the target file.
    :param dev_source_path: Path to the development source file.
    :param dev_target_path: Path to the development target file.
    :param max_seq_len: The maximum sequence length.
    :param restrict_lexicon: Additional translation run with top-k lexicon-based vocabulary restriction.
    :param work_dir: The directory to store the model and other outputs in.
    :param quiet: Suppress the console output of training and decoding.
    :return: A tuple containing perplexity, bleu scores for standard and reduced vocab decoding, chrf score.
    """
    if quiet:
        quiet_arg = "--quiet"
    else:
        quiet_arg = ""
    with TemporaryDirectory(dir=work_dir, prefix="test_train_translate.") as work_dir:
        # Train model
        model_path = os.path.join(work_dir, "model")
        params = "{} {} {}".format(sockeye.train.__file__,
                                   _TRAIN_PARAMS_COMMON.format(train_source=train_source_path,
                                                               train_target=train_target_path,
                                                               dev_source=dev_source_path,
                                                               dev_target=dev_target_path,
                                                               model=model_path,
                                                               max_len=max_seq_len,
                                                               quiet=quiet_arg),
                                   train_params)
        logger.info("Starting training with parameters %s.", train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()

        logger.info("Translating with parameters %s.", translate_params)
        # Translate corpus with the 1st params
        out_path = os.path.join(work_dir, "out.txt")
        params = "{} {} {}".format(sockeye.translate.__file__,
                                   _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                   input=dev_source_path,
                                                                   output=out_path,
                                                                   quiet=quiet_arg),
                                   translate_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        # Translate corpus with the 2nd params
        if translate_params_equiv is not None:
            out_path_equiv = os.path.join(work_dir, "out_equiv.txt")
            params = "{} {} {}".format(sockeye.translate.__file__,
                                       _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                       input=dev_source_path,
                                                                       output=out_path_equiv,
                                                                       quiet=quiet_arg),
                                       translate_params_equiv)
            with patch.object(sys, "argv", params.split()):
                sockeye.translate.main()

            # read-in both outputs, ensure they are the same
            with open(out_path, 'rt') as f:
                lines = f.readlines()
            with open(out_path_equiv, 'rt') as f:
                lines_equiv = f.readlines()
            assert all(a == b for a, b in zip(lines, lines_equiv))

        # Test restrict-lexicon
        out_restrict_path = os.path.join(work_dir, "out-restrict.txt")
        if restrict_lexicon:
            # fast_align lex table
            lex_path = os.path.join(work_dir, "lex")
            generate_fast_align_lex(lex_path)
            # Top-K JSON
            json_path = os.path.join(work_dir, "json")
            params = "{} {}".format(sockeye.lexicon.__file__,
                                    _LEXICON_PARAMS_COMMON.format(input=lex_path,
                                                                  model=model_path,
                                                                  json=json_path,
                                                                  quiet=quiet_arg))
            with patch.object(sys, "argv", params.split()):
                sockeye.lexicon.main()
            # Translate corpus with restrict-lexicon
            params = "{} {} {} {}".format(sockeye.translate.__file__,
                                          _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                          input=dev_source_path,
                                                                          output=out_restrict_path,
                                                                          quiet=quiet_arg),
                                          translate_params,
                                          _TRANSLATE_PARAMS_RESTRICT.format(json=json_path))
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

        # get best validation perplexity
        metrics = sockeye.utils.read_metrics_file(path=os.path.join(model_path, C.METRICS_NAME))
        perplexity = min(m[C.PERPLEXITY + '-val'] for m in metrics)

        hypotheses = open(out_path, "r").readlines()
        references = open(dev_target_path, "r").readlines()

        # compute metrics
        bleu = raw_corpus_bleu(hypotheses=hypotheses, references=references, offset=0.01)
        chrf = corpus_chrf(hypotheses=hypotheses, references=references)

        bleu_restrict = None
        if restrict_lexicon:
            bleu_restrict = raw_corpus_bleu(hypotheses=hypotheses, references=references, offset=0.01)

        # Run BLEU cli
        eval_params = "{} {} ".format(sockeye.evaluate.__file__,
                                      _EVAL_PARAMS_COMMON.format(hypotheses=out_path,
                                                                 references=dev_target_path,
                                                                 metrics="bleu chrf",
                                                                 quiet=quiet_arg))
        with patch.object(sys, "argv", eval_params.split()):
            sockeye.evaluate.main()

        return perplexity, bleu, bleu_restrict, chrf

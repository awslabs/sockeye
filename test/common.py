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
import os
import random
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from unittest.mock import patch

import mxnet as mx
import numpy as np

import sockeye.average
import sockeye.checkpoint_decoder
import sockeye.constants as C
import sockeye.evaluate
import sockeye.lexicon
import sockeye.prepare_data
import sockeye.train
import sockeye.translate
import sockeye.utils
from sockeye.evaluate import raw_corpus_bleu, raw_corpus_chrf

logger = logging.getLogger(__name__)


def gaussian_vector(shape, return_symbol=False):
    """
    Generates random normal tensors (diagonal covariance)

    :param shape: shape of the tensor.
    :param return_symbol: True if the result should be a Symbol, False if it should be an Numpy array.
    :return: A gaussian tensor.
    """
    return mx.sym.random_normal(shape=shape) if return_symbol else np.random.normal(size=shape)


def integer_vector(shape, max_value, min_value=1, return_symbol=False):
    """
    Generates a random positive integer tensor

    :param shape: shape of the tensor.
    :param max_value: maximum integer value.
    :param min_value: minimum integer value.
    :param return_symbol: True if the result should be a Symbol, False if it should be an Numpy array.
    :return: A random integer tensor.
    """
    return mx.sym.round(mx.sym.random.uniform(low=min_value, high=max_value, shape=shape)) if return_symbol \
        else np.random.randint(low=min_value, high=max_value, size=shape)


def uniform_vector(shape, min_value=0, max_value=1, return_symbol=False):
    """
    Generates a uniformly random tensor

    :param shape: shape of the tensor
    :param min_value: minimum possible value
    :param max_value: maximum possible value (exclusive)
    :param return_symbol: True if the result should be a mx.sym.Symbol, False if it should be a Numpy array
    :return:
    """
    return mx.sym.random.uniform(low=min_value, high=max_value, shape=shape) if return_symbol \
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
_MID = 5


def generate_digits_file(source_path: str,
                         target_path: str,
                         line_count: int = 100,
                         line_length: int = 9,
                         sort_target: bool = False,
                         line_count_empty: int = 0,
                         seed=13):
    assert line_count_empty <= line_count
    random_gen = random.Random(seed)
    with open(source_path, "w") as source_out, open(target_path, "w") as target_out:
        all_digits = []
        for _ in range(line_count - line_count_empty):
            digits = [random_gen.choice(_DIGITS) for _ in range(random_gen.randint(1, line_length))]
            all_digits.append(digits)
        for _ in range(line_count_empty):
            all_digits.append([])
        random_gen.shuffle(all_digits)
        for digits in all_digits:
            print(" ".join(digits), file=source_out)
            if sort_target:
                digits.sort()
            print(" ".join(digits), file=target_out)


def generate_low_high_factors(source_path: str,
                              output_path: str):
    """
    Writes low/high factor file given a source file of digit sequences.
    """
    with open(source_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            digits = map(int, line.rstrip().split())
            factors = ["l" if digit < _MID else "h" for digit in digits]
            print(" ".join(factors), file=fout)


def generate_fast_align_lex(lex_path: str):
    """
    Generate a fast_align format lex table for digits.

    :param lex_path: Path to write lex table.
    """
    with open(lex_path, "w") as lex_out:
        for digit in _DIGITS:
            print("{0}\t{0}\t0".format(digit), file=lex_out)


_LEXICON_CREATE_PARAMS_COMMON = "create -i {input} -m {model} -k {topk} -o {lexicon} {quiet}"


@contextmanager
def tmp_digits_dataset(prefix: str,
                       train_line_count: int, train_max_length: int,
                       dev_line_count: int, dev_max_length: int,
                       test_line_count: int, test_line_count_empty: int, test_max_length: int,
                       sort_target: bool = False,
                       seed_train: int = 13, seed_dev: int = 13,
                       with_source_factors: bool = False):
    with TemporaryDirectory(prefix=prefix) as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        test_source_path = os.path.join(work_dir, "test.src")
        test_target_path = os.path.join(work_dir, "test.tgt")
        generate_digits_file(train_source_path, train_target_path, train_line_count,
                             train_max_length, sort_target=sort_target, seed=seed_train)
        generate_digits_file(dev_source_path, dev_target_path, dev_line_count, dev_max_length, sort_target=sort_target,
                             seed=seed_dev)
        generate_digits_file(test_source_path, test_target_path, test_line_count, test_max_length,
                             line_count_empty=test_line_count_empty, sort_target=sort_target, seed=seed_dev)
        data = {'work_dir': work_dir,
                'source': train_source_path,
                'target': train_target_path,
                'validation_source': dev_source_path,
                'validation_target': dev_target_path,
                'test_source': test_source_path,
                'test_target': test_target_path}

        if with_source_factors:
            train_factor_path = train_source_path + ".factors"
            dev_factor_path = dev_source_path + ".factors"
            test_factor_path = test_source_path + ".factors"
            generate_low_high_factors(train_source_path, train_factor_path)
            generate_low_high_factors(dev_source_path, dev_factor_path)
            generate_low_high_factors(test_source_path, test_factor_path)
            data['train_source_factors'] = [train_factor_path]
            data['dev_source_factors'] = [dev_factor_path]
            data['test_source_factors'] = [test_factor_path]

        yield data


_TRAIN_PARAMS_COMMON = "--use-cpu --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --validation-source {dev_source} --validation-target {dev_target} --output {model} {quiet}" \
                       " --seed {seed}"

_PREPARE_DATA_COMMON = " --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --output {output} {quiet}"

_TRAIN_WITH_FACTORS_COMMON = " --source-factors {source_factors}"
_DEV_WITH_FACTORS_COMMON = " --validation-source-factors {dev_source_factors}"

_TRAIN_PARAMS_PREPARED_DATA_COMMON = "--use-cpu --max-seq-len {max_len} --prepared-data {prepared_data}" \
                                     " --validation-source {dev_source} --validation-target {dev_target} " \
                                     "--output {model} {quiet}"

_TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output} {quiet}"

_TRANSLATE_WITH_FACTORS_COMMON = " --input-factors {input_factors}"

_TRANSLATE_PARAMS_RESTRICT = "--restrict-lexicon {lexicon} --restrict-lexicon-topk {topk}"

_EVAL_PARAMS_COMMON = "--hypotheses {hypotheses} --references {references} --metrics {metrics} {quiet}"


def run_train_translate(train_params: str,
                        translate_params: str,
                        translate_params_equiv: Optional[str],
                        train_source_path: str,
                        train_target_path: str,
                        dev_source_path: str,
                        dev_target_path: str,
                        test_source_path: str,
                        test_target_path: str,
                        train_source_factor_paths: Optional[List[str]] = None,
                        dev_source_factor_paths: Optional[List[str]] = None,
                        test_source_factor_paths: Optional[List[str]] = None,
                        use_prepared_data: bool = False,
                        max_seq_len: int = 10,
                        restrict_lexicon: bool = False,
                        work_dir: Optional[str] = None,
                        seed: int = 13,
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
    :param test_source_path: Path to the test source file.
    :param test_target_path: Path to the test target file.
    :param train_source_factor_paths: Optional list of paths to training source factor files.
    :param dev_source_factor_paths: Optional list of paths to dev source factor files.
    :param test_source_factor_paths: Optional list of paths to test source factor files.
    :param use_prepared_data: Whether to use the prepared data functionality.
    :param max_seq_len: The maximum sequence length.
    :param restrict_lexicon: Additional translation run with top-k lexicon-based vocabulary restriction.
    :param work_dir: The directory to store the model and other outputs in.
    :param seed: The seed used for training.
    :param quiet: Suppress the console output of training and decoding.
    :return: A tuple containing perplexity, bleu scores for standard and reduced vocab decoding, chrf score.
    """
    if quiet:
        quiet_arg = "--quiet"
    else:
        quiet_arg = ""
    with TemporaryDirectory(dir=work_dir, prefix="test_train_translate.") as work_dir:
        # Optionally create prepared data directory
        if use_prepared_data:
            prepared_data_path = os.path.join(work_dir, "prepared_data")
            params = "{} {}".format(sockeye.prepare_data.__file__,
                                    _PREPARE_DATA_COMMON.format(train_source=train_source_path,
                                                                train_target=train_target_path,
                                                                output=prepared_data_path,
                                                                max_len=max_seq_len,
                                                                quiet=quiet_arg))
            if train_source_factor_paths is not None:
                params += _TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(train_source_factor_paths))

            logger.info("Creating prepared data folder.")
            with patch.object(sys, "argv", params.split()):
                sockeye.prepare_data.main()
            # Train model
            model_path = os.path.join(work_dir, "model")
            params = "{} {} {}".format(sockeye.train.__file__,
                                       _TRAIN_PARAMS_PREPARED_DATA_COMMON.format(prepared_data=prepared_data_path,
                                                                                 dev_source=dev_source_path,
                                                                                 dev_target=dev_target_path,
                                                                                 model=model_path,
                                                                                 max_len=max_seq_len,
                                                                                 quiet=quiet_arg),
                                       train_params)

            if dev_source_factor_paths is not None:
                params += _DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(dev_source_factor_paths))

            logger.info("Starting training with parameters %s.", train_params)
            with patch.object(sys, "argv", params.split()):
                sockeye.train.main()
        else:
            # Train model
            model_path = os.path.join(work_dir, "model")
            params = "{} {} {}".format(sockeye.train.__file__,
                                       _TRAIN_PARAMS_COMMON.format(train_source=train_source_path,
                                                                   train_target=train_target_path,
                                                                   dev_source=dev_source_path,
                                                                   dev_target=dev_target_path,
                                                                   model=model_path,
                                                                   max_len=max_seq_len,
                                                                   seed=seed,
                                                                   quiet=quiet_arg),
                                       train_params)

            if train_source_factor_paths is not None:
                params += _TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(train_source_factor_paths))
            if dev_source_factor_paths is not None:
                params += _DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(dev_source_factor_paths))

            logger.info("Starting training with parameters %s.", train_params)
            with patch.object(sys, "argv", params.split()):
                sockeye.train.main()

        # run checkpoint decoder on 1% of dev data
        with open(dev_source_path) as dev_fd:
            num_dev_sent = sum(1 for _ in dev_fd)
        sample_size = min(1, int(num_dev_sent * 0.01))
        cp_decoder = sockeye.checkpoint_decoder.CheckpointDecoder(context=mx.cpu(),
                                                                  inputs=[dev_source_path],
                                                                  references=dev_target_path,
                                                                  model=model_path,
                                                                  sample_size=sample_size,
                                                                  batch_size=2,
                                                                  beam_size=2)
        cp_metrics = cp_decoder.decode_and_evaluate()
        logger.info("Checkpoint decoder metrics: %s", cp_metrics)

        logger.info("Translating with parameters %s.", translate_params)
        # Translate corpus with the 1st params
        out_path = os.path.join(work_dir, "out.txt")
        params = "{} {} {}".format(sockeye.translate.__file__,
                                   _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                   input=test_source_path,
                                                                   output=out_path,
                                                                   quiet=quiet_arg),
                                   translate_params)

        if test_source_factor_paths is not None:
            params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))

        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        # Translate corpus with the 2nd params
        if translate_params_equiv is not None:
            out_path_equiv = os.path.join(work_dir, "out_equiv.txt")
            params = "{} {} {}".format(sockeye.translate.__file__,
                                       _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                       input=test_source_path,
                                                                       output=out_path_equiv,
                                                                       quiet=quiet_arg),
                                       translate_params_equiv)

            if test_source_factor_paths is not None:
                params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))

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
            ttable_path = os.path.join(work_dir, "ttable")
            generate_fast_align_lex(ttable_path)
            # Top-K lexicon
            lexicon_path = os.path.join(work_dir, "lexicon")
            params = "{} {}".format(sockeye.lexicon.__file__,
                                    _LEXICON_CREATE_PARAMS_COMMON.format(input=ttable_path,
                                                                         model=model_path,
                                                                         topk=20,
                                                                         lexicon=lexicon_path,
                                                                         quiet=quiet_arg))
            with patch.object(sys, "argv", params.split()):
                sockeye.lexicon.main()
            # Translate corpus with restrict-lexicon
            params = "{} {} {} {}".format(sockeye.translate.__file__,
                                          _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                          input=test_source_path,
                                                                          output=out_restrict_path,
                                                                          quiet=quiet_arg),
                                          translate_params,
                                          _TRANSLATE_PARAMS_RESTRICT.format(lexicon=lexicon_path, topk=1))

            if test_source_factor_paths is not None:
                params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))

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

        with open(out_path, "r") as out:
            hypotheses = out.readlines()
        with open(test_target_path, "r") as ref:
            references = ref.readlines()
        assert len(hypotheses) == len(references)

        # compute metrics
        bleu = raw_corpus_bleu(hypotheses=hypotheses, references=references, offset=0.01)
        chrf = raw_corpus_chrf(hypotheses=hypotheses, references=references)

        bleu_restrict = None
        if restrict_lexicon:
            bleu_restrict = raw_corpus_bleu(hypotheses=hypotheses, references=references, offset=0.01)

        # Run BLEU cli
        eval_params = "{} {} ".format(sockeye.evaluate.__file__,
                                      _EVAL_PARAMS_COMMON.format(hypotheses=out_path,
                                                                 references=test_target_path,
                                                                 metrics="bleu chrf",
                                                                 quiet=quiet_arg), )
        with patch.object(sys, "argv", eval_params.split()):
            sockeye.evaluate.main()

        return perplexity, bleu, bleu_restrict, chrf

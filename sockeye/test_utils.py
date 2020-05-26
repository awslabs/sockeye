# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np

import sockeye.average
import sockeye.checkpoint_decoder
import sockeye.evaluate
import sockeye.extract_parameters
import sockeye.lexicon
import sockeye.model
import sockeye.prepare_data
import sockeye.score
import sockeye.train
import sockeye.translate
import sockeye.utils

logger = logging.getLogger(__name__)


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
            factors = ("l" if digit < _MID else "h" for digit in digits)
            print(" ".join(factors), file=fout)


def generate_fast_align_lex(lex_path: str):
    """
    Generate a fast_align format lex table for digits.

    :param lex_path: Path to write lex table.
    """
    with open(lex_path, "w") as lex_out:
        for digit in _DIGITS:
            print("{0}\t{0}\t0".format(digit), file=lex_out)


LEXICON_CREATE_PARAMS_COMMON = "create -i {input} -m {model} -k {topk} -o {lexicon}"


@contextmanager
def tmp_digits_dataset(prefix: str,
                       train_line_count: int, train_line_count_empty: int, train_max_length: int,
                       dev_line_count: int, dev_max_length: int,
                       test_line_count: int, test_line_count_empty: int, test_max_length: int,
                       sort_target: bool = False,
                       seed_train: int = 13, seed_dev: int = 13,
                       with_n_source_factors: int = 0) -> Dict[str, Any]:
    """
    Creates a temporary dataset with train, dev, and test. Returns a dictionary with paths to the respective temporary
    files.
    """
    with TemporaryDirectory(prefix=prefix) as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        test_source_path = os.path.join(work_dir, "test.src")
        test_target_path = os.path.join(work_dir, "test.tgt")
        generate_digits_file(train_source_path, train_target_path, train_line_count, train_max_length,
                             line_count_empty=train_line_count_empty, sort_target=sort_target, seed=seed_train)
        generate_digits_file(dev_source_path, dev_target_path, dev_line_count, dev_max_length, sort_target=sort_target,
                             seed=seed_dev)
        generate_digits_file(test_source_path, test_target_path, test_line_count, test_max_length,
                             line_count_empty=test_line_count_empty, sort_target=sort_target, seed=seed_dev)
        data = {'work_dir': work_dir,
                'train_source': train_source_path,
                'train_target': train_target_path,
                'dev_source': dev_source_path,
                'dev_target': dev_target_path,
                'test_source': test_source_path,
                'test_target': test_target_path}

        if with_n_source_factors > 0:
            data['train_source_factors'] = []
            data['dev_source_factors'] = []
            data['test_source_factors'] = []
            for i in range(with_n_source_factors):
                train_factor_path = train_source_path + ".factors%d" % i
                dev_factor_path = dev_source_path + ".factors%d" % i
                test_factor_path = test_source_path + ".factors%d" % i
                generate_low_high_factors(train_source_path, train_factor_path)
                generate_low_high_factors(dev_source_path, dev_factor_path)
                generate_low_high_factors(test_source_path, test_factor_path)
                data['train_source_factors'].append(train_factor_path)
                data['dev_source_factors'].append(dev_factor_path)
                data['test_source_factors'].append(test_factor_path)

        yield data


TRAIN_PARAMS_COMMON = "--use-cpu --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --validation-source {dev_source} --validation-target {dev_target} --output {model}" \
                       " --seed {seed}"

PREPARE_DATA_COMMON = " --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --output {output} --pad-vocab-to-multiple-of 16"

TRAIN_WITH_FACTORS_COMMON = " --source-factors {source_factors}"
DEV_WITH_FACTORS_COMMON = " --validation-source-factors {dev_source_factors}"

TRAIN_PARAMS_PREPARED_DATA_COMMON = "--use-cpu --max-seq-len {max_len} --prepared-data {prepared_data}" \
                                     " --validation-source {dev_source} --validation-target {dev_target} " \
                                     "--output {model}"

TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output} " \
                           "--output-type translation_with_score"

TRANSLATE_WITH_FACTORS_COMMON = " --input-factors {input_factors}"

TRANSLATE_PARAMS_RESTRICT = "--restrict-lexicon {lexicon} --restrict-lexicon-topk {topk}"

SCORE_PARAMS_COMMON = "--use-cpu --model {model} --source {source} --target {target} --output {output}"

SCORE_WITH_FACTORS_COMMON = " --source-factors {source_factors}"


def run_train_translate(train_params: str,
                        translate_params: str,
                        data: Dict[str, Any],
                        use_prepared_data: bool = False,
                        max_seq_len: int = 10,
                        seed: int = 13) -> Dict[str, Any]:
    """
    Train a model and translate a test set. Returns the updated data dictionary containing paths to translation outputs
    and scores.

    :param train_params: Command line args for model training.
    :param translate_params: First command line args for translation.
    :param data: Dictionary containing test data
    :param use_prepared_data: Whether to use the prepared data functionality.
    :param max_seq_len: The maximum sequence length.
    :param seed: The seed used for training.
    :return: Data dictionary, updated with translation outputs and scores
    """
    work_dir = os.path.join(data['work_dir'], 'train_translate')
    data['model'] = os.path.join(work_dir, "model")
    # Optionally create prepared data directory
    if use_prepared_data:
        data['train_prepared'] = os.path.join(work_dir, "prepared_data")
        prepare_params = "{} {}".format(sockeye.prepare_data.__file__,
                                        PREPARE_DATA_COMMON.format(train_source=data['train_source'],
                                                                   train_target=data['train_target'],
                                                                   output=data['train_prepared'],
                                                                   max_len=max_seq_len))
        if 'train_source_factors' in data:
            prepare_params += TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(data['train_source_factors']))

        if '--weight-tying-type src_trg' in train_params:
            prepare_params += ' --shared-vocab'

        logger.info("Preparing data with parameters %s.", prepare_params)
        with patch.object(sys, "argv", prepare_params.split()):
            sockeye.prepare_data.main()
        # Train model
        params = "{} {} {}".format(sockeye.train.__file__,
                                   TRAIN_PARAMS_PREPARED_DATA_COMMON.format(prepared_data=data['train_prepared'],
                                                                            dev_source=data['dev_source'],
                                                                            dev_target=data['dev_target'],
                                                                            model=data['model'],
                                                                            max_len=max_seq_len),
                                   train_params)

        if 'dev_source_factors' in data:
            params += DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(data['dev_source_factors']))

        logger.info("Starting training with parameters %s.", train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()
    else:
        # Train model
        params = "{} {} {}".format(sockeye.train.__file__,
                                   TRAIN_PARAMS_COMMON.format(train_source=data['train_source'],
                                                              train_target=data['train_target'],
                                                              dev_source=data['dev_source'],
                                                              dev_target=data['dev_target'],
                                                              model=data['model'],
                                                              max_len=max_seq_len,
                                                              seed=seed),
                                   train_params)

        if 'train_source_factors' in data:
            params += TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(data['train_source_factors']))
        if 'dev_source_factors' in data:
            params += DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(data['dev_source_factors']))

        logger.info("Starting training with parameters %s.", train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()

    # create Top-K lexicon from simple ttable mapping digit to digit
    ttable_path = os.path.join(data['work_dir'], "ttable")
    generate_fast_align_lex(ttable_path)
    lexicon_path = os.path.join(data['work_dir'], "lexicon")
    params = "{} {}".format(sockeye.lexicon.__file__,
                            LEXICON_CREATE_PARAMS_COMMON.format(input=ttable_path,
                                                                model=data['model'],
                                                                topk=20,
                                                                lexicon=lexicon_path))
    with patch.object(sys, "argv", params.split()):
        sockeye.lexicon.main()
    data['lexicon'] = lexicon_path

    # Translate corpus with the 1st params and scoring output handler to obtain scores
    data['test_output'] = os.path.join(work_dir, "test.out")
    params = "{} {} {}".format(sockeye.translate.__file__,
                               TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                                              input=data['test_source'],
                                                              output=data['test_output']),
                               translate_params)

    if 'test_source_factors' in data:
        params += TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(data['test_source_factors']))

    logger.info("Translating with params %s", params)
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()

    # Collect test inputs
    with open(data['test_source']) as inputs:
        data['test_inputs'] = [line.strip() for line in inputs]

    # Collect test references
    with open(data['test_target'], "r") as ref:
        data['test_targets'] = [line.strip() for line in ref]

    # Collect test translate outputs and scores
    data['test_outputs'], data['test_scores'] = collect_translate_output_and_scores(data['test_output'])
    assert len(data['test_inputs']) == len(data['test_targets']) == len(data['test_outputs']) == len(data['test_scores'])
    return data


def run_translate_restrict(data: Dict[str, Any], translate_params: str) -> Dict[str, Any]:
    """
    Runs sockeye.translate with vocabulary selection and checks if number of outputs are the same as without
    vocabulary selection. Adds restricted outputs and scores to the data dictionary.
    """
    out_path = os.path.join(data['work_dir'], "out-restrict.txt")
    # Translate corpus with restrict-lexicon
    params = "{} {} {} {}".format(sockeye.translate.__file__,
                                  TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                                                 input=data['test_source'],
                                                                 output=out_path),
                                  translate_params,
                                  TRANSLATE_PARAMS_RESTRICT.format(lexicon=data['lexicon'], topk=1))
    if 'test_source_factors' in data:
        params += TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(data['test_source_factors']))
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()

    # Collect test translate outputs and scores
    data['test_outputs_restricted'], data['test_scores_restricted'] = collect_translate_output_and_scores(out_path)
    assert len(data['test_outputs_restricted']) == len(data['test_outputs'])
    return data


def create_reference_constraints(translate_inputs: List[str], translate_outputs: List[str]) -> List[Dict[str, Any]]:
    constrained_inputs = []
    for sentno, (source, translate_output) in enumerate(zip(translate_inputs, translate_outputs)):
        constrained_inputs.append(json.dumps({'text': source, 'constraints': ['<s> {} </s>'.format(translate_output)]},
                                             ensure_ascii=False))
    return constrained_inputs


def collect_translate_output_and_scores(out_path: str) -> Tuple[List[str], List[float]]:
    """
    Collects translation outputs and scores from an output file
    produced with the 'translation_and_score' or nbest output handler.
    """
    logger.debug("collect_translate_output_and_scores(%s)", out_path)
    translations = []  # type: List[str]
    scores = []  # type: List[float]
    with open(out_path) as out_fh:
        for line in out_fh:
            logger.debug(" line: %s", line.strip())
            output = line.strip()
            translation = ''
            score = -np.inf
            try:
                json_output = json.loads(output)
                try:
                    translation = json_output['translation']
                    score = json_output['score']
                except IndexError:
                    pass
            except:
                try:
                    score, translation = output.split('\t', 1)
                except ValueError:
                    pass
            translations.append(translation)
            scores.append(float(score))
    return translations, scores

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

import mxnet as mx
import numpy as np

import sockeye.average
import sockeye.checkpoint_decoder
import sockeye.constants as C
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


_LEXICON_CREATE_PARAMS_COMMON = "create -i {input} -m {model} -k {topk} -o {lexicon}"


@contextmanager
def tmp_digits_dataset(prefix: str,
                       train_line_count: int, train_line_count_empty: int, train_max_length: int,
                       dev_line_count: int, dev_max_length: int,
                       test_line_count: int, test_line_count_empty: int, test_max_length: int,
                       sort_target: bool = False,
                       seed_train: int = 13, seed_dev: int = 13,
                       with_source_factors: bool = False) -> Dict[str, Any]:
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
                       " --validation-source {dev_source} --validation-target {dev_target} --output {model}" \
                       " --seed {seed}"

_PREPARE_DATA_COMMON = " --max-seq-len {max_len} --source {train_source} --target {train_target}" \
                       " --output {output} --pad-vocab-to-multiple-of 16"

_TRAIN_WITH_FACTORS_COMMON = " --source-factors {source_factors}"
_DEV_WITH_FACTORS_COMMON = " --validation-source-factors {dev_source_factors}"

_TRAIN_PARAMS_PREPARED_DATA_COMMON = "--use-cpu --max-seq-len {max_len} --prepared-data {prepared_data}" \
                                     " --validation-source {dev_source} --validation-target {dev_target} " \
                                     "--output {model}"

_TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output} " \
                           "--output-type translation_with_score"

_TRANSLATE_WITH_FACTORS_COMMON = " --input-factors {input_factors}"

_TRANSLATE_PARAMS_RESTRICT = "--restrict-lexicon {lexicon} --restrict-lexicon-topk {topk}"

_SCORE_PARAMS_COMMON = "--use-cpu --model {model} --source {source} --target {target} --output {output}"

_SCORE_WITH_FACTORS_COMMON = " --source-factors {source_factors}"


def check_train_translate(train_params: str,
                          translate_params: str,
                          data: Dict[str, Any],
                          use_prepared_data: bool,
                          max_seq_len: int,
                          compare_translate_vs_scoring_scores: bool = True,
                          seed: int = 13) -> Dict[str, Any]:
    """
    Tests core features (training, inference).
    """
    # train model and translate test set
    data = run_train_translate(train_params=train_params,
                               translate_params=translate_params,
                               data=data,
                               use_prepared_data=use_prepared_data,
                               max_seq_len=max_seq_len,
                               seed=seed)

    # Test equivalence of batch decoding
    translate_params_batch = translate_params + " --batch-size 2"
    test_translate_equivalence(data, translate_params_batch)

    # Run translate with restrict-lexicon
    data = run_translate_restrict(data, translate_params)

    # Test scoring by ensuring that the sockeye.scoring module produces the same scores when scoring the output
    # of sockeye.translate. However, since this training is on very small datasets, the output of sockeye.translate
    # is often pure garbage or empty and cannot be scored. So we only try to score if we have some valid output
    # to work with.
    # Only run scoring under these conditions. Why?
    # - translate splits up too-long sentences and translates them in sequence, invalidating the score, so skip that
    # - scoring requires valid translation output to compare against
    if '--max-input-len' not in translate_params and _translate_output_is_valid(data['test_outputs']):
        test_scoring(data, translate_params, compare_translate_vs_scoring_scores)

    return data


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
        params = "{} {}".format(sockeye.prepare_data.__file__,
                                _PREPARE_DATA_COMMON.format(train_source=data['train_source'],
                                                            train_target=data['train_target'],
                                                            output=data['train_prepared'],
                                                            max_len=max_seq_len))
        if 'train_source_factors' in data:
            params += _TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(data['train_source_factors']))

        logger.info("Creating prepared data folder.")
        with patch.object(sys, "argv", params.split()):
            sockeye.prepare_data.main()
        # Train model
        params = "{} {} {}".format(sockeye.train.__file__,
                                   _TRAIN_PARAMS_PREPARED_DATA_COMMON.format(prepared_data=data['train_prepared'],
                                                                             dev_source=data['dev_source'],
                                                                             dev_target=data['dev_target'],
                                                                             model=data['model'],
                                                                             max_len=max_seq_len),
                                   train_params)

        if 'dev_source_factors' in data:
            params += _DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(data['dev_source_factors']))

        logger.info("Starting training with parameters %s.", train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()
    else:
        # Train model
        params = "{} {} {}".format(sockeye.train.__file__,
                                   _TRAIN_PARAMS_COMMON.format(train_source=data['train_source'],
                                                               train_target=data['train_target'],
                                                               dev_source=data['dev_source'],
                                                               dev_target=data['dev_target'],
                                                               model=data['model'],
                                                               max_len=max_seq_len,
                                                               seed=seed),
                                   train_params)

        if 'train_source_factors' in data:
            params += _TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(data['train_source_factors']))
        if 'dev_source_factors' in data:
            params += _DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(data['dev_source_factors']))

        logger.info("Starting training with parameters %s.", train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.train.main()

    # Translate corpus with the 1st params and scoring output handler to obtain scores
    data['test_output'] = os.path.join(work_dir, "test.out")
    params = "{} {} {}".format(sockeye.translate.__file__,
                               _TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                                               input=data['test_source'],
                                                               output=data['test_output']),
                               translate_params)

    if 'test_source_factors' in data:
        params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(data['test_source_factors']))

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
    # fast_align lex table
    ttable_path = os.path.join(data['work_dir'], "ttable")
    generate_fast_align_lex(ttable_path)
    # Top-K lexicon
    lexicon_path = os.path.join(data['work_dir'], "lexicon")
    params = "{} {}".format(sockeye.lexicon.__file__,
                            _LEXICON_CREATE_PARAMS_COMMON.format(input=ttable_path,
                                                                 model=data['model'],
                                                                 topk=20,
                                                                 lexicon=lexicon_path))
    with patch.object(sys, "argv", params.split()):
        sockeye.lexicon.main()
    # Translate corpus with restrict-lexicon
    params = "{} {} {} {}".format(sockeye.translate.__file__,
                                  _TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                                                  input=data['test_source'],
                                                                  output=out_path),
                                  translate_params,
                                  _TRANSLATE_PARAMS_RESTRICT.format(lexicon=lexicon_path, topk=1))
    if 'test_source_factors' in data:
        params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(data['test_source_factors']))
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()

    # Collect test translate outputs and scores
    data['test_outputs_restricted'], data['test_scores_restricted'] = collect_translate_output_and_scores(out_path)
    assert len(data['test_outputs_restricted']) == len(data['test_outputs'])
    return data


def test_translate_equivalence(data: Dict[str, Any], translate_params_equiv: str):
    """
    Tests whether the output and scores generated by sockeye.translate with translate_params_equiv are equal to
    the previously generated outputs, referenced in the data dictionary.
    """
    out_path = os.path.join(data['work_dir'], "test.out.equiv")
    params = "{} {} {}".format(sockeye.translate.__file__,
                               _TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                                               input=data['test_source'],
                                                               output=out_path),
                               translate_params_equiv)
    if 'test_source_factors' in data:
        params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(data['test_source_factors']))
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()
    # Collect translate outputs and scores
    translate_outputs_equiv, translate_scores_equiv = collect_translate_output_and_scores(out_path)

    assert 'test_outputs' in data and 'test_scores' in data
    assert all(a == b for a, b in zip(data['test_outputs'], translate_outputs_equiv))
    assert all(abs(a - b) < 0.01 or np.isnan(a - b) for a, b in zip(data['test_scores'], translate_scores_equiv))
def _create_reference_constraints(translate_inputs: List[str], translate_outputs: List[str]) -> List[Dict[str, Any]]:
    constrained_inputs = []
    for sentno, (source, translate_output) in enumerate(zip(translate_inputs, translate_outputs)):
        constrained_inputs.append(json.dumps({'text': source, 'constraints': ['<s> {} </s>'.format(translate_output)]}, ensure_ascii=False))
    return constrained_inputs


def test_constrained_decoding_against_ref(data: Dict[str, Any], translate_params: str):
    constrained_inputs = _create_reference_constraints(data['test_inputs'], data['test_outputs'])
    new_test_source_path = os.path.join(data['work_dir'], "test_constrained.txt")
    with open(new_test_source_path, 'w') as out:
        for json_line in constrained_inputs:
            print(json_line, file=out)
    out_path_constrained = os.path.join(data['work_dir'], "out_constrained.txt")
    params = "{} {} {} --json-input --output-type translation_with_score --beam-size 1 --batch-size 1 --nbest-size 1".format(
        sockeye.translate.__file__,
        _TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                        input=new_test_source_path,
                                        output=out_path_constrained),
        translate_params)
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()
    constrained_outputs, constrained_scores = collect_translate_output_and_scores(out_path_constrained)
    assert len(constrained_outputs) == len(data['test_outputs']) == len(constrained_inputs)
    for json_input, constrained_out, unconstrained_out in zip(constrained_inputs, constrained_outputs, data['test_outputs']):
        # Make sure the constrained output is the same as we got when decoding unconstrained
        assert constrained_out == unconstrained_out

    data['test_constrained_inputs'] = constrained_inputs
    data['test_constrained_outputs'] = constrained_outputs
    data['test_constrained_scores'] = constrained_scores
    return data


def test_scoring(data: Dict[str, Any], translate_params: str, test_similar_scores: bool):
    """
    Tests the scoring CLI and checks for score equivalence with previously generated translate scores.
    """
    # Translate params that affect the score need to be used for scoring as well.
    # Currently, the only relevant flag passed is the --softmax-temperature flag.
    relevant_params = {'--softmax-temperature'}
    score_params = ''
    params = translate_params.split()
    for i, param in enumerate(params):
        if param in relevant_params:
            score_params = '{} {}'.format(param, params[i + 1])
    out_path = os.path.join(data['work_dir'], "score.out")

    # write translate outputs as target file for scoring and collect tokens
    target_path = os.path.join(data['work_dir'], "score.target")
    translate_tokens = []
    with open(target_path, 'w') as target_out:
        for output in data['test_outputs']:
            print(output, file=target_out)
            translate_tokens.append(output.split())

    params = "{} {} {}".format(sockeye.score.__file__,
                               _SCORE_PARAMS_COMMON.format(model=data['model'],
                                                           source=data['test_source'],
                                                           target=target_path,
                                                           output=out_path),
                               score_params)
    if 'test_source_factors' in data:
        params += _SCORE_WITH_FACTORS_COMMON.format(source_factors=" ".join(data['test_source_factors']))
    logger.info("Scoring with params %s", params)
    with patch.object(sys, "argv", params.split()):
        sockeye.score.main()

    # Collect scores from output file
    with open(out_path) as score_out:
        score_scores = [float(line.strip()) for line in score_out]

    # Compare scored output to original translation output. Unfortunately, sockeye.translate doesn't enforce
    # generation of </s> and have had length normalization applied. So, skip all sentences that are as long
    # as the maximum length, in order to safely exclude them.
    if test_similar_scores:
        model_config = sockeye.model.SockeyeModel.load_config(os.path.join(data['model'], C.CONFIG_NAME))
        max_len = model_config.config_data.max_seq_len_target

        valid_outputs = list(filter(lambda x: len(x[0]) < max_len - 1,
                                    zip(translate_tokens, data['test_scores'], score_scores)))
        for translate_tokens, translate_score, score_score in valid_outputs:
            # Skip sentences that are close to the maximum length to avoid confusion about whether
            # the length penalty was applied
            if len(translate_tokens) >= max_len - 2:
                continue
            assert (translate_score == -np.inf and score_score == -np.inf) or abs(translate_score - score_score) < 0.02


def _translate_output_is_valid(translate_outputs: List[str]) -> bool:
    """
    True if there are invalid tokens in out_path, or if no valid outputs were found.
    """
    # At least one output must be non-empty
    found_valid_output = False
    bad_tokens = set(C.VOCAB_SYMBOLS)
    for output in translate_outputs:
        if output:
            found_valid_output = True
        if any(token for token in output.split() if token in bad_tokens):
            # There must be no bad tokens
            return False
    return found_valid_output


def collect_translate_output_and_scores(out_path: str) -> Tuple[List[str], List[float]]:
    """
    Collects translation outputs and scores from an output file
    produced with the 'translation_and_score' or nbest output handler.
    """
    translations = []  # type: List[str]
    scores = []  # type: List[float]
    with open(out_path) as out_fh:
        for line in out_fh:
            output = line.strip()
            translation = ''
            score = -np.inf
            try:
                output = json.loads(output)
                try:
                    translation = output['translation']
                    score = output['score']
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

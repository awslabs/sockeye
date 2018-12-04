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

import json
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
import sockeye.extract_parameters
import sockeye.lexicon
import sockeye.model
import sockeye.prepare_data
import sockeye.score
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


_LEXICON_CREATE_PARAMS_COMMON = "create -i {input} -m {model} -k {topk} -o {lexicon}"


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

_EVAL_PARAMS_COMMON = "--hypotheses {hypotheses} --references {references} --metrics {metrics}"

_EXTRACT_PARAMS = "--input {input} --names target_output_bias --list-all --output {output}"


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
                        use_target_constraints: bool = False,
                        max_seq_len: int = 10,
                        work_dir: Optional[str] = None,
                        seed: int = 13) -> Tuple[float, float, float, float]:
    """
    Train a model and translate a test set.  Report validation perplexity and BLEU.

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
    :param use_target_constraints: Whether to use lexical constraints in the second translation pass.
    :param max_seq_len: The maximum sequence length.
    :param work_dir: The directory to store the model and other outputs in.
    :param seed: The seed used for training.
    :return: A tuple containing perplexity, bleu scores for standard and reduced vocab decoding, chrf score.
    """
    with TemporaryDirectory(dir=work_dir, prefix="test_train_translate.") as work_dir:
        # Optionally create prepared data directory
        if use_prepared_data:
            prepared_data_path = os.path.join(work_dir, "prepared_data")
            params = "{} {}".format(sockeye.prepare_data.__file__,
                                    _PREPARE_DATA_COMMON.format(train_source=train_source_path,
                                                                train_target=train_target_path,
                                                                output=prepared_data_path,
                                                                max_len=max_seq_len))
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
                                                                                 max_len=max_seq_len),
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
                                                                   seed=seed),
                                       train_params)

            if train_source_factor_paths is not None:
                params += _TRAIN_WITH_FACTORS_COMMON.format(source_factors=" ".join(train_source_factor_paths))
            if dev_source_factor_paths is not None:
                params += _DEV_WITH_FACTORS_COMMON.format(dev_source_factors=" ".join(dev_source_factor_paths))

            logger.info("Starting training with parameters %s.", train_params)
            with patch.object(sys, "argv", params.split()):
                sockeye.train.main()

        # run checkpoint decoder on 1% of dev data
        test_checkpoint_decoder(dev_source_path, dev_target_path, model_path)

        # Translate corpus with the 1st params and scoring output handler to obtain scores
        out_path = os.path.join(work_dir, "test.out")
        params = "{} {} {}".format(sockeye.translate.__file__,
                                   _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                   input=test_source_path,
                                                                   output=out_path),
                                   translate_params)

        if test_source_factor_paths is not None:
            params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))

        logger.info("Translating with params %s", params)
        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        # Collect test inputs
        with open(test_source_path) as inputs:
            test_inputs = [line.strip() for line in inputs]

        # Collect test references
        with open(test_target_path, "r") as ref:
            test_references = [line.strip() for line in ref]

        # Collect test translate outputs and scores
        translate_outputs, translate_scores = _collect_translate_output_and_scores(out_path)

        assert len(test_inputs) == len(test_references) == len(translate_outputs) == len(translate_scores)

        # Test target lexical constraints
        if use_target_constraints:
            test_lexical_constraints(model_path, test_inputs, translate_outputs, translate_params, work_dir)

        # Test scoring by ensuring that the sockeye.scoring module produces the same scores when scoring the output
        # of sockeye.translate. However, since this training is on very small datasets, the output of sockeye.translate
        # is often pure garbage or empty and cannot be scored. So we only try to score if we have some valid output
        # to work with.
        # Only run scoring under these conditions. Why?
        # - scoring isn't compatible with prepared data because that loses the source ordering
        # - translate splits up too-long sentences and translates them in sequence, invalidating the score, so skip that
        # - scoring requires valid translation output to compare against
        if '--max-input-len' not in translate_params and _translate_output_is_valid(translate_outputs):
                test_scoring(model_path, work_dir, test_source_factor_paths, test_source_path,
                             translate_outputs, translate_params, translate_scores)

        # Translate corpus with 2nd set of translate params that are supposed to produce the same output
        if translate_params_equiv is not None:
            test_equiv_translate(model_path, test_source_path, test_source_factor_paths, translate_outputs,
                                 translate_scores, translate_params_equiv, work_dir)

        translate_outputs_restrict = test_restrict_lexicon(model_path, test_source_factor_paths,
                                                           test_source_path, translate_params, work_dir)
        test_averaging(model_path)
        test_parameter_extraction(model_path)
        test_evaluate(out_path, test_target_path)

        # get best validation perplexity
        metrics = sockeye.utils.read_metrics_file(path=os.path.join(model_path, C.METRICS_NAME))
        perplexity = min(m[C.PERPLEXITY + '-val'] for m in metrics)

        # compute metrics
        bleu = raw_corpus_bleu(hypotheses=translate_outputs, references=test_references, offset=0.01)
        chrf = raw_corpus_chrf(hypotheses=translate_outputs, references=test_references)
        bleu_restrict = raw_corpus_bleu(hypotheses=translate_outputs_restrict, references=test_references, offset=0.01)

        return perplexity, bleu, bleu_restrict, chrf


def test_evaluate(out_path, test_target_path):
    # Run evaluate cli
    eval_params = "{} {} ".format(sockeye.evaluate.__file__,
                                  _EVAL_PARAMS_COMMON.format(hypotheses=out_path,
                                                             references=test_target_path,
                                                             metrics="bleu chrf rouge1"))
    with patch.object(sys, "argv", eval_params.split()):
        sockeye.evaluate.main()


def test_parameter_extraction(model_path):
    extract_params = _EXTRACT_PARAMS.format(output=os.path.join(model_path, "params.extracted"),
                                            input=model_path)
    with patch.object(sys, "argv", extract_params.split()):
        sockeye.extract_parameters.main()
    with np.load(os.path.join(model_path, "params.extracted.npz")) as data:
        assert "target_output_bias" in data


def test_averaging(model_path):
    points = sockeye.average.find_checkpoints(model_path=model_path,
                                              size=1,
                                              strategy='best',
                                              metric=C.PERPLEXITY)
    assert len(points) > 0
    averaged_params = sockeye.average.average(points)
    assert averaged_params


def test_restrict_lexicon(model_path, test_source_factor_paths, test_source_path, translate_params, work_dir):
    out_path = os.path.join(work_dir, "out-restrict.txt")
    # fast_align lex table
    ttable_path = os.path.join(work_dir, "ttable")
    generate_fast_align_lex(ttable_path)
    # Top-K lexicon
    lexicon_path = os.path.join(work_dir, "lexicon")
    params = "{} {}".format(sockeye.lexicon.__file__,
                            _LEXICON_CREATE_PARAMS_COMMON.format(input=ttable_path,
                                                                 model=model_path,
                                                                 topk=20,
                                                                 lexicon=lexicon_path))
    with patch.object(sys, "argv", params.split()):
        sockeye.lexicon.main()
    # Translate corpus with restrict-lexicon
    params = "{} {} {} {}".format(sockeye.translate.__file__,
                                  _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                  input=test_source_path,
                                                                  output=out_path),
                                  translate_params,
                                  _TRANSLATE_PARAMS_RESTRICT.format(lexicon=lexicon_path, topk=1))
    if test_source_factor_paths is not None:
        params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()

    # Collect test translate outputs and scores
    translate_outputs, translate_scores = _collect_translate_output_and_scores(out_path)
    return translate_outputs



def test_equiv_translate(model_path, test_source_path, test_source_factor_paths, translate_outputs,
                         translate_scores, translate_params_equiv, work_dir):
    out_path = os.path.join(work_dir, "test.out.equiv")
    params = "{} {} {}".format(sockeye.translate.__file__,
                               _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                               input=test_source_path,
                                                               output=out_path),
                               translate_params_equiv)
    if test_source_factor_paths is not None:
        params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()
    # Collect translate outputs and scores
    translate_outputs_equiv, translate_scores_equiv = _collect_translate_output_and_scores(out_path)

    assert all(a == b for a, b in zip(translate_outputs, translate_outputs_equiv))
    assert all(abs(a - b) < 0.01 or np.isnan(a - b) for a, b in zip(translate_scores, translate_scores_equiv))


def test_checkpoint_decoder(dev_source_path, dev_target_path, model_path):
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
    assert 'bleu-val' in cp_metrics
    assert 'chrf-val' in cp_metrics
    assert 'decode-walltime-val' in cp_metrics


def test_scoring(model_path, work_dir, test_source_factor_paths, test_source_path, translate_outputs, translate_params,
                 translate_scores):
    # Translate params that affect the score need to be used for scoring as well.
    # Currently, the only relevant flag passed is the --softmax-temperature flag.
    score_params = ''
    if 'softmax-temperature' in translate_params:
        params = translate_params.split()
        for i, param in enumerate(params):
            if param == '--softmax-temperature':
                score_params = '--softmax-temperature {}'.format(params[i + 1])
                break
    out_path = os.path.join(work_dir, "score.out")
    target_path = os.path.join(work_dir, "score.target")
    with open(target_path, 'w') as target_out:  # TODO UGLY!!!
        for o in translate_outputs:
            print(o, file=target_out)

    params = "{} {} {}".format(sockeye.score.__file__,
                               _SCORE_PARAMS_COMMON.format(model=model_path,
                                                           source=test_source_path,
                                                           target=target_path,
                                                           output=out_path),
                               score_params)
    if test_source_factor_paths is not None:
        params += _SCORE_WITH_FACTORS_COMMON.format(source_factors=" ".join(test_source_factor_paths))
    logger.info("Scoring with params %s", params)
    with patch.object(sys, "argv", params.split()):
        sockeye.score.main()

    # Collect scores from output file
    with open(out_path) as score_out:
        score_scores = [float(line.strip()) for line in score_out]

    translate_tokens = [output.split() for output in translate_outputs]

    # Compare scored output to original translation output. There are a few tricks: for blank source sentences,
    # inference will report a score of -inf, so skip these. Second, we don't know if the scores include the
    # generation of </s> and have had length normalization applied. So, skip all sentences that are as long
    # as the maximum length, in order to safely exclude them.
    model_config = sockeye.model.SockeyeModel.load_config(os.path.join(model_path, C.CONFIG_NAME))
    max_len = model_config.config_data.max_seq_len_target
    # Filter out sockeye.translate outputs that had -inf score or are too long (which sockeye.score will have skipped)
    valid_outputs = list(filter(lambda x: len(x[0]) < max_len and not np.isinf(x[1]), zip(translate_tokens,
                                                                                          translate_scores)))
    assert len(valid_outputs) == len(score_scores)

    for (translate_tokens, translate_score), score_score in zip(valid_outputs, score_scores):
        # Skip sentences that are close to the maximum length to avoid confusion about whether
        # the length penalty was applied
        if len(translate_tokens) >= max_len - 2:
            continue
        assert abs(translate_score - score_score) < 0.01


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


def _collect_translate_output_and_scores(out_path) -> Tuple[List[str], List[float]]:
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
                    translation = output['translations'][0]
                    score = output['scores'][0]
                except IndexError:
                    pass
            except json.JSONDecodeError:
                try:
                    score, translation = output.split('\t', 1)
                except ValueError:
                    pass
            translations.append(translation)
            scores.append(float(score))
    return translations, scores


def test_lexical_constraints(model_path, translate_inputs, translate_outputs, translate_params, work_dir):
    """
    Read in the unconstrained system output from the first pass and use it to generate positive
    and negative constraints. It is important to generate a mix of positive, negative, and no
    constraints per batch, to test these production-realistic interactions as well.
    """
    # 'constraint' = positive constraints (must appear), 'avoid' = negative constraints (must not appear)
    for constraint_type in ["constraints", "avoid"]:
        constrained_inputs = []
        for sentno, (source, translate_output) in enumerate(zip(translate_inputs, translate_outputs)):
            target_words = translate_output.split()
            target_len = len(target_words)
            new_source = {'text': source}
            # From the odd-numbered sentences that are not too long, create constraints. We do
            # only odds to ensure we get batches with mixed constraints / lack of constraints.
            if target_len > 0 and sentno % 2 == 0:
                start_pos = 0
                end_pos = min(target_len, 3)
                constraint = ' '.join(target_words[start_pos:end_pos])
                new_source[constraint_type] = [constraint]
            constrained_inputs.append(json.dumps(new_source))

        new_test_source_path = os.path.join(work_dir, "test_constrained.txt")
        with open(new_test_source_path, 'w') as out:
            for json_line in constrained_inputs:
                print(json_line, file=out)

        out_path_constrained = os.path.join(work_dir, "out_constrained.txt")
        params = "{} {} {} --json-input --output-type translation_with_score".format(
            sockeye.translate.__file__,
            _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                            input=new_test_source_path,
                                            output=out_path_constrained),
            translate_params)

        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        constrained_outputs, constrained_scores = _collect_translate_output_and_scores(out_path_constrained)

        assert len(constrained_outputs) == len(translate_outputs) == len(constrained_inputs)
        for json_source, constrained_out, unconstrained_out in zip(constrained_inputs,
                                                                   constrained_outputs,
                                                                   translate_outputs):
            jobj = json.loads(json_source)
            if jobj.get(constraint_type) is None:
                # if there were no constraints, make sure the output is the same as the unconstrained output
                assert constrained_out == unconstrained_out
            else:
                restriction = jobj[constraint_type][0]
                if constraint_type == 'constraints':
                    # for positive constraints, ensure the constraint is in the constrained output
                    assert restriction in constrained_out
                else:
                    # for negative constraints, ensure the constraints is *not* in the constrained output
                    assert restriction not in constrained_out

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
                       " --output {output} {quiet} --pad-vocab-to-multiple-of 16"

_TRAIN_WITH_FACTORS_COMMON = " --source-factors {source_factors}"
_DEV_WITH_FACTORS_COMMON = " --validation-source-factors {dev_source_factors}"

_TRAIN_PARAMS_PREPARED_DATA_COMMON = "--use-cpu --max-seq-len {max_len} --prepared-data {prepared_data}" \
                                     " --validation-source {dev_source} --validation-target {dev_target} " \
                                     "--output {model} {quiet}"

_TRANSLATE_PARAMS_COMMON = "--use-cpu --models {model} --input {input} --output {output} {quiet}"

_TRANSLATE_WITH_FACTORS_COMMON = " --input-factors {input_factors}"

_TRANSLATE_PARAMS_RESTRICT = "--restrict-lexicon {lexicon} --restrict-lexicon-topk {topk}"

_SCORE_PARAMS_COMMON = "--use-cpu --model {model} --source {source} --target {target} --output {output}"

_SCORE_WITH_FACTORS_COMMON = " --source-factors {source_factors}"

_EVAL_PARAMS_COMMON = "--hypotheses {hypotheses} --references {references} --metrics {metrics} {quiet}"

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
        translate_score_path = os.path.join(work_dir, "out.scores.txt")
        params = "{} {} {} --output-type translation_with_score".format(sockeye.translate.__file__,
                                                                        _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                                                        input=test_source_path,
                                                                                                        output=out_path,
                                                                                                        quiet=quiet_arg),
                                                                        translate_params)

        if test_source_factor_paths is not None:
            params += _TRANSLATE_WITH_FACTORS_COMMON.format(input_factors=" ".join(test_source_factor_paths))

        with patch.object(sys, "argv", params.split()):
            sockeye.translate.main()

        # Break out translation and score
        with open(out_path) as out_fh:
            outputs = out_fh.readlines()
        with open(out_path, 'w') as out_translate, open(translate_score_path, 'w') as out_scores:
            for output in outputs:
                output = output.strip()
                # blank lines on test input will have only one field output (-inf for the score)
                try:
                    score, translation = output.split('\t')
                except ValueError:
                    score = output
                    translation = ""
                print(translation, file=out_translate)
                print(score, file=out_scores)

        # Test target constraints
        if use_target_constraints:
            """
            Read in the unconstrained system output from the first pass and use it to generate positive
            and negative constraints. It is important to generate a mix of positive, negative, and no
            constraints per batch, to test these production-realistic interactions as well.
            """
            # 'constraint' = positive constraints (must appear), 'avoid' = negative constraints (must not appear)
            for constraint_type in ["constraints", "avoid"]:
                constrained_sources = []
                with open(test_source_path) as source_inp, open(out_path) as system_out:
                    for sentno, (source, target) in enumerate(zip(source_inp, system_out)):
                        target_words = target.rstrip().split()
                        target_len = len(target_words)
                        new_source = {'text': source.rstrip()}
                        # From the odd-numbered sentences that are not too long, create constraints. We do
                        # only odds to ensure we get batches with mixed constraints / lack of constraints.
                        if target_len > 0 and sentno % 2 == 0:
                            start_pos = 0
                            end_pos = min(target_len, 3)
                            constraint = ' '.join(target_words[start_pos:end_pos])
                            new_source[constraint_type] = [constraint]
                        constrained_sources.append(json.dumps(new_source))

                new_test_source_path = os.path.join(work_dir, "test_constrained.txt")
                with open(new_test_source_path, 'w') as out:
                    for json_line in constrained_sources:
                        print(json_line, file=out)

                out_path_constrained = os.path.join(work_dir, "out_constrained.txt")
                params = "{} {} {} --json-input".format(sockeye.translate.__file__,
                                                        _TRANSLATE_PARAMS_COMMON.format(model=model_path,
                                                                                        input=new_test_source_path,
                                                                                        output=out_path_constrained,
                                                                                        quiet=quiet_arg),
                                                        translate_params)

                with patch.object(sys, "argv", params.split()):
                    sockeye.translate.main()

                for json_input, constrained_out, unconstrained_out in zip(open(new_test_source_path),
                                                                          open(out_path_constrained),
                                                                          open(out_path)):
                    jobj = json.loads(json_input)
                    if jobj.get(constraint_type, None) == None:
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

        # Test scoring by ensuring that the sockeye.scoring module produces the same scores when scoring the output
        # of sockeye.translate. However, since this training is on very small datasets, the output of sockeye.translate
        # is often pure garbage or empty and cannot be scored. So we only try to score if we have some valid output
        # to work with.

        # Skip if there are invalid tokens in the output, or if no valid outputs were found
        translate_output_is_valid = True
        with open(out_path) as out_fh:
            sentences = list(map(lambda x: x.rstrip(), out_fh.readlines()))
            # At least one output must be non-empty
            found_valid_output = any(sentences)

            # There must be no bad tokens
            found_bad_tokens = any([bad_token in ' '.join(sentences) for bad_token in C.VOCAB_SYMBOLS])

            translate_output_is_valid = found_valid_output and not found_bad_tokens

        # Only run scoring under these conditions. Why?
        # - scoring isn't compatible with prepared data because that loses the source ordering
        # - scoring doesn't support skipping softmax (which can be enabled explicitly or implicitly by using a beam size of 1)
        # - translate splits up too-long sentences and translates them in sequence, invalidating the score, so skip that
        # - scoring requires valid translation output to compare against
        if not use_prepared_data \
           and '--beam-size 1' not in translate_params \
           and '--max-input-len' not in translate_params \
           and translate_output_is_valid:

            ## Score
            # We use the translation parameters, but have to remove irrelevant arguments from it.
            # Currently, the only relevant flag passed is the --softmax-temperature flag.
            score_params = ''
            if 'softmax-temperature' in translate_params:
                params = translate_params.split(C.TOKEN_SEPARATOR)
                for i, param in enumerate(params):
                    if param == '--softmax-temperature':
                        score_params = '--softmax-temperature {}'.format(params[i + 1])
                        break

            scores_output_file = out_path + '.score'
            params = "{} {} {}".format(sockeye.score.__file__,
                                       _SCORE_PARAMS_COMMON.format(model=model_path,
                                                                   source=test_source_path,
                                                                   target=out_path,
                                                                   output=scores_output_file),
                                       score_params)

            if test_source_factor_paths is not None:
                params += _SCORE_WITH_FACTORS_COMMON.format(source_factors=" ".join(test_source_factor_paths))

            with patch.object(sys, "argv", params.split()):
                sockeye.score.main()

            # Compare scored output to original translation output. There are a few tricks: for blank source sentences,
            # inference will report a score of -inf, so skip these. Second, we don't know if the scores include the
            # generation of </s> and have had length normalization applied. So, skip all sentences that are as long
            # as the maximum length, in order to safely exclude them.
            with open(translate_score_path) as in_translate, open(out_path) as in_words, open(scores_output_file) as in_score:
                model_config = sockeye.model.SockeyeModel.load_config(os.path.join(model_path, C.CONFIG_NAME))
                max_len = model_config.config_data.max_seq_len_target

                # Filter out sockeye.translate sentences that had -inf or were too long (which sockeye.score will have skipped)
                translate_scores = []
                translate_lens = []
                score_scores = in_score.readlines()
                for score, sent in zip(in_translate.readlines(), in_words.readlines()):
                    if score != '-inf\n' and len(sent.split()) < max_len:
                        translate_scores.append(score)
                        translate_lens.append(len(sent.split()))

                assert len(translate_scores) == len(score_scores)

                # Compare scores (using 0.002 which covers common noise comparing e.g., 1.234 and 1.235)
                for translate_score, translate_len, score_score in zip(translate_scores, translate_lens, score_scores):
                    # Skip sentences that are close to the maximum length to avoid confusion about whether
                    # the length penalty was applied
                    if translate_len >= max_len - 2:
                        continue

                    translate_score = float(translate_score)
                    score_score = float(score_score)

                    assert abs(translate_score - score_score) < 0.002

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

        # test parameter extraction
        extract_params = _EXTRACT_PARAMS.format(output=os.path.join(model_path, "params.extracted"),
                                                input=model_path)
        with patch.object(sys, "argv", extract_params.split()):
            sockeye.extract_parameters.main()
        with np.load(os.path.join(model_path, "params.extracted.npz")) as data:
            assert "target_output_bias" in data

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

        # Run evaluate cli
        eval_params = "{} {} ".format(sockeye.evaluate.__file__,
                                      _EVAL_PARAMS_COMMON.format(hypotheses=out_path,
                                                                 references=test_target_path,
                                                                 metrics="bleu chrf rouge1",
                                                                 quiet=quiet_arg), )
        with patch.object(sys, "argv", eval_params.split()):
            sockeye.evaluate.main()

        return perplexity, bleu, bleu_restrict, chrf

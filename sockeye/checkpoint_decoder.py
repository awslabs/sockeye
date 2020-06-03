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

"""
Implements a thin wrapper around Translator to compute BLEU scores on (a sample of) validation data during training.
"""
import logging
import os
import random
import time
from contextlib import ExitStack
from typing import Any, Dict, Optional, List

import mxnet as mx

import sockeye.model
import sockeye.output_handler
import sockeye.translate
from . import constants as C
from . import data_io
from . import evaluate
from . import inference
from . import utils
from . import vocab

logger = logging.getLogger(__name__)


class CheckpointDecoder:
    """
    Decodes a (random sample of a) dataset using parameters at given checkpoint and computes BLEU against references.

    :param model_folder: The model folder where checkpoint decoder outputs will be written to.
    :param inputs: Path(s) to file containing input sentences (and their factors).
    :param references: Path to file containing references.
    :param source_vocabs: The source vocabularies.
    :param target_vocab: The target vocabulary.
    :param context: The devices to use for decoding.
    :param model: The translation model.
    :param max_input_len: Maximum input length.
    :param batch_size: Batch size.
    :param beam_size: Size of the beam.
    :param nbest_size: Size of nbest lists.
    :param length_penalty_alpha: Alpha factor for the length penalty
    :param length_penalty_beta: Beta factor for the length penalty
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param sample_size: Maximum number of sentences to sample and decode. If <=0, all sentences are used.
    :param random_seed: Random seed for sampling. Default: 42.
    :param hybridize: Turn on hybridization of the translator.
    """

    def __init__(self,
                 model_folder: str,
                 inputs: List[str],
                 references: str,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 model: sockeye.model.SockeyeModel,
                 context: mx.Context,
                 max_input_len: Optional[int] = None,
                 batch_size: int = 16,
                 beam_size: int = C.DEFAULT_BEAM_SIZE,
                 nbest_size: int = C.DEFAULT_NBEST_SIZE,
                 bucket_width_source: int = 10,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 ensemble_mode: str = 'linear',
                 sample_size: int = -1,
                 random_seed: int = 42,
                 hybridize: bool = True) -> None:
        self.max_input_len = max_input_len
        self.max_output_length_num_stds = max_output_length_num_stds
        self.ensemble_mode = ensemble_mode
        self.beam_size = beam_size
        self.nbest_size = nbest_size
        self.batch_size = batch_size
        self.bucket_width_source = bucket_width_source
        self.length_penalty_alpha = length_penalty_alpha
        self.length_penalty_beta = length_penalty_beta
        self.model = model

        with ExitStack() as exit_stack:
            inputs_fins = [exit_stack.enter_context(data_io.smart_open(f)) for f in inputs]
            references_fin = exit_stack.enter_context(data_io.smart_open(references))

            inputs_sentences = [f.readlines() for f in inputs_fins]
            target_sentences = references_fin.readlines()

            utils.check_condition(all(len(l) == len(target_sentences) for l in inputs_sentences),
                                  "Sentences differ in length")
            utils.check_condition(all(len(sentence.strip()) > 0 for sentence in target_sentences),
                                  "Empty target validation sentence.")

            if sample_size <= 0:
                sample_size = len(inputs_sentences[0])
            if sample_size < len(inputs_sentences[0]):
                self.target_sentences, *self.inputs_sentences = parallel_subsample(
                    [target_sentences] + inputs_sentences, sample_size, random_seed)
            else:
                self.inputs_sentences, self.target_sentences = inputs_sentences, target_sentences

            if sample_size < self.batch_size:
                self.batch_size = sample_size

        for i, factor in enumerate(self.inputs_sentences):
            write_to_file(factor, os.path.join(model_folder, C.DECODE_IN_NAME % i))
        write_to_file(self.target_sentences, os.path.join(model_folder, C.DECODE_REF_NAME))

        self.inputs_sentences = list(zip(*self.inputs_sentences))  # type: List[List[str]]

        scorer = inference.CandidateScorer(
            length_penalty_alpha=length_penalty_alpha,
            length_penalty_beta=length_penalty_beta,
            brevity_penalty_weight=0.0,
            prefix='scorer_')

        # TODO: possibly support decoding on multiple GPUs
        self.translator = inference.Translator(
            batch_size=self.batch_size,
            context=context,
            ensemble_mode=self.ensemble_mode,
            scorer=scorer,
            beam_search_stop='all',
            nbest_size=self.nbest_size,
            models=[self.model],
            source_vocabs=source_vocabs,
            target_vocab=target_vocab,
            restrict_lexicon=None,
            hybridize=hybridize)

        logger.info("Created CheckpointDecoder(max_input_len=%d, beam_size=%d, num_sentences=%d)",
                    max_input_len if max_input_len is not None else -1, beam_size, len(self.target_sentences))

    def decode_and_evaluate(self,
                            output_name: str = os.devnull) -> Dict[str, float]:
        """
        Decodes data set and evaluates given a checkpoint.

        :param output_name: Filename to write translations to. Defaults to /dev/null.
        :return: Mapping of metric names to scores.
        """

        # 1. Translate
        trans_wall_time = 0.0
        translations = []
        with data_io.smart_open(output_name, 'w') as output:
            handler = sockeye.output_handler.StringOutputHandler(output)
            tic = time.time()
            trans_inputs = []  # type: List[inference.TranslatorInput]
            for i, inputs in enumerate(self.inputs_sentences):
                trans_inputs.append(sockeye.inference.make_input_from_multiple_strings(i, inputs))
            trans_outputs = self.translator.translate(trans_inputs)
            trans_wall_time = time.time() - tic
            for trans_input, trans_output in zip(trans_inputs, trans_outputs):
                handler.handle(trans_input, trans_output)
                translations.append(trans_output.translation)
        avg_time = trans_wall_time / len(self.target_sentences)

        # 2. Evaluate
        return {C.BLEU: evaluate.raw_corpus_bleu(hypotheses=translations,
                                                 references=self.target_sentences,
                                                 offset=0.01),
                C.CHRF: evaluate.raw_corpus_chrf(hypotheses=translations,
                                                 references=self.target_sentences),
                C.ROUGE1: evaluate.raw_corpus_rouge1(hypotheses=translations,
                                                     references=self.target_sentences),
                C.ROUGE2: evaluate.raw_corpus_rouge2(hypotheses=translations,
                                                     references=self.target_sentences),
                C.ROUGEL: evaluate.raw_corpus_rougel(hypotheses=translations,
                                                     references=self.target_sentences),
                C.LENRATIO: evaluate.raw_corpus_length_ratio(hypotheses=translations,
                                                             references=self.target_sentences),
                C.AVG_TIME: avg_time,
                C.DECODING_TIME: trans_wall_time}


def parallel_subsample(parallel_sequences: List[List[Any]], sample_size: int, seed: int) -> List[Any]:
    # custom random number generator to guarantee the same samples across runs in order to be able to
    # compare metrics across independent runs
    random_gen = random.Random(seed)
    parallel_sample = list(zip(*random_gen.sample(list(zip(*parallel_sequences)), sample_size)))
    return parallel_sample


def write_to_file(data: List[str], fname: str):
    with data_io.smart_open(fname, 'w') as f:
        for x in data:
            print(x.rstrip(), file=f)

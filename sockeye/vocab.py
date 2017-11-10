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

import argparse
import json
import logging
import os
import pickle
from collections import Counter
from contextlib import ExitStack
from itertools import chain, islice
from typing import Dict, Iterable, List, Mapping

from sockeye.data_io import get_tokens, smart_open
from . import utils
from . import arguments
from . import constants as C
from . import log

logger = logging.getLogger(__name__)


def build_from_paths(paths: List[str], num_words: int = 50000, min_count: int = 1) -> Dict[str, int]:
    """
    Creates vocabulary from paths to a file in sentence-per-line format. A sentence is just a whitespace delimited
    list of tokens. Note that special symbols like the beginning of sentence (BOS) symbol will be added to the
    vocabulary.

    :param paths: List of paths to files with one sentence per line.
    :param num_words: Maximum number of words in the vocabulary.
    :param min_count: Minimum occurrences of words to be included in the vocabulary.
    :return: Word-to-id mapping.
    """
    with ExitStack() as stack:
        logger.info("Building vocabulary from dataset(s): %s", paths)
        files = (stack.enter_context(smart_open(path)) for path in paths)
        return build_vocab(chain(*files), num_words, min_count)


def build_vocab(data: Iterable[str], num_words: int = 50000, min_count: int = 1) -> Dict[str, int]:
    """
    Creates a vocabulary mapping from words to ids. Increasing integer ids are assigned by word frequency,
    using lexical sorting as a tie breaker. The only exception to this are special symbols such as the padding symbol
    (PAD).

    :param data: Sequence of sentences containing whitespace delimited tokens.
    :param num_words: Maximum number of words in the vocabulary.
    :param min_count: Minimum occurrences of words to be included in the vocabulary.
    :return: Word-to-id mapping.
    """
    vocab_symbols_set = set(C.VOCAB_SYMBOLS)
    raw_vocab = Counter(token for line in data for token in get_tokens(line)
                        if token not in vocab_symbols_set)
    logger.info("Initial vocabulary: %d types" % len(raw_vocab))

    # For words with the same count, they will be ordered reverse alphabetically.
    # Not an issue since we only care for consistency
    pruned_vocab = sorted(((c, w) for w, c in raw_vocab.items() if c >= min_count), reverse=True)
    logger.info("Pruned vocabulary: %d types (min frequency %d)", len(pruned_vocab), min_count)

    vocab = islice((w for c, w in pruned_vocab), num_words)

    word_to_id = {word: idx for idx, word in enumerate(chain(C.VOCAB_SYMBOLS, vocab))}
    logger.info("Final vocabulary: %d types (min frequency %d, top %d types)",
                len(word_to_id), min_count, num_words)

    # Important: pad symbol becomes index 0
    assert word_to_id[C.PAD_SYMBOL] == C.PAD_ID
    return word_to_id


def vocab_to_pickle(vocab: Mapping, path: str):
    """
    Saves vocabulary in pickle format.

    :param vocab: Vocabulary mapping.
    :param path: Output file path.
    """
    with open(path, 'wb') as out:
        pickle.dump(vocab, out)
        logger.info('Vocabulary saved to "%s"', path)


def vocab_to_json(vocab: Mapping, path: str):
    """
    Saves vocabulary in human-readable json.

    :param vocab: Vocabulary mapping.
    :param path: Output file path.
    """
    with open(path, "w", encoding=C.VOCAB_ENCODING) as out:
        json.dump(vocab, out, indent=4, ensure_ascii=False)
        logger.info('Vocabulary saved to "%s"', path)


def vocab_from_json_or_pickle(path) -> Dict:
    """
    Try loading the json version of the vocab and fall back to pickle for backwards compatibility.

    :param path: Path to vocab without the json suffix. If it exists the `path` + '.json' will be loaded as a JSON
        object and otherwise `path` is loaded as a pickle object.
    :return: The loaded vocabulary.
    """
    if os.path.exists(path + C.JSON_SUFFIX):
        return vocab_from_json(path + C.JSON_SUFFIX)
    else:
        return vocab_from_pickle(path)


def vocab_from_pickle(path: str) -> Dict:
    """
    Saves vocabulary in pickle format.

    :param path: Path to pickle file containing the vocabulary.
    :return: The loaded vocabulary.
    """
    with open(path, 'rb') as inp:
        vocab = pickle.load(inp)
        logger.info('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab


def vocab_from_json(path: str) -> Dict:
    """
    Saves vocabulary in json format.

    :param path: Path to json file containing the vocabulary.
    :return: The loaded vocabulary.
    """
    with open(path, encoding=C.VOCAB_ENCODING) as inp:
        vocab = json.load(inp)
        logger.info('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab


def reverse_vocab(vocab: Mapping) -> Dict:
    """
    Returns value-to-key mapping from key-to-value-mapping.

    :param vocab: Key to value mapping.
    :return: A mapping from values to keys.
    """
    return {v: k for k, v in vocab.items()}


def main():
    params = argparse.ArgumentParser(description='CLI to build source and target vocab(s).')
    arguments.add_build_vocab_args(params)
    args = params.parse_args()

    num_words, num_words_other = args.num_words
    utils.check_condition(num_words == num_words_other,
                          "Vocabulary CLI only allows a common value for --num-words")
    word_min_count, word_min_count_other = args.word_min_count
    utils.check_condition(word_min_count == word_min_count_other,
                          "Vocabulary CLI only allows a common value for --word-min-count")

    global logger
    logger = log.setup_main_logger("build_vocab", file_logging=True, console=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    vocab = build_from_paths(args.inputs, num_words=num_words, min_count=word_min_count)
    logger.info("Vocabulary size: %d ", len(vocab))
    vocab_to_json(vocab, args.output + C.JSON_SUFFIX)


if __name__ == "__main__":
    main()

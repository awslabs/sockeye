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
from collections import Counter
from contextlib import ExitStack
from itertools import chain, islice
from typing import Dict, Iterable, List, Optional, Tuple

from . import constants as C
from . import log
from . import utils

logger = logging.getLogger(__name__)


Vocab = Dict[str, int]
InverseVocab = Dict[int, str]


def build_from_paths(paths: List[str], num_words: Optional[int] = None, min_count: int = 1,
                     pad_to_multiple_of: Optional[int] = None) -> Vocab:
    """
    Creates vocabulary from paths to a file in sentence-per-line format. A sentence is just a whitespace delimited
    list of tokens. Note that special symbols like the beginning of sentence (BOS) symbol will be added to the
    vocabulary.

    :param paths: List of paths to files with one sentence per line.
    :param num_words: Optional maximum number of words in the vocabulary.
    :param min_count: Minimum occurrences of words to be included in the vocabulary.
    :param pad_to_multiple_of: If not None, pads the vocabulary to a size that is the next multiple of this int.
    :return: Word-to-id mapping.
    """
    with ExitStack() as stack:
        logger.info("Building vocabulary from dataset(s): %s", paths)
        files = (stack.enter_context(utils.smart_open(path)) for path in paths)
        return build_vocab(chain(*files), num_words, min_count, pad_to_multiple_of)


def build_vocab(data: Iterable[str], num_words: Optional[int] = None, min_count: int = 1,
                pad_to_multiple_of: Optional[int] = None) -> Vocab:
    """
    Creates a vocabulary mapping from words to ids. Increasing integer ids are assigned by word frequency,
    using lexical sorting as a tie breaker. The only exception to this are special symbols such as the padding symbol
    (PAD).

    :param data: Sequence of sentences containing whitespace delimited tokens.
    :param num_words: Optional maximum number of words in the vocabulary.
    :param min_count: Minimum occurrences of words to be included in the vocabulary.
    :param pad_to_multiple_of: If not None, pads the vocabulary to a size that is the next multiple of this int.
    :return: Word-to-id mapping.
    """
    vocab_symbols_set = set(C.VOCAB_SYMBOLS)
    raw_vocab = Counter(token for line in data for token in utils.get_tokens(line)
                        if token not in vocab_symbols_set)
    # For words with the same count, they will be ordered reverse alphabetically.
    # Not an issue since we only care for consistency
    pruned_vocab = [w for c, w in sorted(((c, w) for w, c in raw_vocab.items() if c >= min_count), reverse=True)]

    if num_words is not None:
        vocab = list(islice(pruned_vocab, num_words))
        num_words_log = str(num_words)
    else:
        vocab = pruned_vocab
        num_words_log = "None"

    if pad_to_multiple_of is not None:
        current_vocab_size = len(vocab) + len(C.VOCAB_SYMBOLS)
        rest = current_vocab_size % pad_to_multiple_of
        padded_vocab_size = current_vocab_size if rest == 0 else current_vocab_size + pad_to_multiple_of - rest
        logger.info("Padding vocabulary to a multiple of %d: %d -> %d",
                    pad_to_multiple_of, current_vocab_size, padded_vocab_size)
        pad_entries = [C.PAD_FORMAT % idx for idx in range(current_vocab_size, padded_vocab_size)]
        pad_to_multiple_log = str(pad_to_multiple_of)
    else:
        pad_entries = []
        pad_to_multiple_log = "None"

    word_to_id = {word: idx for idx, word in enumerate(chain(C.VOCAB_SYMBOLS, vocab, pad_entries))}
    logger.info("Vocabulary: types: %d/%d/%d/%d (initial/min_pruned/max_pruned/+special) " +
                "[min_frequency=%d, max_num_types=%s, pad_to_multiple_of=%s]",
                len(raw_vocab), len(pruned_vocab), len(vocab),
                len(word_to_id), min_count, num_words_log, pad_to_multiple_log)

    # Important: pad symbol becomes index 0
    assert word_to_id[C.PAD_SYMBOL] == C.PAD_ID
    return word_to_id


def vocab_to_json(vocab: Vocab, path: str):
    """
    Saves vocabulary in human-readable json.

    :param vocab: Vocabulary mapping.
    :param path: Output file path.
    """
    with open(path, "w", encoding=C.VOCAB_ENCODING) as out:
        json.dump(vocab, out, indent=4, ensure_ascii=False)
        logger.info('Vocabulary saved to "%s"', path)


def vocab_from_json(path: str, encoding: str = C.VOCAB_ENCODING) -> Vocab:
    """
    Saves vocabulary in json format.

    :param path: Path to json file containing the vocabulary.
    :param encoding: Vocabulary encoding.
    :return: The loaded vocabulary.
    """
    with open(path, encoding=encoding) as inp:
        vocab = json.load(inp)
        logger.info('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab


def save_source_vocabs(source_vocabs: List[Vocab], folder: str):
    """
    Saves source vocabularies (primary surface form vocabulary) and optional factor vocabularies to folder.

    :param source_vocabs: List of source vocabularies.
    :param folder: Destination folder.
    """
    for i, vocab in enumerate(source_vocabs):
        vocab_to_json(vocab, os.path.join(folder, C.VOCAB_SRC_NAME % i))


def save_target_vocab(target_vocab: Vocab, folder: str):
    """
    Saves target vocabulary to folder.

    :param target_vocab: Target vocabulary.
    :param folder: Destination folder.
    """
    vocab_to_json(target_vocab, os.path.join(folder, C.VOCAB_TRG_NAME % 0))


def load_source_vocabs(folder: str) -> List[Vocab]:
    """
    Loads source vocabularies from folder. The first element in the list is the primary source vocabulary.
    Other elements correspond to optional additional source factor vocabularies found in folder.

    :param folder: Source folder.
    :return: List of vocabularies.
    """
    return [vocab_from_json(os.path.join(folder, fname)) for fname in
            sorted([f for f in os.listdir(folder) if f.startswith(C.VOCAB_SRC_PREFIX)])]


def load_target_vocab(folder: str) -> Vocab:
    """
    Loads target vocabulary from folder.

    :param folder: Source folder.
    :return: Target vocabulary
    """
    return vocab_from_json(os.path.join(folder, C.VOCAB_TRG_NAME % 0))


def load_or_create_vocab(data: str, vocab_path: Optional[str], num_words: int, word_min_count: int,
                         pad_to_multiple_of: Optional[int] = None) -> Vocab:
    """
    If the vocabulary path is defined, the vocabulary is loaded from the path.
    Otherwise, it is built from the data file. No writing to disk occurs.
    """
    if vocab_path is None:
        return build_from_paths(paths=[data], num_words=num_words, min_count=word_min_count,
                                pad_to_multiple_of=pad_to_multiple_of)
    else:
        return vocab_from_json(vocab_path)


def load_or_create_vocabs(source_paths: List[str],
                          target_path: str,
                          source_vocab_paths: List[Optional[str]],
                          target_vocab_path: Optional[str],
                          shared_vocab: bool,
                          num_words_source: Optional[int], word_min_count_source: int,
                          num_words_target: Optional[int], word_min_count_target: int,
                          pad_to_multiple_of: Optional[int] = None) -> Tuple[List[Vocab], Vocab]:
    """
    Returns vocabularies for source files (including factors) and target.
    If the respective vocabulary paths are not None, the vocabulary is read from the path and returned.
    Otherwise, it is built from the support and saved to the path.

    :param source_paths: The path to the source text (and optional token-parallel factor files).
    :param target_path: The target text.
    :param source_vocab_paths: The source vocabulary path (and optional factor vocabulary paths).
    :param target_vocab_path: The target vocabulary path.
    :param shared_vocab: Whether the source and target vocabularies are shared.
    :param num_words_source: Number of words in the source vocabulary.
    :param word_min_count_source: Minimum frequency of words in the source vocabulary.
    :param num_words_target: Number of words in the target vocabulary.
    :param word_min_count_target: Minimum frequency of words in the target vocabulary.
    :param pad_to_multiple_of: If not None, pads the vocabularies to a size that is the next multiple of this int.
    :return: List of source vocabularies (for source and factors), and target vocabulary.
    """
    source_path, *source_factor_paths = source_paths
    source_vocab_path, *source_factor_vocab_paths = source_vocab_paths

    logger.info("=============================")
    logger.info("Loading/creating vocabularies")
    logger.info("=============================")
    logger.info("(1) Surface form vocabularies (source & target)")

    if shared_vocab:
        if source_vocab_path and target_vocab_path:
            vocab_source = vocab_from_json(source_vocab_path)
            vocab_target = vocab_from_json(target_vocab_path)
            utils.check_condition(are_identical(vocab_source, vocab_target),
                                  "Shared vocabulary requires identical source and target vocabularies. "
                                  "The vocabularies in %s and %s are not identical." % (source_vocab_path,
                                                                                        target_vocab_path))

        elif source_vocab_path is None and target_vocab_path is None:
            utils.check_condition(num_words_source == num_words_target,
                                  "A shared vocabulary requires the number of source and target words to be the same.")
            utils.check_condition(word_min_count_source == word_min_count_target,
                                  "A shared vocabulary requires the minimum word count for source and target "
                                  "to be the same.")
            vocab_source = vocab_target = build_from_paths(paths=[source_path, target_path],
                                                           num_words=num_words_source,
                                                           min_count=word_min_count_source,
                                                           pad_to_multiple_of=pad_to_multiple_of)

        else:
            vocab_path = source_vocab_path if source_vocab_path is not None else target_vocab_path
            logger.info("Using %s as a shared source/target vocabulary." % vocab_path)
            vocab_source = vocab_target = vocab_from_json(vocab_path)

    else:
        vocab_source = load_or_create_vocab(source_path, source_vocab_path, num_words_source, word_min_count_source,
                                            pad_to_multiple_of=pad_to_multiple_of)
        vocab_target = load_or_create_vocab(target_path, target_vocab_path, num_words_target, word_min_count_target,
                                            pad_to_multiple_of=pad_to_multiple_of)

    vocab_source_factors = []  # type: List[Vocab]
    if source_factor_paths:
        logger.info("(2) Additional source factor vocabularies")
        # source factor vocabs are always created
        for factor_path, factor_vocab_path in zip(source_factor_paths, source_factor_vocab_paths):
            vocab_source_factors.append(load_or_create_vocab(factor_path, factor_vocab_path,
                                                             num_words_source, word_min_count_target))

    return [vocab_source] + vocab_source_factors, vocab_target


def reverse_vocab(vocab: Vocab) -> InverseVocab:
    """
    Returns value-to-key mapping from key-to-value-mapping.

    :param vocab: Key to value mapping.
    :return: A mapping from values to keys.
    """
    return {v: k for k, v in vocab.items()}


def get_ordered_tokens_from_vocab(vocab: Vocab) -> List[str]:
    """
    Returns the list of tokens in a vocabulary, ordered by increasing vocabulary id.

    :param vocab: Input vocabulary.
    :return: List of tokens.
    """
    return [token for token, token_id in sorted(vocab.items(), key=lambda i: i[1])]


def are_identical(*vocabs: Vocab):
    assert len(vocabs) > 0, "At least one vocabulary needed."
    return all(set(vocab.items()) == set(vocabs[0].items()) for vocab in vocabs)


def main():
    from . import arguments
    params = argparse.ArgumentParser(description='CLI to build source and target vocab(s).')
    arguments.add_build_vocab_args(params)
    args = params.parse_args()

    num_words, num_words_other = args.num_words
    num_words = num_words if num_words > 0 else None
    num_words_other = num_words_other if num_words_other > 0 else None
    utils.check_condition(num_words == num_words_other,
                          "Vocabulary CLI only allows a common value for --num-words")
    word_min_count, word_min_count_other = args.word_min_count
    utils.check_condition(word_min_count == word_min_count_other,
                          "Vocabulary CLI only allows a common value for --word-min-count")

    global logger
    logger = log.setup_main_logger("build_vocab", file_logging=True, console=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    vocab = build_from_paths(args.inputs,
                             num_words=num_words,
                             min_count=word_min_count,
                             pad_to_multiple_of=args.pad_vocab_to_multiple_of)
    logger.info("Vocabulary size: %d ", len(vocab))
    vocab_to_json(vocab, args.output)


if __name__ == "__main__":
    main()

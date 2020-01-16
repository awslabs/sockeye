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

import argparse
import os
import sys
import time
import logging
from itertools import groupby
from operator import itemgetter
from typing import Dict, Generator, Tuple, Optional

import mxnet as mx
import numpy as np

from . import arguments
from . import constants as C
from . import vocab
from .data_io import smart_open, get_tokens, tokens2ids
from .log import setup_main_logger, log_sockeye_version

logger = logging.getLogger(__name__)


def lexicon_iterator(path: str,
                     vocab_source: Dict[str, int],
                     vocab_target: Dict[str, int]) -> Generator[Tuple[int, int, float], None, None]:
    """
    Yields lines from a translation table of format: src, trg, logprob.

    :param path: Path to lexicon file.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :return: Generator returning tuples (src_id, trg_id, prob).
    """
    assert C.UNK_SYMBOL in vocab_source
    assert C.UNK_SYMBOL in vocab_target
    src_unk_id = vocab_source[C.UNK_SYMBOL]
    trg_unk_id = vocab_target[C.UNK_SYMBOL]
    with smart_open(path) as fin:
        for line in fin:
            src, trg, logprob = line.rstrip("\n").split("\t")
            prob = np.exp(float(logprob))
            src_id = vocab_source.get(src, src_unk_id)
            trg_id = vocab_target.get(trg, trg_unk_id)
            yield src_id, trg_id, prob


def read_lexicon(path: str, vocab_source: Dict[str, int], vocab_target: Dict[str, int]) -> np.ndarray:
    """
    Loads lexical translation probabilities from a translation table of format: src, trg, logprob.
    Source words unknown to vocab_source are discarded.
    Target words unknown to vocab_target contribute to p(unk|source_word).
    See Incorporating Discrete Translation Lexicons into Neural Machine Translation, Section 3.1 & Equation 5
    (https://arxiv.org/pdf/1606.02006.pdf))

    :param path: Path to lexicon file.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :return: Lexicon array. Shape: (vocab_source_size, vocab_target_size).
    """
    src_unk_id = vocab_source[C.UNK_SYMBOL]
    trg_unk_id = vocab_target[C.UNK_SYMBOL]
    lexicon = np.zeros((len(vocab_source), len(vocab_target)))
    n = 0
    for src_id, trg_id, prob in lexicon_iterator(path, vocab_source, vocab_target):
        if src_id == src_unk_id:
            continue
        if trg_id == trg_unk_id:
            lexicon[src_id, trg_unk_id] += prob
        else:
            lexicon[src_id, trg_id] = prob
        n += 1
    logger.info("Loaded lexicon from '%s' with %d entries", path, n)
    return lexicon


class LexiconInitializer(mx.initializer.Initializer):
    """
    Given a lexicon NDArray, initialize the variable named C.LEXICON_NAME with it.

    :param lexicon: Lexicon array.
    """

    def __init__(self, lexicon: mx.nd.NDArray) -> None:
        super().__init__()
        self.lexicon = lexicon

    def _init_default(self, sym_name, arr):
        assert sym_name == C.LEXICON_NAME, "This initializer should only be used for a lexicon parameter variable"
        logger.info("Initializing '%s' with lexicon.", sym_name)
        assert len(arr.shape) == 2, "Only 2d weight matrices supported."
        self.lexicon.copyto(arr)


class TopKLexicon:
    """
    Lexicon component that stores the k most likely target words for each source word.  Used during
    decoding to restrict target vocabulary for each source sequence.

    :param vocab_source: Trained model source vocabulary.
    :param vocab_target: Trained mode target vocabulary.
    """

    def __init__(self,
                 vocab_source: Dict[str, int],
                 vocab_target: Dict[str, int]) -> None:
        self.vocab_source = vocab_source
        self.vocab_target = vocab_target
        # Shape: (vocab_source_size, k), k determined at create() or load()
        self.lex = None  # type: np.ndarray
        # Always allow special vocab symbols in target vocab
        self.always_allow = np.array([vocab_target[symbol] for symbol in C.VOCAB_SYMBOLS], dtype=np.int)

    def create(self, path: str, k: int = 20):
        """
        Create from a scored lexicon file (fast_align format) using vocab from a trained Sockeye model.

        :param path: Path to lexicon file.
        :param k: Number of target entries per source to keep.
        """
        self.lex = np.zeros((len(self.vocab_source), k), dtype=np.int)
        src_unk_id = self.vocab_source[C.UNK_SYMBOL]
        trg_unk_id = self.vocab_target[C.UNK_SYMBOL]
        num_insufficient = 0  # number of source tokens with insufficient number of translations given k
        for src_id, group in groupby(lexicon_iterator(path, self.vocab_source, self.vocab_target), key=itemgetter(0)):
            # Unk token will always be part of target vocab, so no need to track it here
            if src_id == src_unk_id:
                continue

            # filter trg_unk_id
            filtered_group = ((trg_id, prob) for src_id, trg_id, prob in group if trg_id != trg_unk_id)
            # sort by prob and take top k
            top_k = [trg_id for trg_id, prob in sorted(filtered_group, key=itemgetter(1), reverse=True)[:k]]
            if len(top_k) < k:
                num_insufficient += 1

            self.lex[src_id, :len(top_k)] = top_k

        logger.info("Created top-k lexicon from \"%s\", k=%d. %d source tokens with fewer than %d translations",
                    path, k, num_insufficient, k)

    def save(self, path: str):
        """
        Save lexicon in Numpy array format.  Lexicon will be specific to Sockeye model.

        :param path: Path to Numpy array output file.
        """
        with open(path, 'wb') as out:
            np.save(out, self.lex)
        logger.info("Saved top-k lexicon to \"%s\"", path)

    def load(self, path: str, k: Optional[int] = None):
        """
        Load lexicon from Numpy array file. The top-k target ids will be sorted by increasing target id.

        :param path: Path to Numpy array file.
        :param k: Optionally load less items than stored in path.
        """
        load_time_start = time.time()
        with open(path, 'rb') as inp:
            _lex = np.load(inp)
        loaded_k = _lex.shape[1]
        if k is not None:
            top_k = min(k, loaded_k)
            if k > loaded_k:
                logger.warning("Can not load top-%d translations from lexicon that "
                               "contains at most %d entries per source.", k, loaded_k)
        else:
            top_k = loaded_k
        self.lex = np.zeros((len(self.vocab_source), top_k), dtype=_lex.dtype)
        for src_id, trg_ids in enumerate(_lex):
            self.lex[src_id, :] = np.sort(trg_ids[:top_k])
        load_time = time.time() - load_time_start
        logger.info("Loaded top-%d lexicon from \"%s\" in %.4fs.", top_k, path, load_time)

    def get_trg_ids(self, src_ids: np.ndarray) -> np.ndarray:
        """
        Lookup possible target ids for input sequence of source ids.

        :param src_ids: Sequence(s) of source ids (any shape).
        :return: Possible target ids for source (unique sorted, always includes special symbols).
        """
        # TODO: When MXNet adds support for set operations, we can migrate to avoid conversions to/from NumPy.
        unique_src_ids = np.lib.arraysetops.unique(src_ids)
        trg_ids = np.lib.arraysetops.union1d(self.always_allow, self.lex[unique_src_ids, :].reshape(-1))
        return trg_ids


def create(args):
    setup_main_logger(console=not args.quiet, file_logging=not args.no_logfile, path=args.output + ".log")
    global logger
    logger = logging.getLogger('create')
    log_sockeye_version(logger)
    logger.info("Creating top-k lexicon from \"%s\"", args.input)
    logger.info("Reading source and target vocab from \"%s\"", args.model)
    vocab_source = vocab.load_source_vocabs(args.model)[0]
    vocab_target = vocab.load_target_vocab(args.model)
    logger.info("Building top-%d lexicon", args.k)
    lexicon = TopKLexicon(vocab_source, vocab_target)
    lexicon.create(args.input, args.k)
    lexicon.save(args.output)


def inspect(args):
    setup_main_logger(console=True, file_logging=False)
    global logger
    logger = logging.getLogger('inspect')
    log_sockeye_version(logger)
    logger.info("Inspecting top-k lexicon at \"%s\"", args.lexicon)
    vocab_source = vocab.load_source_vocabs(args.model)[0]
    vocab_target = vocab.vocab_from_json(os.path.join(args.model, C.VOCAB_TRG_NAME))
    vocab_target_inv = vocab.reverse_vocab(vocab_target)
    lexicon = TopKLexicon(vocab_source, vocab_target)
    lexicon.load(args.lexicon, args.k)
    logger.info("Reading from STDIN...")
    for line in sys.stdin:
        tokens = list(get_tokens(line))
        if not tokens:
            continue
        ids = tokens2ids(tokens, vocab_source)
        print("Input:  n=%d" % len(tokens), " ".join("%s(%d)" % (tok, i) for tok, i in zip(tokens, ids)))
        trg_ids = lexicon.get_trg_ids(np.array(ids))
        tokens_trg = [vocab_target_inv.get(trg_id, C.UNK_SYMBOL) for trg_id in trg_ids]
        print("Output: n=%d" % len(tokens_trg), " ".join("%s(%d)" % (tok, i) for tok, i in zip(tokens_trg, trg_ids)))
        print()


def main():
    """
    Commandline interface for building/inspecting top-k lexicons using during decoding.
    """
    params = argparse.ArgumentParser(description="Create or inspect a top-k lexicon for use during decoding.")
    subparams = params.add_subparsers(title="Commands")

    params_create = subparams.add_parser('create', description="Create top-k lexicon for use during decoding. "
                                                               "See sockeye_contrib/fast_align/README.md "
                                                               "for information on creating input lexical tables.")
    arguments.add_lexicon_args(params_create)
    arguments.add_lexicon_create_args(params_create)
    arguments.add_logging_args(params_create)
    params_create.set_defaults(func=create)

    params_inspect = subparams.add_parser('inspect', description="Inspect top-k lexicon for use during decoding.")
    arguments.add_lexicon_inspect_args(params_inspect)
    arguments.add_lexicon_args(params_inspect)
    params_inspect.set_defaults(func=inspect)

    args = params.parse_args()
    if 'func' not in args:
        params.print_help()
        return 1
    else:
        args.func(args)


if __name__ == "__main__":
    main()

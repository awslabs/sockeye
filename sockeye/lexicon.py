# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import collections
import os
import sys
import time
import logging
from itertools import groupby
from operator import itemgetter
from typing import Dict, Generator, List, Tuple, Optional
from abc import abstractmethod, ABC

import numpy as np

from sockeye.data_io import SequenceReader

from . import arguments
from . import constants as C
from . import vocab
from .utils import smart_open, get_tokens
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


class RestrictLexicon(ABC):
    """
    Lexicon component that potentially restricts the set of output words.

    If `is_blocking()` is True the set of target ids pose a negative constraint as tokens ids that must not be used on
    the target side. Conversely, if `is_blocking` is False the lexicon poses a positive constraint of returning the set
    of allowed target words.
    """

    lex: Optional[np.ndarray] = None

    def save(self, path: str):
        """
        Save lexicon in Numpy array format.  Lexicon will be specific to Sockeye model.

        :param path: Path to Numpy array output file.
        """
        assert self.lex is not None, "Lexicon uninitialized, can't be saved."
        with open(path, 'wb') as out:
            np.save(out, self.lex)
        logger.info("Saved lexicon to \"%s\"", path)

    @abstractmethod
    def load_np(self, lex: np.ndarray, k: Optional[int] = None):
        raise NotImplementedError()

    @abstractmethod
    def requires_src_ids(self) -> bool:
        """ If true src_ids are required as an argument to get_trg_ids. Otherwise the set of target ids are source
        independent and `None` may be passed instead. """
        raise NotImplementedError()

    @abstractmethod
    def is_blocking(self) -> bool:
        """ If true use get_blocked_trg_ids to obtain blocked ids, otherwise use get_allowed_trg_ids to get allowed
            target ids(inverts the meaning of the target ids)."""
        raise NotImplementedError()

    @abstractmethod
    def get_allowed_trg_ids(self, src_ids: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_blocked_trg_ids(self, src_ids: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError()


def load_restrict_lexicon(
        path: str,
        vocab_source: Optional[Dict[str, int]] = None,
        vocab_target: Optional[Dict[str, int]] = None,
        k: Optional[int] = None) -> RestrictLexicon:
    load_time_start = time.time()
    with open(path, 'rb') as inp:
        lex = np.load(inp)
        load_time = time.time() - load_time_start
        # Both lexicon types are serialized as numpy arrays and we distinguish them by their shape
        logger.info("Loaded lexicon from \"%s\" in %.4fs.", path, load_time)
        if len(lex.shape) == 1:
            lexicon = StaticBlockLexicon()  # type: RestrictLexicon
            lexicon.load_np(lex)
        elif len(lex.shape) == 2:
            lexicon = TopKLexicon(vocab_source, vocab_target)
            lexicon.load_np(lex, k=k)
        else:
            raise ValueError("Expected a 1d or 2d array.")
        return lexicon


class TopKLexicon(RestrictLexicon):
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
        self.lex = None  # type: Optional[np.ndarray]
        # Always allow special vocab symbols in target vocab
        self.always_allow = np.array([vocab_target[symbol] for symbol in C.VOCAB_SYMBOLS], dtype='int32')

    def create(self, path: str, k: int = 20):
        """
        Create from a scored lexicon file (fast_align format) using vocab from a trained Sockeye model.

        :param path: Path to lexicon file.
        :param k: Number of target entries per source to keep.
        """
        self.lex = np.zeros((len(self.vocab_source), k), dtype='int32')
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

    def load_np(self, lex: np.ndarray, k: Optional[int] = None):
        load_time_start = time.time()
        loaded_k = lex.shape[1]
        if k is not None:
            top_k = min(k, loaded_k)
            if k > loaded_k:
                logger.warning("Can not load top-%d translations from lexicon that "
                               "contains at most %d entries per source.", k, loaded_k)
        else:
            top_k = loaded_k
        self.lex = np.zeros((len(self.vocab_source), top_k), dtype=lex.dtype)
        for src_id, trg_ids in enumerate(lex):
            self.lex[src_id, :] = np.sort(trg_ids[:top_k])
        load_time = time.time() - load_time_start
        logger.info("Created top-%d lexicon in %.4fs.", top_k, load_time)

    def load(self, path: str, k: Optional[int] = None):
        """
        Load lexicon from Numpy array file. The top-k target ids will be sorted by increasing target id.

        :param path: Path to Numpy array file.
        :param k: Optionally load less items than stored in path.
        """
        load_time_start = time.time()
        with open(path, 'rb') as inp:
            lex = np.load(inp)
            load_time = time.time() - load_time_start
            logger.info("Loaded lexicon from \"%s\" in %.4fs.", path, load_time)
            return self.load_np(lex, k)

    def requires_src_ids(self):
        return True

    def is_blocking(self) -> bool:
        return False

    def get_trg_ids(self, src_ids: np.ndarray) -> np.ndarray:
        # Note: we have this function for backwards compatibility when `get_trg_ids` was the only function that returned
        # allowed target ids
        return self.get_allowed_trg_ids(src_ids)

    def get_allowed_trg_ids(self, src_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Lookup possible target ids for input sequence of source ids.

        :param src_ids: Sequence(s) of source ids (any shape).
        :return: Possible target ids for source (unique sorted, always includes special symbols).
        """
        unique_src_ids = np.lib.arraysetops.unique(src_ids)  # type: ignore
        trg_ids = np.lib.arraysetops.union1d(self.always_allow, self.lex[unique_src_ids, :].reshape(-1))  # type: ignore
        logger.debug(f"lookup: {trg_ids.shape[0]} unique targets for {unique_src_ids.shape[0]} unique sources")
        return trg_ids

    def get_blocked_trg_ids(self, src_ids):
        raise NotImplementedError()


class StaticBlockLexicon(RestrictLexicon):
    """
    A lexicon that blocks a fixed set of target ids independent of the src_ids.
    """

    def __init__(self, lex: Optional[np.ndarray] = None):
        if lex is not None:
            self.lex = lex

    def create(self, block_tokens: List[str], vocab_target: Dict[str, List[int]]):
        # We do not default to UNK because we want to only block on real tokens
        # We also exclude any other special symbols
        block_tokens_set = set(block_tokens)
        logger.info(f"Creating static block lexicon with tokens: {block_tokens_set}")
        num_not_in_vocab = 0
        block_token_ids = []
        for token in block_tokens:
            if token in C.VOCAB_SYMBOLS:
                continue
            if token not in vocab_target:
                num_not_in_vocab += 1
                continue
            block_token_ids.extend(vocab_target[token])
        block_token_ids = list(set(block_token_ids))

        self.lex = np.array(block_token_ids, dtype='int32')
        logger.info("Created static block lexicon with %d tokens, %d skipped because they were not in the vocabulary",
                    len(block_token_ids),
                    num_not_in_vocab)

    def load_np(self, lex: np.ndarray, k: Optional[int] = None):
        self.lex = lex
    
    def requires_src_ids(self):
        return False

    def is_blocking(self):
        return True

    def get_blocked_trg_ids(self, src_ids: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.lex is not None, "Lexicon not loaded yet."
        return self.lex

    def get_allowed_trg_ids(self, src_ids):
        raise NotImplementedError()


def create(args):
    setup_main_logger(console=not args.quiet, file_logging=not args.no_logfile, path=args.output + ".log")
    global logger
    logger = logging.getLogger('create')
    log_sockeye_version(logger)
    logger.info("Creating top-k lexicon from \"%s\"", args.input)
    logger.info("Reading source and target vocab from \"%s\"", args.model)
    vocab_source = vocab.load_source_vocabs(args.model)[0]
    vocab_target = vocab.load_target_vocabs(args.model)[0]
    logger.info("Building top-%d lexicon", args.k)
    lexicon = TopKLexicon(vocab_source, vocab_target)
    lexicon.create(args.input, args.k)
    lexicon.save(args.output)



def create_block_lexicon_from_file(args):
    setup_main_logger(console=not args.quiet, file_logging=not args.no_logfile, path=args.output + ".log")
    global logger
    logger = logging.getLogger('create-block')
    log_sockeye_version(logger)

    fname = args.input
    model_path = args.model
    output_path = args.output
    with open(fname) as data:
        block_tokens = list(set(token for line in data for token in line.rstrip().split()))
        return create_block_lexicon_for_model(block_tokens, model_path, output_path)


def create_block_lexicon_for_model(block_tokens: List[str], model_path: str, output_path: str, lowercase: bool = False):
    vocab_target = vocab.load_target_vocabs(model_path)[0]
    return create_block_lexicon(block_tokens, vocab_target, output_path, lowercase)


def create_block_lexicon(block_tokens: List[str], vocab_target: vocab.Vocab, output_path: str, lowercase: bool = False):
    if lowercase:
        # Lowercase vocabulary entries + block words:
        # lowercased entries map to multiple word ids
        vocab_target_lower = collections.defaultdict(list)
        for k, v in vocab_target.items():
            vocab_target_lower[k.lower()].append(v)
        block_tokens = [token.lower() for token in block_tokens]
        vocab_target_for_lexicon = dict(vocab_target_lower)
    else:
        vocab_target_for_lexicon = {k: [v] for k, v in vocab_target.items()}

    lexicon = StaticBlockLexicon()
    lexicon.create(block_tokens, vocab_target_for_lexicon)
    lexicon.save(output_path)


def inspect(args):
    from .data_io import tokens2ids
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
        trg_ids = lexicon.get_allowed_trg_ids(np.array(ids))
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

    params_block = subparams.add_parser('create-block', description="Create block lexicon for use during decoding.")
    arguments.add_lexicon_args(params_block, is_for_block_lexicon=True)
    arguments.add_lexicon_create_args(params_block, is_for_block_lexicon=True)
    arguments.add_logging_args(params_block)
    params_block.set_defaults(func=create_block_lexicon_from_file)

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

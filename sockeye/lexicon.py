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
import collections
import operator
import os
import sys
import time
from typing import Dict, Generator, Tuple, Optional

import mxnet as mx
import numpy as np

from . import arguments
from . import constants as C
from . import vocab
from .data_io import smart_open, get_tokens, tokens2ids
from .log import setup_main_logger, log_sockeye_version
from .utils import check_condition

logger = setup_main_logger(__name__, console=True, file_logging=False)


class Lexicon:
    """
    Lexicon model component. Stores lexicon and supports two operations:
        (1) Given source batch, lookup translation distributions in the lexicon
        (2) Given attention score vector and lexicon lookups, compute the lexical bias for the decoder

    :param source_vocab_size: Source vocabulary size.
    :param target_vocab_size: Target vocabulary size.
    :param learn: Whether to adapt lexical biases during training.
    """

    def __init__(self, source_vocab_size: int, target_vocab_size: int, learn: bool = False) -> None:
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        # TODO: once half-precision works, use float16 for this variable to save memory
        self.lexicon = mx.sym.Variable(name=C.LEXICON_NAME,
                                       shape=(self.source_vocab_size,
                                              self.target_vocab_size))
        if not learn:
            logger.info("Fixed lexicon bias terms")
            self.lexicon = mx.sym.BlockGrad(self.lexicon)
        else:
            logger.info("Learning lexicon bias terms")

    def lookup(self, source: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Lookup lexicon distributions for source.

        :param source: Input. Shape: (batch_size, source_seq_len).
        :return: Lexicon distributions for input. Shape: (batch_size, target_vocab_size, source_seq_len).
        """
        return mx.sym.swapaxes(data=mx.sym.Embedding(data=source,
                                                     input_dim=self.source_vocab_size,
                                                     weight=self.lexicon,
                                                     output_dim=self.target_vocab_size,
                                                     name=C.LEXICON_NAME + "_lookup"), dim1=1, dim2=2)

    @staticmethod
    def calculate_lex_bias(source_lexicon: mx.sym.Symbol, attention_prob_score: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Given attention/alignment scores, calculates a weighted sum over lexical distributions
        that serve as a bias for the decoder softmax.
        * https://arxiv.org/pdf/1606.02006.pdf
        * http://www.aclweb.org/anthology/W/W16/W16-4610.pdf

        :param source_lexicon: Lexical biases for sentence Shape: (batch_size, target_vocab_size, source_seq_len).
        :param attention_prob_score: Attention score. Shape: (batch_size, source_seq_len).
        :return: Lexical bias. Shape: (batch_size, 1, target_vocab_size).
        """
        # attention_prob_score: (batch_size, source_seq_len) -> (batch_size, source_seq_len, 1)
        attention_prob_score = mx.sym.expand_dims(attention_prob_score, axis=2)
        # lex_bias: (batch_size, target_vocab_size, 1)
        lex_bias = mx.sym.batch_dot(source_lexicon, attention_prob_score)
        # lex_bias: (batch_size, 1, target_vocab_size)
        lex_bias = mx.sym.swapaxes(data=lex_bias, dim1=1, dim2=2)
        return lex_bias


def initialize_lexicon(cmdline_arg: str, vocab_source: Dict[str, int], vocab_target: Dict[str, int]) -> mx.nd.NDArray:
    """
    Reads a probabilistic word lexicon as given by the commandline argument and converts
    to log probabilities.
    If specified, smooths with custom value, uses 0.001 otherwise.

    :param cmdline_arg: Commandline argument.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :return: Lexicon array. Shape: (vocab_source_size, vocab_target_size).
    """
    fields = cmdline_arg.split(":", 1)
    path = fields[0]
    lexicon = read_lexicon(path, vocab_source, vocab_target)
    assert lexicon.shape == (len(vocab_source), len(vocab_target)), "Invalid lexicon shape"
    eps = 0.001
    if len(fields) == 2:
        eps = float(fields[1])
        check_condition(eps > 0, "epsilon must be >0")
    logger.info("Smoothing lexicon with eps=%.4f", eps)
    lexicon = mx.nd.array(np.log(lexicon + eps))
    return lexicon


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
        # Read lexicon
        src_unk_id = self.vocab_source[C.UNK_SYMBOL]
        trg_unk_id = self.vocab_target[C.UNK_SYMBOL]
        _lex = collections.defaultdict(dict)  # type: Dict[int, Dict[int, float]]
        for src_id, trg_id, prob in lexicon_iterator(path, self.vocab_source, self.vocab_target):
            # Unk token will always be part of target vocab, so no need to track it here
            if src_id == src_unk_id or trg_id == trg_unk_id:
                continue
            _lex[src_id][trg_id] = prob
        # Sort and copy top-k trg_ids to lex array row src_id
        for src_id, trg_entries in _lex.items():
            top_k = list(sorted(trg_entries.items(), key=operator.itemgetter(1), reverse=True))[:k]
            self.lex[src_id, :len(top_k)] = list(trg_id for trg_id, _ in top_k)
            # Free memory after copy
            trg_entries.clear()
        logger.info("Created top-k lexicon from \"%s\", k=%d.", path, k)

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
    global logger
    logger = setup_main_logger('create', console=not args.quiet, file_logging=True, path=args.output + ".log")
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
    global logger
    logger = setup_main_logger('inspect', console=True, file_logging=False)
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

    params_create = subparams.add_parser('create', description="Create top-k lexicon for use during decoding.")
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

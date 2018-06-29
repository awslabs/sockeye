# -*- coding: utf-8 -*-

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import io
import gzip
import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Iterable, Optional

from sockeye import utils


DIR_THIRD_PARTY = "third_party"
DIR_LOGS = "logs"

# Moses, which contains the Moses tokenizer
# License: LGPL-2.1
MOSES_REPO = "https://github.com/moses-smt/mosesdecoder.git"
# Paths to include in sparse checkout
MOSES_SPARSE_CHECKOUT = ["COPYING", "scripts/share", "scripts/tokenizer"]
MOSES_DEST = "mosesdecoder"
MOSES_COMMIT = "686034488aad6ccee564e262aef9e07a85c1b784"

# Subword-nmt, which contains a byte-pair encoding implementation
# License: MIT
SUBWORD_NMT_REPO = "https://github.com/rsennrich/subword-nmt.git"
SUBWORD_NMT_DEST = "subword-nmt"
SUBWORD_NMT_COMMIT = "9a95f9f7400a3a891a9d8168186229a54347fc0b"
SUBWORD_SPECIAL = "@@"

# Unicode underscore
PLACEHOLDER = "▁".encode("utf-8")


def bin_open(fname: str):
    """
    Returns a file descriptor for a plain text or gzipped file, binary read mode
    for subprocess interaction.

    :param fname: The filename to open.
    :return: File descriptor in binary read mode.
    """
    if fname.endswith(".gz"):
        return gzip.open(fname, "rb")
    return open(fname, "rb")


def check_git():
    """Check if git command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["git", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure git is installed and on your path.")


def check_perl():
    """Check if perl command is available."""
    try:
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(["perl", "--version"], stdout=devnull, stderr=devnull)
    except:
        raise RuntimeError("Please make sure perl is installed and on your path.")


def checkout_moses_tokenizer(workspace_dir: str):
    """
    Checkout Moses tokenizer (sparse checkout of Moses).

    :param workspace_dir: Workspace directory.
    """
    # Prerequisites
    check_git()
    check_perl()
    # Check cache
    dest = os.path.join(workspace_dir, DIR_THIRD_PARTY, MOSES_DEST)
    if confirm_checkout(dest, MOSES_COMMIT):
        logging.info("Usable: %s", dest)
        return
    # Need to (re-)checkout
    if os.path.exists(dest):
        shutil.rmtree(dest)
    logging.info("Checkout: %s -> %s", MOSES_REPO, dest)
    os.makedirs(dest)
    log_fname = os.path.join(workspace_dir, DIR_LOGS, "checkout.{}.{}.log".format(MOSES_DEST, os.getpid()))
    with open(log_fname, "wb") as log:
        logging.info("Log: %s", log_fname)
        subprocess.call(["git", "init"], cwd=dest, stdout=log, stderr=log)
        subprocess.call(["git", "remote", "add", "origin", MOSES_REPO], cwd=dest, stdout=log, stderr=log)
        subprocess.call(["git", "config", "core.sparsecheckout", "true"], cwd=dest, stdout=log, stderr=log)
        with open(os.path.join(dest, ".git", "info", "sparse-checkout"), "w") as out:
            for path in MOSES_SPARSE_CHECKOUT:
                print(path, file=out)
        subprocess.call(["git", "pull", "origin", "master"], cwd=dest, stdout=log, stderr=log)
        subprocess.call(["git", "checkout", MOSES_COMMIT], cwd=dest, stdout=log, stderr=log)


def checkout_subword_nmt(workspace_dir: str):
    """
    Checkout subword-nmt implementation of byte-pair encoding.

    :param workspace_dir: Workspace third-party directory.
    """
    # Prerequisites
    check_git()
    # Check cache
    dest = os.path.join(workspace_dir, DIR_THIRD_PARTY, SUBWORD_NMT_DEST)
    if confirm_checkout(dest, SUBWORD_NMT_COMMIT):
        logging.info("Usable: %s", dest)
        return
    # Need to (re-)checkout
    if os.path.exists(dest):
        shutil.rmtree(dest)
    logging.info("Checkout: %s -> %s", SUBWORD_NMT_REPO, dest)
    log_fname = os.path.join(workspace_dir, DIR_LOGS, "checkout.{}.{}.log".format(SUBWORD_NMT_DEST, os.getpid()))
    with open(log_fname, "wb") as log:
        logging.info("Log: %s", log_fname)
        subprocess.call(["git", "clone", SUBWORD_NMT_REPO, dest], stdout=log, stderr=log)
        subprocess.call(["git", "checkout", SUBWORD_NMT_COMMIT], cwd=dest, stdout=log, stderr=log)


def confirm_checkout(dest: str, commit: str) -> bool:
    """
    Confirm that git repository is checked out.

    :param dest: Local directory for checkout.
    :param commit: Git commit.
    :return: True if checkout is usable.
    """
    usable = False
    if os.path.exists(dest):
        try:
            rev = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=dest).decode("utf-8").strip()
            usable = (rev == commit)
        except subprocess.CalledProcessError:
            pass
        if not usable:
            logging.info("Problem with %s, requires new checkout.", dest)
    return usable


def call_moses_tokenizer(workspace_dir: str,
                         input_fname: str,
                         output_fname: str,
                         lang_code: str,
                         num_threads: int = 4):
    """
    Call Moses tokenizer.

    :param workspace_dir: Workspace third-party directory where Moses
                            tokenizer is checked out.
    :param input_fname: Path of raw input file, plain text or gzipped.
    :param output_fname: Path of tokenized output file, gzipped.
    :param lang_code: Language code for rules and non-breaking prefixes.
    :param num_threads: Number of threads to use.
    """
    tokenizer_fname = os.path.join(workspace_dir,
                                   DIR_THIRD_PARTY,
                                   MOSES_DEST,
                                   "scripts",
                                   "tokenizer",
                                   "tokenizer.perl")
    with bin_open(input_fname) as inp, gzip.open(output_fname, "wb") as out, open(os.devnull, "wb") as devnull:
        tokenizer = subprocess.Popen(["perl", tokenizer_fname, "-l", lang_code, "-threads", str(num_threads)],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=devnull)
        tokenizer_thread = threading.Thread(target=copy_out, args=(tokenizer.stdout, out))
        tokenizer_thread.start()
        for line in inp:
            tokenizer.stdin.write(line)
        tokenizer.stdin.close()
        tokenizer_thread.join()
        tokenizer.wait()


def call_moses_detokenizer(workspace_dir: str, input_fname: str, output_fname: str, lang_code: Optional[str] = None):
    """
    Call Moses detokenizer.

    :param workspace_dir: Workspace third-party directory where Moses
                          tokenizer is checked out.
    :param input_fname: Path of tokenized input file, plain text or gzipped.
    :param output_fname: Path of tokenized output file, plain text.
    :param lang_code: Language code for rules and non-breaking prefixes.  Can be
                      None if unknown (using pre-tokenized data), which will
                      cause the tokenizer to default to English.
    """
    detokenizer_fname = os.path.join(workspace_dir,
                                     DIR_THIRD_PARTY,
                                     MOSES_DEST,
                                     "scripts",
                                     "tokenizer",
                                     "detokenizer.perl")
    with bin_open(input_fname) as inp, open(output_fname, "wb") as out, open(os.devnull, "wb") as devnull:
        command = ["perl", detokenizer_fname]
        if lang_code:
            command.append("-l")
            command.append(lang_code)
        detokenizer = subprocess.Popen(command,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=devnull)
        detokenizer_thread = threading.Thread(target=copy_out, args=(detokenizer.stdout, out))
        detokenizer_thread.start()
        for line in inp:
            detokenizer.stdin.write(line)
        detokenizer.stdin.close()
        detokenizer_thread.join()
        detokenizer.wait()


def call_learn_bpe(workspace_dir: str, source_fname: str, target_fname: str, model_fname: str, num_ops: int = 32000):
    """
    Call script to learn byte-pair encoding model.

    :param workspace_dir: Workspace third-party directory where subword-nmt is
                            checked out.
    :param source_fname: Path of source corpus file, plain text or gzipped.
    :param target_fname: Path of target corpus file, plain text or gzipped.
    :param model_fname: Path to write out model.
    :param num_ops: Number of operations.
    """
    learn_bpe_fname = os.path.join(workspace_dir, DIR_THIRD_PARTY, SUBWORD_NMT_DEST, "learn_bpe.py")
    with bin_open(source_fname) as src_in, bin_open(target_fname) as trg_in, open(model_fname, "wb") as out:
        learn_bpe = subprocess.Popen([sys.executable, learn_bpe_fname, "-s", str(num_ops)],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        learn_bpe_thread = threading.Thread(target=copy_out, args=(learn_bpe.stdout, out))
        learn_bpe_thread.start()
        for inp in (src_in, trg_in):
            for line in inp:
                learn_bpe.stdin.write(line)
        learn_bpe.stdin.close()
        learn_bpe_thread.join()
        learn_bpe.wait()


def call_apply_bpe(workspace_dir: str, input_fname: str, output_fname: str, model_fname: str):
    """
    Call BPE apply script.

    :param workspace_dir: Workspace directory where subword-nmt is checked out.
    :param input_fname: Path of tokenized input file, plain text or gzipped.
    :param output_fname: Path of byte-pair encoded output file, gzipped.
    :param model_fname: Path of BPE model file (codes).
    """
    apply_bpe_fname = os.path.join(workspace_dir, DIR_THIRD_PARTY, SUBWORD_NMT_DEST, "apply_bpe.py")
    with bin_open(input_fname) as inp, gzip.open(output_fname, "wb") as out:
        apply_bpe = subprocess.Popen([sys.executable, apply_bpe_fname, "-c", model_fname],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        apply_bpe_thread = threading.Thread(target=copy_out, args=(apply_bpe.stdout, out, True))
        apply_bpe_thread.start()
        for line in inp:
            # Use an empty line placeholder to avoid blank line duplication
            # issues with BPE script
            if not line.strip():
                line = PLACEHOLDER + b"\n"
            apply_bpe.stdin.write(line)
        apply_bpe.stdin.close()
        apply_bpe_thread.join()
        apply_bpe.wait()


def merge_bpe(input_fname: str, output_fname: str):
    """
    Merge byte-pair encoded sub-words.

    :param input_fname: Path of byte-pair encoded input file, plain text or
                        gzipped.
    :param output_fname: Path of tokenized output file, plain text.
    """
    with utils.smart_open(input_fname, "r") as inp, open(output_fname, "w", encoding="utf-8") as out:
        for line in inp:
            # Merge on special markers and strip stray markers (end of line)
            merged = line.replace(SUBWORD_SPECIAL + " ", "").replace(SUBWORD_SPECIAL, "")
            out.write(merged)


def copy_out(source: Iterable[bytes], dest: io.BytesIO, use_placeholders: bool = False):
    """
    Copy lines from source to destination.

    :param source: Source line iterable.
    :param dest: Destination open file.
    :param use_placeholders: When true, convert lines containing placeholders to
                             empty lines and drop true empty lines (assume to be
                             spuriously generated).
    """
    for line in source:
        if use_placeholders:
            # True empty lines are assumed to be spurious as the placeholder
            # should be passed through
            if not line.strip():
                continue
            if line.startswith(PLACEHOLDER):
                line = b"\n"
        dest.write(line)

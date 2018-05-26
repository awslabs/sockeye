#!/usr/bin/env python3

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

import argparse
import gzip
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, IO, Iterable, List, Optional, Tuple
import urllib.request
import zipfile

# Make sure sockeye is on the system path
try:
    from sockeye import constants as C
    from sockeye import utils
except ImportError:
    SOCKEYE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise RuntimeError("Please install the sockeye module or add the sockeye root directory to your Python path. Ex: export PYTHONPATH=%s"
                       % SOCKEYE_ROOT)

from contrib.autopilot.tasks import ARCHIVE_NONE, ARCHIVE_TAR, ARCHIVE_ZIP
from contrib.autopilot.tasks import TEXT_UTF8_RAW, TEXT_UTF8_RAW_SGML, TEXT_UTF8_RAW_BITEXT
from contrib.autopilot.tasks import TEXT_UTF8_RAW_BITEXT_REVERSE, TEXT_REQUIRES_TOKENIZATION
from contrib.autopilot.tasks import TEXT_UTF8_TOKENIZED
from contrib.autopilot.tasks import RAW_FILES
from contrib.autopilot.tasks import Task, TASKS
from contrib.autopilot.models import MODELS, MODEL_NONE, MODEL_TEST_ARGS
from contrib.autopilot.models import DECODE_ARGS, DECODE_STANDARD
from contrib.autopilot import third_party


# Formats for custom files
CUSTOM_UTF8_RAW = "raw"
CUSTOM_UTF8_TOK = "tok"
CUSTOM_UTF8_BPE = "bpe"
CUSTOM_TEXT_TYPES = [CUSTOM_UTF8_RAW, CUSTOM_UTF8_TOK, CUSTOM_UTF8_BPE]

# Special file names
DIR_SOCKEYE_AUTOPILOT = "sockeye_autopilot"
FILE_WORKSPACE = ".workspace"
FILE_COMPLETE = ".complete"

# Sub-task directory and file names
DIR_CACHE = "cache"
DIR_LOGS = "logs"
DIR_SYSTEMS = "systems"
DIR_DATA = "data"
DIR_RAW = "raw"
DIR_TOK = "tok"
DIR_BPE = "bpe"
PREFIX_TRAIN = "train."
PREFIX_DEV = "dev."
PREFIX_TEST = "test."
DATA_SRC = "src"
DATA_TRG = "trg"
SUFFIX_SRC_GZ = DATA_SRC + ".gz"
SUFFIX_TRG_GZ = DATA_TRG + ".gz"
DIR_BPE_MODEL = "model.bpe"
FILE_BPE_CODES = "codes"
DIR_PREFIX_MODEL = "model."
DIR_RESULTS = "results"
FILE_COMMAND = "command.{}.sh"
SUFFIX_COMMAND = "command.sh"
SUFFIX_BPE = "bpe"
SUFFIX_TOK = "tok"
SUFFIX_DETOK = "detok"
SUFFIX_BLEU = "bleu"
SUFFIX_SACREBLEU = "sacrebleu"
SUFFIX_TEST = ".test"

# Reasonable defaults for model averaging
AVERAGE_NUM_CHECKPOINTS = 8
AVERAGE_METRIC = "perplexity"
AVERAGE_STRATEGY = "best"
PARAMS_BEST_SINGLE = "params.best.single"
PARAMS_AVERAGE = "params.average"

# Scaled down settings for test mode
TEST_BPE_OPS = 1024


def identify_raw_files(task: Task, test_mode: bool = False) -> List[str]:
    """
    Identify raw files that need to be downloaded for a given task.

    :param task: Sequence-to-sequence task.
    :param test_mode: Run in test mode, only downloading test data.
    :return: List of raw file names.
    """
    raw_files = set()
    all_sets = [task.test,] if test_mode else [task.train, task.dev, task.test]
    for file_sets in all_sets:
        for file_set in file_sets:
            for fname in file_set[:2]:
                raw_file = fname.split("/", 1)[0]
                if raw_file not in RAW_FILES:
                    raise RuntimeError("Unknown raw file %s found in path %s" % (raw_file, fname))
                raw_files.add(raw_file)
    return sorted(raw_files)


def download_extract_raw_files(names: List[str], cache_dir: str, dest_dir: str):
    """
    Download and extract raw files, making use of a cache directory.
    - Downloaded files are verified by MD5 sum.
    - Extraction overwrites existing files.

    :param names: List of raw file names in RAW_FILES.
    :param cache_dir: Cache directory for downloading raw files.
    :param dest_dir: Destination directory for extracting raw files.
    """

    for name in names:
        raw_file = RAW_FILES[name]
        local_dir = os.path.join(cache_dir, name)
        local_fname = os.path.join(local_dir, os.path.basename(raw_file.url))

        # Download file if not present
        if not os.path.exists(local_dir):
            logging.info("Create: %s", local_dir)
            os.makedirs(local_dir)
        if not os.path.exists(local_fname):
            logging.info("Download: %s -> %s", raw_file.url, local_fname)
            urllib.request.urlretrieve(raw_file.url, local_fname)

        # Check MD5 sum, attempt one re-download on mismatch
        md5 = md5sum(local_fname)
        if not md5 == raw_file.md5:
            logging.info("MD5 mismatch for %s, attempt re-download %s", local_fname, raw_file.url)
            urllib.request.urlretrieve(raw_file.url, local_fname)
            md5 = md5sum(local_fname)
            if not md5 == raw_file.md5:
                raise RuntimeError("MD5 mismatch for %s after re-download.  Check validity of %s"
                                   % (local_fname, raw_file.url))
        logging.info("Confirmed MD5: %s (%s)", local_fname, md5)

        # Extract file(s), overwriting directory if exists
        extract_path = os.path.join(dest_dir, name)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path)
        logging.info("Extract: %s -> %s", local_fname, extract_path)
        if raw_file.archive_type == ARCHIVE_NONE:
            os.symlink(local_fname, os.path.join(extract_path, os.path.basename(local_fname)))
        elif raw_file.archive_type == ARCHIVE_TAR:
            tar = tarfile.open(local_fname)
            tar.extractall(path=extract_path)
        elif raw_file.archive_type == ARCHIVE_ZIP:
            zipf = zipfile.ZipFile(local_fname, "r")
            zipf.extractall(path=extract_path)
        else:
            raise RuntimeError("Unknown archive type: %s" % raw_file.archive_type)


def md5sum(fname: str) -> str:
    """Compute MD5 sum of file."""
    with open(fname, "rb") as inp:
        md5 = hashlib.md5(inp.read()).hexdigest()
    return md5


def populate_parallel_text(extract_dir: str,
                           file_sets: List[Tuple[str, str, str]],
                           dest_prefix: str,
                           keep_separate: bool,
                           head_n: int = 0):
    """
    Create raw parallel train, dev, or test files with a given prefix.

    :param extract_dir: Directory where raw files (inputs) are extracted.
    :param file_sets: Sets of files to use.
    :param dest_prefix: Prefix for output files.
    :param keep_separate: True if each file set (source-target pair) should have
                          its own file (used for test sets).
    :param head_n: If N>0, use only the first N lines (used in test mode).
    """
    source_out = None  # type: IO[Any]
    target_out = None  # type: IO[Any]
    lines_written = 0
    # Single output file for each side
    if not keep_separate:
        source_dest = dest_prefix + SUFFIX_SRC_GZ
        target_dest = dest_prefix + SUFFIX_TRG_GZ
        logging.info("Populate: %s %s", source_dest, target_dest)
        source_out = gzip.open(source_dest, "wt", encoding="utf-8")
        target_out = gzip.open(target_dest, "wt", encoding="utf-8")
    for i, (source_fname, target_fname, text_type) in enumerate(file_sets):
        # One output file per input file for each side
        if keep_separate:
            if source_out:
                source_out.close()
            if target_out:
                target_out.close()
            source_dest = dest_prefix + str(i) + "." + SUFFIX_SRC_GZ
            target_dest = dest_prefix + str(i) + "." + SUFFIX_TRG_GZ
            logging.info("Populate: %s %s", source_dest, target_dest)
            source_out = gzip.open(source_dest, "wt", encoding="utf-8")
            target_out = gzip.open(target_dest, "wt", encoding="utf-8")
        for source_line, target_line in zip(
                plain_text_iter(os.path.join(extract_dir, source_fname), text_type, DATA_SRC),
                plain_text_iter(os.path.join(extract_dir, target_fname), text_type, DATA_TRG)):
            # Only write N lines total if requested, but reset per file when
            # keeping files separate
            if head_n > 0 and lines_written >= head_n:
                if keep_separate:
                    lines_written = 0
                break
            source_out.write("{}\n".format(source_line))
            target_out.write("{}\n".format(target_line))
            lines_written += 1
    source_out.close()
    target_out.close()


def copy_parallel_text(file_list: List[str], dest_prefix: str):
    """
    Copy pre-compiled raw parallel files with a given prefix.  Perform
    whitespace character normalization to ensure that only ASCII newlines are
    considered line breaks.

    :param file_list: List of file pairs to use.
    :param dest_prefix: Prefix for output files.
    """
    # Group files into source-target pairs
    file_sets = []
    for i in range(0, len(file_list), 2):
        file_sets.append((file_list[i], file_list[i + 1]))
    multiple_sets = len(file_sets) > 1
    for i, (source_fname, target_fname) in enumerate(file_sets):
        if multiple_sets:
            source_dest = dest_prefix + str(i) + "." + SUFFIX_SRC_GZ
            target_dest = dest_prefix + str(i) + "." + SUFFIX_TRG_GZ
        else:
            source_dest = dest_prefix + SUFFIX_SRC_GZ
            target_dest = dest_prefix + SUFFIX_TRG_GZ
        logging.info("Populate: %s %s", source_dest, target_dest)
        with gzip.open(source_dest, "wb") as source_out, gzip.open(target_dest, "wb") as target_out:
            with third_party.bin_open(source_fname) as inp:
                for line in inp:
                    line = (re.sub(r"\s", " ", line.decode("utf-8"))).encode("utf-8") + b"\n"
                    source_out.write(line)
            with third_party.bin_open(target_fname) as inp:
                for line in inp:
                    line = (re.sub(r"\s", " ", line.decode("utf-8"))).encode("utf-8") + b"\n"
                    target_out.write(line)


def plain_text_iter(fname: str, text_type: str, data_side: str) -> Iterable[str]:
    """
    Extract plain text from file as iterable.  Also take steps to ensure that
    whitespace characters (including unicode newlines) are normalized and
    outputs are line-parallel with inputs considering ASCII newlines only.

    :param fname: Path of possibly gzipped input file.
    :param text_type: One of TEXT_*, indicating data format.
    :param data_side: DATA_SRC or DATA_TRG.
    """
    if text_type in (TEXT_UTF8_RAW, TEXT_UTF8_TOKENIZED):
        with third_party.bin_open(fname) as inp:
            for line in inp:
                line = re.sub(r"\s", " ", line.decode("utf-8"))
                yield line.strip()
    elif text_type == TEXT_UTF8_RAW_SGML:
        with third_party.bin_open(fname) as inp:
            for line in inp:
                line = re.sub(r"\s", " ", line.decode("utf-8"))
                if line.startswith("<seg "):
                    # Extract segment text
                    text = re.sub(r"<seg.*?>(.*)</seg>.*?", "\\1", line)
                    text = re.sub(r"\s+", " ", text.strip())
                    # Unescape XML entities
                    text = text.replace("&quot;", "\"")
                    text = text.replace("&apos;", "'")
                    text = text.replace("&lt;", "<")
                    text = text.replace("&gt;", ">")
                    text = text.replace("&amp;", "&")
                    yield text
    elif text_type in (TEXT_UTF8_RAW_BITEXT, TEXT_UTF8_RAW_BITEXT_REVERSE):
        # Select source or target field, reversing if needed
        if text_type == TEXT_UTF8_RAW_BITEXT:
            field_id = 0 if data_side == DATA_SRC else 1
        else:
            field_id = 1 if data_side == DATA_SRC else 0
        with third_party.bin_open(fname) as inp:
            for line in inp:
                line = re.sub(r"\s", " ", line.decode("utf-8"))
                fields = line.split("|||")
                yield fields[field_id].strip()
    else:
        raise RuntimeError("Unknown text type: %s" % text_type)


def touch_file(fname: str):
    """Create a file if not present, update access time."""
    # Reference not needed since there will be no reads or writes
    with open(fname, "a"):
        os.utime(fname, None)


def renew_step_dir(step_dir: str):
    """Delete step directory if exists and create, reporting actions."""
    if os.path.exists(step_dir):
        logging.info("Remove unfinished step %s", step_dir)
        shutil.rmtree(step_dir)
    logging.info("Create: %s", step_dir)
    os.makedirs(step_dir)


def call_sockeye_train(args: List[str],
                       bpe_dir: str,
                       model_dir: str,
                       log_fname: str,
                       num_gpus: int,
                       test_mode: bool = False):
    """
    Call sockeye.train with specified arguments on prepared inputs.  Will resume
    partial training or skip training if model is already finished.  Record
    command for future use.

    :param args: Command line arguments for sockeye.train.
    :param bpe_dir: Directory of BPE-encoded input data.
    :param model_dir: Model output directory.
    :param log_fname: Location to write log file.
    :param num_gpus: Number of GPUs to use for training (0 for CPU).
    :param test_mode: Run in test mode, stopping after a small number of
                      updates.
    """
    # Inputs and outputs
    fnames = ["--source={}".format(os.path.join(bpe_dir, PREFIX_TRAIN + SUFFIX_SRC_GZ)),
              "--target={}".format(os.path.join(bpe_dir, PREFIX_TRAIN + SUFFIX_TRG_GZ)),
              "--validation-source={}".format(os.path.join(bpe_dir, PREFIX_DEV + SUFFIX_SRC_GZ)),
              "--validation-target={}".format(os.path.join(bpe_dir, PREFIX_DEV + SUFFIX_TRG_GZ)),
              "--output={}".format(model_dir)]
    # Assemble command
    command = [sys.executable, "-m", "sockeye.train"] + fnames + args
    # Request GPUs or specify CPU
    if num_gpus > 0:
        command.append("--device-ids=-{}".format(num_gpus))
    else:
        command.append("--use-cpu")
    # Test mode trains a smaller model for a small number of steps
    if test_mode:
        command += MODEL_TEST_ARGS
    command_fname = os.path.join(model_dir, FILE_COMMAND.format("sockeye.train"))
    # Run unless training already finished
    if not os.path.exists(command_fname):
        # Call Sockeye training
        with open(log_fname, "wb") as log:
            logging.info("sockeye.train: %s", model_dir)
            logging.info("Log: %s", log_fname)
            logging.info("(This step can take several days. See log file or TensorBoard for progress)")
            subprocess.check_call(command, stderr=log)
        # Record successful command
        logging.info("Command: %s", command_fname)
        print_command(command, command_fname)


def call_sockeye_average(model_dir: str, log_fname: str):
    """
    Call sockeye.average with reasonable defaults.

    :param model_dir: Trained model directory.
    :param log_fname: Location to write log file.
    """
    params_best_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
    params_best_single_fname = os.path.join(model_dir, PARAMS_BEST_SINGLE)
    params_average_fname = os.path.join(model_dir, PARAMS_AVERAGE)
    command = [sys.executable,
               "-m",
               "sockeye.average",
               "--metric={}".format(AVERAGE_METRIC),
               "-n",
               str(AVERAGE_NUM_CHECKPOINTS),
               "--output={}".format(params_average_fname),
               "--strategy={}".format(AVERAGE_STRATEGY),
               model_dir]
    command_fname = os.path.join(model_dir, FILE_COMMAND.format("sockeye.average"))
    # Run average if not previously run
    if not os.path.exists(command_fname):
        # Re-link best point to best single point
        os.symlink(os.path.basename(os.path.realpath(params_best_fname)), params_best_single_fname)
        os.remove(params_best_fname)
        # Call Sockeye average
        with open(log_fname, "wb") as log:
            logging.info("sockeye.average: %s", os.path.join(model_dir, params_best_fname))
            logging.info("Log: %s", log_fname)
            subprocess.check_call(command, stderr=log)
        # Link averaged point as new best
        os.symlink(PARAMS_AVERAGE, params_best_fname)
        # Record successful command
        logging.info("Command: %s", command_fname)
        print_command(command, command_fname)


def call_sockeye_translate(args: List[str],
                           input_fname: str,
                           output_fname: str,
                           model_dir: str,
                           log_fname: str,
                           use_cpu: bool):
    """
    Call sockeye.translate with specified arguments using a trained model.

    :param args: Command line arguments for sockeye.translate.
    :param input_fname: Input file (byte-pair encoded).
    :param output_fname: Raw decoder output file.
    :param model_dir: Model output directory.
    :param log_fname: Location to write log file.
    :param use_cpu: Use CPU instead of GPU for decoding.
    """
    # Inputs and outputs
    fnames = ["--input={}".format(input_fname),
              "--output={}".format(output_fname),
              "--models={}".format(model_dir)]
    # Assemble command
    command = [sys.executable, "-m", "sockeye.translate"] + fnames + args
    # Request GPUs or specify CPU
    if use_cpu:
        command.append("--use-cpu")
    command_fname = output_fname + "." + SUFFIX_COMMAND
    # Run unless translate already finished
    if not os.path.exists(command_fname):
        # Call Sockeye translate
        with open(log_fname, "wb") as log:
            logging.info("sockeye.translate: %s -> %s", input_fname, output_fname)
            logging.info("Log: %s", log_fname)
            subprocess.check_call(command, stderr=log)
        # Cleanup redundant log file
        try:
            os.remove(output_fname + ".log")
        except FileNotFoundError:
            pass

        # Record successful command
        logging.info("Command: %s", command_fname)
        print_command(command, command_fname)


def call_sacrebleu(input_fname: str, ref_fname: str, output_fname: str, log_fname: str, tokenized: bool = False):
    """
    Call contrib.sacrebleu.sacrebleu on tokenized or detokenized inputs.

    :param input_fname: Input translation file.
    :param ref_fname: Reference translation file.
    :param output_fname: Output score file.
    :param log_fname: Location to write log file.
    :param tokenized: Whether inputs are tokenized (or byte-pair encoded).
    """
    # Assemble command
    command = [sys.executable,
               "-m",
               "contrib.sacrebleu.sacrebleu",
               "--score-only",
               "--input={}".format(input_fname),
               ref_fname]
    # Already tokenized?
    if tokenized:
        command.append("--tokenize=none")
    # Call sacrebleu
    with open(log_fname, "wb") as log:
        logging.info("contrib.sacrebleu.sacrebleu: %s -> %s", input_fname, output_fname)
        logging.info("Log: %s", log_fname)
        score = subprocess.check_output(command, stderr=log)
    # Record successful score
    with open(output_fname, "wb") as out:
        out.write(score)


def print_command(command: List[str], fname: str):
    """
    Format and print command to file.

    :param command: Command in args list form.
    :param fname: File name to write out.
    """
    with open(fname, "w", encoding="utf-8") as out:
        print(" \\\n".join(command), file=out)


def run_steps(args: argparse.Namespace):
    """Run all steps required to complete task.  Called directly from main."""

    logging.basicConfig(level=logging.INFO, format="sockeye.autopilot: %(message)s")

    # (1) Establish task

    logging.info("=== Start Autopilot ===")
    # Listed task
    if args.task:
        task = TASKS[args.task]
        logging.info("Task: %s", task.description)
        logging.info("URL: %s", task.url)

        def report_data(file_sets):
            for file_set in file_sets:
                for fname in file_set[:2]:
                    logging.info("    %s", fname)

        logging.info("  Train:")
        report_data(task.train)
        logging.info("  Dev:")
        report_data(task.dev)
        logging.info("  Test:")
        report_data(task.test)
    # Custom task
    else:
        logging.info("Task: custom")
    # Source and target language codes
    lang_codes = (task.src_lang, task.trg_lang) if args.task else args.custom_lang

    # (2) Establish workspace and task directories

    logging.info("=== Establish working directories ===")
    logging.info("Workspace: %s", args.workspace)
    special_fname = os.path.join(args.workspace, FILE_WORKSPACE)
    if not os.path.exists(args.workspace):
        logging.info("Create: %s", args.workspace)
        os.makedirs(args.workspace)
        touch_file(special_fname)
    else:
        if not os.path.exists(special_fname):
            raise RuntimeError("Directory %s exists but %s does not, stopping to avoid overwriting files in non-workspace directory"
                            % (args.workspace, special_fname))

    dir_third_party = os.path.join(args.workspace, third_party.DIR_THIRD_PARTY)
    dir_cache = os.path.join(args.workspace, DIR_CACHE)
    dir_logs = os.path.join(args.workspace, DIR_LOGS)
    dir_systems = os.path.join(args.workspace, DIR_SYSTEMS)
    task_name = args.task if args.task else args.custom_task
    if args.test:
        task_name += SUFFIX_TEST
    dir_task = os.path.join(dir_systems, task_name)
    for dirname in (dir_third_party, dir_cache, dir_logs, dir_systems, dir_task):
        if os.path.exists(dirname):
            logging.info("Exists: %s", dirname)
        else:
            logging.info("Create: %s", dirname)
            os.makedirs(dirname)

    # (3) Checkout necessary tools

    logging.info("=== Checkout third-party tools ===")
    # Requires tokenization?
    if args.task or args.custom_text_type == CUSTOM_UTF8_RAW:
        third_party.checkout_moses_tokenizer(args.workspace)
    # Requires byte-pair encoding?
    if args.task or args.custom_text_type in (CUSTOM_UTF8_RAW, CUSTOM_UTF8_TOK):
        third_party.checkout_subword_nmt(args.workspace)

    # (4) Populate train/dev/test data

    # This step also normalizes whitespace on data population or copy, ensuring
    # that for all input data, only ASCII newlines are considered line breaks.
    logging.info("=== Populate train/dev/test data ===")
    step_dir_raw = os.path.join(dir_task, DIR_DATA, DIR_RAW)
    complete_fname = os.path.join(step_dir_raw, FILE_COMPLETE)
    if os.path.exists(complete_fname):
        logging.info("Re-use completed step: %s", step_dir_raw)
    else:
        # Listed task
        if args.task:
            raw_files = identify_raw_files(task, test_mode=args.test)
            with tempfile.TemporaryDirectory(prefix="raw.", dir=dir_task) as raw_dir:
                # Download (or locate in cache) and extract raw files to temp directory
                logging.info("=== Download and extract raw files ===")
                download_extract_raw_files(raw_files, dir_cache, raw_dir)
                # Copy required files to train/dev/test
                logging.info("=== Create input data files ===")
                renew_step_dir(step_dir_raw)
                # Test mode uses the full test set as training data and the
                # first line of the test set as dev and test data
                populate_parallel_text(raw_dir,
                                       task.test if args.test else task.train,
                                       os.path.join(step_dir_raw, PREFIX_TRAIN),
                                       False)
                populate_parallel_text(raw_dir,
                                       task.test if args.test else task.dev,
                                       os.path.join(step_dir_raw, PREFIX_DEV),
                                       False,
                                       head_n=1 if args.test else 0)
                populate_parallel_text(raw_dir,
                                       task.test,
                                       os.path.join(step_dir_raw, PREFIX_TEST),
                                       True,
                                       head_n=1 if args.test else 0)
        # Custom task
        else:
            logging.info("=== Copy input data files ===")
            renew_step_dir(step_dir_raw)
            copy_parallel_text(args.custom_train, os.path.join(step_dir_raw, PREFIX_TRAIN))
            copy_parallel_text(args.custom_dev, os.path.join(step_dir_raw, PREFIX_DEV))
            copy_parallel_text(args.custom_test, os.path.join(step_dir_raw, PREFIX_TEST))
        # Record success
        touch_file(complete_fname)
        logging.info("Step complete: %s", step_dir_raw)

    # (5) Tokenize train/dev/test data

    # Task requires tokenization if _any_ raw file is not already tokenized
    requires_tokenization = False
    if args.task:
        for file_sets in (task.train, task.dev, task.test):
            for _, _, text_type in file_sets:
                if text_type in TEXT_REQUIRES_TOKENIZATION:
                    requires_tokenization = True
    else:
        if args.custom_text_type == CUSTOM_UTF8_RAW:
            requires_tokenization = True
    logging.info("=== Tokenize train/dev/test data ===")
    step_dir_tok = os.path.join(dir_task, DIR_DATA, DIR_TOK)
    complete_fname = os.path.join(step_dir_tok, FILE_COMPLETE)
    if os.path.exists(complete_fname):
        logging.info("Re-use completed step: %s", step_dir_tok)
    else:
        renew_step_dir(step_dir_tok)

        # Tokenize each data file using the appropriate language code OR link
        # raw file if already tokenized.
        for fname in os.listdir(step_dir_raw):
            if fname.startswith("."):
                continue
            input_fname = os.path.join(step_dir_raw, fname)
            output_fname = os.path.join(step_dir_tok, fname)
            if requires_tokenization:
                lang_code = lang_codes[0] if fname.endswith(SUFFIX_SRC_GZ) else lang_codes[1]
                logging.info("Tokenize (%s): %s -> %s", lang_code, input_fname, output_fname)
                third_party.call_moses_tokenizer(workspace_dir=args.workspace,
                                                 input_fname=input_fname,
                                                 output_fname=output_fname,
                                                 lang_code=lang_code)
            else:
                logging.info("Link pre-tokenized: %s -> %s", input_fname, output_fname)
                os.symlink(os.path.join("..", DIR_RAW, fname), output_fname)
        # Record success
        touch_file(complete_fname)
        logging.info("Step complete: %s", step_dir_tok)

    # (6) Learn byte-pair encoding model

    # Task requires byte-pair encoding unless using pre-encoded custom data
    skip_bpe = (not args.task) and args.custom_text_type == CUSTOM_UTF8_BPE
    logging.info("=== Learn byte-pair encoding model ===")
    step_dir_bpe_model = os.path.join(dir_task, DIR_BPE_MODEL)
    complete_fname = os.path.join(step_dir_bpe_model, FILE_COMPLETE)
    if os.path.exists(complete_fname):
        logging.info("Re-use completed step: %s", step_dir_bpe_model)
    else:
        renew_step_dir(step_dir_bpe_model)
        if skip_bpe:
            logging.info("BPE model not required for pre-encoded data")
        else:
            source_fname = os.path.join(step_dir_tok, PREFIX_TRAIN + SUFFIX_SRC_GZ)
            target_fname = os.path.join(step_dir_tok, PREFIX_TRAIN + SUFFIX_TRG_GZ)
            codes_fname = os.path.join(step_dir_bpe_model, FILE_BPE_CODES)
            num_ops = task.bpe_op if args.task else args.custom_bpe_op
            if args.test:
                num_ops = TEST_BPE_OPS
            logging.info("BPE Learn (%s): %s + %s -> %s", num_ops, source_fname, target_fname, codes_fname)
            third_party.call_learn_bpe(workspace_dir=args.workspace,
                                       source_fname=source_fname,
                                       target_fname=target_fname,
                                       model_fname=codes_fname,
                                       num_ops=num_ops)
        # Record success
        touch_file(complete_fname)
        logging.info("Step complete: %s", step_dir_bpe_model)

    # (7) Byte-pair encode data

    logging.info("=== Byte-pair encode train/dev/test data ===")
    step_dir_bpe = os.path.join(dir_task, DIR_DATA, DIR_BPE)
    complete_fname = os.path.join(step_dir_bpe, FILE_COMPLETE)
    if os.path.exists(complete_fname):
        logging.info("Re-use completed step: %s", step_dir_bpe)
    else:
        renew_step_dir(step_dir_bpe)
        # Encode each data file
        for fname in os.listdir(step_dir_tok):
            if fname.startswith("."):
                continue
            input_fname = os.path.join(step_dir_tok, fname)
            output_fname = os.path.join(step_dir_bpe, fname)
            if skip_bpe:
                logging.info("Link pre-encoded: %s -> %s", input_fname, output_fname)
                os.symlink(os.path.join("..", DIR_TOK, fname), output_fname)
            else:
                codes_fname = os.path.join(step_dir_bpe_model, FILE_BPE_CODES)
                logging.info("BPE: %s -> %s", input_fname, output_fname)
                third_party.call_apply_bpe(workspace_dir=args.workspace,
                                           input_fname=input_fname,
                                           output_fname=output_fname,
                                           model_fname=codes_fname)
        # Record success
        touch_file(complete_fname)
        logging.info("Step complete: %s", step_dir_bpe)

    # Done if only running data preparation steps
    if args.model == MODEL_NONE:
        return

    # (8) Run Sockeye training

    logging.info("=== Train translation model ===")
    logging.info("Model: %s", args.model)
    step_dir_model = os.path.join(dir_task, DIR_PREFIX_MODEL + args.model)
    complete_fname = os.path.join(step_dir_model, FILE_COMPLETE)
    if os.path.exists(complete_fname):
        logging.info("Re-use completed step: %s", step_dir_model)
    else:
        log_fname = os.path.join(args.workspace,
                                 DIR_LOGS,
                                 "sockeye.{{}}.{}.{}.{}.log".format(task_name, args.model, os.getpid()))
        call_sockeye_train(MODELS[args.model],
                           step_dir_bpe,
                           step_dir_model,
                           log_fname.format("train"),
                           args.gpus,
                           test_mode=args.test)
        call_sockeye_average(step_dir_model, log_fname.format("average"))
        # Record success
        touch_file(complete_fname)
        logging.info("Step complete: %s", step_dir_model)

    # (9) Decode test sets

    logging.info("=== Decode test sets ===")
    logging.info("Settings: %s", args.decode_settings)
    step_dir_results = os.path.join(dir_task, DIR_RESULTS)
    if not os.path.exists(step_dir_results):
        logging.info("Create: %s", step_dir_results)
        os.makedirs(step_dir_results)
    # To collect BPE output names
    output_fnames_bpe = []
    # For each test file
    for fname in os.listdir(step_dir_bpe):
        if fname.startswith(PREFIX_TEST) and fname.endswith(SUFFIX_SRC_GZ):
            input_fname = os.path.join(step_dir_bpe, fname)
            # /path/to/results/test[.N].<model>.<settings>
            output_fname = os.path.join(step_dir_results, "{}.{}.{}.{}".format(args.model,
                                                                               args.decode_settings,
                                                                               fname[:-len(SUFFIX_SRC_GZ) - 1],
                                                                               SUFFIX_BPE))
            output_fnames_bpe.append(output_fname)
            # For the shared results directory, a command file indicates that
            # the step has completed successfully.
            command_fname = output_fname + "." + SUFFIX_COMMAND
            if os.path.exists(command_fname):
                logging.info("Re-use output: %s", output_fname)
            else:
                log_fname = os.path.join(args.workspace,
                                 DIR_LOGS,
                                 "sockeye.translate.{}.{}.{}.{}.log".format(task_name,
                                                                            args.model,
                                                                            fname[:-len(SUFFIX_SRC_GZ) - 1],
                                                                            os.getpid()))
                call_sockeye_translate(args=DECODE_ARGS[args.decode_settings],
                                       input_fname=input_fname,
                                       output_fname=output_fname,
                                       model_dir=step_dir_model,
                                       log_fname=log_fname,
                                       use_cpu=(args.gpus == 0))

    # (10) Evaluate test sets (bpe/tok/detok)

    lang_code = lang_codes[1] if lang_codes else None
    logging.info("=== Score outputs ===")
    # For each output file
    for fname_bpe in output_fnames_bpe:
        # Score byte-pair encoded
        fname_base = os.path.basename(fname_bpe)[:-len(SUFFIX_BPE)].split(".", 2)[2]
        fname_ref_bpe = os.path.join(step_dir_bpe, fname_base + SUFFIX_TRG_GZ)
        fname_bleu_bpe = fname_bpe + "." + SUFFIX_BLEU
        if os.path.exists(fname_bleu_bpe):
            logging.info("Re-use output: %s", fname_bleu_bpe)
        else:
            fname_log = os.path.join(args.workspace,
                         DIR_LOGS,
                         "contrib.sacrebleu.sacrebleu.{}.{}.{}.{}.log".format(task_name,
                                                                              args.model,
                                                                              fname_base + SUFFIX_BPE,
                                                                              os.getpid()))
            call_sacrebleu(input_fname=fname_bpe,
                           ref_fname=fname_ref_bpe,
                           output_fname=fname_bleu_bpe,
                           log_fname=fname_log,
                           tokenized=True)
        # Score tokenized
        fname_tok = fname_bpe[:-len(SUFFIX_BPE)] + SUFFIX_TOK
        fname_ref_tok = os.path.join(step_dir_tok, fname_base + SUFFIX_TRG_GZ)
        fname_bleu_tok = fname_tok + "." + SUFFIX_BLEU
        if os.path.exists(fname_bleu_tok):
            logging.info("Re-use output: %s", fname_bleu_tok)
        else:
            # Merge BPE
            logging.info("Merge BPE: %s -> %s", fname_bpe, fname_tok)
            third_party.merge_bpe(input_fname=fname_bpe, output_fname=fname_tok)
            fname_log = os.path.join(args.workspace,
                         DIR_LOGS,
                         "contrib.sacrebleu.sacrebleu.{}.{}.{}.{}.log".format(task_name,
                                                                              args.model,
                                                                              fname_base + SUFFIX_TOK,
                                                                              os.getpid()))
            call_sacrebleu(input_fname=fname_tok,
                           ref_fname=fname_ref_tok,
                           output_fname=fname_bleu_tok,
                           log_fname=fname_log,
                           tokenized=True)
        # Score detokenized (WMT-compatible BLEU)
        fname_detok = fname_bpe[:-len(SUFFIX_BPE)] + SUFFIX_DETOK
        fname_ref_raw = os.path.join(step_dir_raw, fname_base + SUFFIX_TRG_GZ)
        fname_bleu_detok = fname_detok + "." + SUFFIX_SACREBLEU
        if os.path.exists(fname_bleu_detok):
            logging.info("Re-use output: %s", fname_bleu_detok)
        else:
            if not requires_tokenization:
                logging.info("WARNING: Task uses pre-tokenized data, cannot reliably detokenize to compute WMT-compatible scores")
                continue
            # Detokenize
            logging.info("Detokenize (%s): %s -> %s", lang_code, fname_tok, fname_detok)
            third_party.call_moses_detokenizer(workspace_dir=args.workspace,
                                               input_fname=fname_tok,
                                               output_fname=fname_detok,
                                               lang_code=lang_code)
            fname_log = os.path.join(args.workspace,
                         DIR_LOGS,
                         "contrib.sacrebleu.sacrebleu.{}.{}.{}.{}.log".format(task_name,
                                                                              args.model,
                                                                              fname_base + SUFFIX_DETOK,
                                                                              os.getpid()))
            call_sacrebleu(input_fname=fname_detok,
                           ref_fname=fname_ref_raw,
                           output_fname=fname_bleu_detok,
                           log_fname=fname_log,
                           tokenized=False)


def main():
    default_workspace = os.path.join(os.path.expanduser("~"), DIR_SOCKEYE_AUTOPILOT)

    arg_parser = argparse.ArgumentParser(description="Sockeye Autopilot: end-to-end model training and evaluation.")
    arg_parser.add_argument("--workspace", type=str, metavar="DIR", default=default_workspace,
                            help="Base directory to use for building systems (download files, train models, etc.). Default: %(default)s.")
    arg_parser.add_argument("--task", type=str, choices=sorted(TASKS.keys()),
                            help="Pre-defined data set for model training.")
    arg_parser.add_argument("--model", type=str, choices=sorted(MODELS.keys()),
                            help="Type of translation model to train.")
    arg_parser.add_argument("--decode-settings", type=str, choices=sorted(DECODE_ARGS.keys()), default=DECODE_STANDARD,
                            help="Decoding settings. Default: %(default)s.")
    arg_parser.add_argument("--custom-task", type=str, metavar="NAME",
                            help="Name of custom task (used for directory naming).")
    arg_parser.add_argument("--custom-train", type=str, nargs=2, metavar=("SRC", "TRG"),
                            help="Custom training data (source and target).")
    arg_parser.add_argument("--custom-dev", type=str, nargs=2, metavar=("SRC", "TRG"),
                            help="Custom development data (source and target).")
    arg_parser.add_argument("--custom-test", type=str, nargs="+", metavar="SRC TRG",
                            help="Custom test data (pairs of source and target).")
    arg_parser.add_argument("--custom-text-type", type=str, choices=CUSTOM_TEXT_TYPES, default=CUSTOM_UTF8_RAW,
                            help="Level of pre-processing already applied to data for custom task: none (raw), tokenization, or byte-pair encoding. Default: %(default)s.")
    arg_parser.add_argument("--custom-lang", type=str, nargs=2, metavar=("SRC", "TRG"),
                            help="Source and target language codes for custom task (en, fr, de, etc.).")
    arg_parser.add_argument("--custom-bpe-op", type=int, default=32000,
                            help="Number of byte-pair encoding operations for custom task. Default: %(default)s.")
    arg_parser.add_argument("--gpus", type=int, metavar="N", default=1,
                            help="Number of GPUs to use. 0 for CPU only. Default: %(default)s.")
    arg_parser.add_argument("--test", action="store_true", default=False,
                            help="Run in test mode (much abbreviated system build).")

    args = arg_parser.parse_args()

    # Listed task or fully specified custom task
    utils.check_condition(args.task or all((args.custom_train, args.custom_dev, args.custom_test)),
            "Please specify --task or all of: --custom-task --custom-train --custom-dev --custom-test")

    # Required args for different custom tasks
    if not args.task:
        if args.custom_text_type == CUSTOM_UTF8_RAW:
            utils.check_condition(args.custom_lang, "Please specify --custom-lang for source and target tokenization")

    # Require explicit request to not train model
    if not args.model:
        raise RuntimeError("Please specify --model.  Use --model %s to run data preparation steps only" % MODEL_NONE)

    run_steps(args)


if __name__ == "__main__":
    main()

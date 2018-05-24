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

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List

# Make sure the version of sockeye being tested is first on the system path
try:
    import contrib.autopilot.autopilot as autopilot
except ImportError:
    SOCKEYE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PYTHONPATH = "PYTHONPATH"
    if os.environ.get(PYTHONPATH, None):
        os.environ[PYTHONPATH] += os.pathsep + SOCKEYE_ROOT
    else:
        os.environ[PYTHONPATH] = SOCKEYE_ROOT
    sys.path.append(SOCKEYE_ROOT)
    import contrib.autopilot.autopilot as autopilot


# Test-specific constants
WNMT_TASK = "wnmt18_en_de"
DATA_ONLY_TASK = "wmt14_fr_en"
WMT_TASK = "wmt14_de_en"
WMT_SRC = "de"
WMT_TRG = "en"
WMT_BPE = 32000
PREFIX_ZERO = "0."


def run_test(command: List[str], workspace: str):
    """
    Run a test command in a given workspace directory.  If it succeeds, clean up
    model files.  If it fails, print the last log file.
    """
    success = False
    try:
        subprocess.check_call(command + ["--workspace={}".format(workspace)])
        success = True
    except subprocess.CalledProcessError:
        pass
    if not success:
        print("Error running command. Final log file:", file=sys.stderr)
        print("==========", file=sys.stderr)
        log_dir = os.path.join(workspace, autopilot.DIR_LOGS)
        last_log = sorted(os.listdir(log_dir), key=lambda fname: os.stat(os.path.join(log_dir, fname)).st_mtime)[-1]
        with open(os.path.join(log_dir, last_log), "r") as log:
            for line in log:
                print(line, file=sys.stderr, end="")
        print("==========", file=sys.stderr)
        raise RuntimeError("Test failed: %s" % " ".join(command))
    # Cleanup models, leaving data avaiable for use as custom inputs to other
    # tasks
    model_dirs = glob.glob(os.path.join(workspace, autopilot.DIR_SYSTEMS, "*", "model.*"))
    for model_dir in model_dirs:
        shutil.rmtree(model_dir)


def main():
    """
    Build test systems with different types of pre-defined data and custom data
    with all levels of pre-processing.
    """
    with tempfile.TemporaryDirectory(prefix="sockeye.autopilot.") as tmp_dir:
        work_dir = os.path.join(tmp_dir, "workspace")

        # WNMT task with pre-tokenized data
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--task={}".format(WNMT_TASK),
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        run_test(command, workspace=work_dir)

        # WMT task, prepare data only
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--task={}".format(DATA_ONLY_TASK),
                   "--model=none",
                   "--gpus=0",
                   "--test"]
        run_test(command, workspace=work_dir)

        # WMT task with raw data
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--task={}".format(WMT_TASK),
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        run_test(command, workspace=work_dir)

        # Custom task (raw data)
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--custom-task=custom_raw",
                   "--custom-train",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_RAW, autopilot.PREFIX_TRAIN + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_RAW, autopilot.PREFIX_TRAIN + autopilot.SUFFIX_TRG_GZ),
                   "--custom-dev",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_RAW, autopilot.PREFIX_DEV + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_RAW, autopilot.PREFIX_DEV + autopilot.SUFFIX_TRG_GZ),
                   "--custom-test",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_RAW, autopilot.PREFIX_TEST + PREFIX_ZERO + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_RAW, autopilot.PREFIX_TEST + PREFIX_ZERO + autopilot.SUFFIX_TRG_GZ),
                   "--custom-lang",
                   WMT_SRC,
                   WMT_TRG,
                   "--custom-bpe-op={}".format(WMT_BPE),
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        run_test(command, workspace=work_dir)

        # Custom task (tokenized data)
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--custom-task=custom_tok",
                   "--custom-train",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_TOK, autopilot.PREFIX_TRAIN + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_TOK, autopilot.PREFIX_TRAIN + autopilot.SUFFIX_TRG_GZ),
                   "--custom-dev",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_TOK, autopilot.PREFIX_DEV + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_TOK, autopilot.PREFIX_DEV + autopilot.SUFFIX_TRG_GZ),
                   "--custom-test",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_TOK, autopilot.PREFIX_TEST + PREFIX_ZERO + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_TOK, autopilot.PREFIX_TEST + PREFIX_ZERO + autopilot.SUFFIX_TRG_GZ),
                   "--custom-text-type=tok",
                   "--custom-bpe-op={}".format(WMT_BPE),
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        run_test(command, workspace=work_dir)

        # Custom task (byte-pair encoded data)
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--custom-task=custom_bpe",
                   "--custom-train",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_BPE, autopilot.PREFIX_TRAIN + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_BPE, autopilot.PREFIX_TRAIN + autopilot.SUFFIX_TRG_GZ),
                   "--custom-dev",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_BPE, autopilot.PREFIX_DEV + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_BPE, autopilot.PREFIX_DEV + autopilot.SUFFIX_TRG_GZ),
                   "--custom-test",
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_BPE, autopilot.PREFIX_TEST + PREFIX_ZERO + autopilot.SUFFIX_SRC_GZ),
                   os.path.join(work_dir, autopilot.DIR_SYSTEMS, WMT_TASK + autopilot.SUFFIX_TEST, autopilot.DIR_DATA,
                                autopilot.DIR_BPE, autopilot.PREFIX_TEST + PREFIX_ZERO + autopilot.SUFFIX_TRG_GZ),
                   "--custom-text-type=bpe",
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        run_test(command, workspace=work_dir)

if __name__ == "__main__":
    main()

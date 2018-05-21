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

import os
import subprocess
import sys
import tempfile

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
                   "--workspace={}".format(work_dir),
                   "--task={}".format(WNMT_TASK),
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        subprocess.check_call(command)

        # WMT task, prepare data only
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--workspace={}".format(work_dir),
                   "--task={}".format(DATA_ONLY_TASK),
                   "--model=none",
                   "--gpus=0",
                   "--test"]
        subprocess.check_call(command)

        # WMT task with raw data
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--workspace={}".format(work_dir),
                   "--task={}".format(WMT_TASK),
                   "--model=transformer",
                   "--gpus=0",
                   "--test"]
        subprocess.check_call(command)

        # Custom task (raw data)
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--workspace={}".format(work_dir),
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
        subprocess.check_call(command)

        # Custom task (tokenized data)
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--workspace={}".format(work_dir),
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
        subprocess.check_call(command)

        # Custom task (byte-pair encoded data)
        command = [sys.executable,
                   "-m",
                   "contrib.autopilot.autopilot",
                   "--workspace={}".format(work_dir),
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
        subprocess.check_call(command)

if __name__ == "__main__":
    main()

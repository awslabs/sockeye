# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
'''Handle special settings that must be applied before mxnet is imported'''

import logging
import os
import sys


OMP_NUM_THREADS = 'OMP_NUM_THREADS'
OMP_NUM_THREADS_ARG = '--omp-num-threads'


logger = logging.getLogger(__name__)
initialized = False


def handle_omp_num_threads():
    for i, arg in enumerate(sys.argv):
        if arg.startswith(OMP_NUM_THREADS_ARG):
            if '=' in arg:
                val = arg.split('=')[1]
            else:
                val = sys.argv[i + 1]
            logger.warning('Setting %s=%s', OMP_NUM_THREADS, val)
            os.environ[OMP_NUM_THREADS] = val


def init():
    '''Call before importing mxnet module'''
    global initialized
    if not initialized:
        handle_omp_num_threads()
        initialized = True

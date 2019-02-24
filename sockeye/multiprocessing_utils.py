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

"""
Our checkpoint decoder runs in a separate python process. When launching this process (and also the sempaphore tracker
process that gets launched by Python's own multiprocessing) one needs to be careful that MXNet, MKL or CUDA resources
are not leaked from the parent to the child processes, as otherwise deadlocks can occur.
We achieve this by using the forkserver spawn method. Specifically, we create the forkserver before MXNet gets imported,
when the Python interpreter process is still in a "clean" state. All subsequent checkpoint decoder processes are then
forked from this clean process. Additionally, we trigger the creation of the sempahore tracker process before MXNet
is imported. In order to achieve this `initialize` must be called right after startup.
"""


import multiprocessing as mp
import logging
import os
import sys

logger = logging.getLogger(__name__)


def __dummy_function_to_start_semaphore_tracker():
    logger.info('Semphore tracker and forkserver started.')


__context = None


def initialize():
    global __context

    if __context is not None:
        # Already initialized
        return

    if not __context:
        if os.name == 'nt':
            # Windows does not support the forkserver spawn method, we use the default instead
            __context = mp.get_context()
        else:
            try:
                __context = mp.get_context('forkserver')

                # In order to ensure the forkserver is in a clean state we need to make sure initialize was called
                # before mxnet was imported from anywhere.
                all_imported_modules = sys.modules.keys()

                assert 'mxnet' not in all_imported_modules, ("sockeye.multiprocessing_utils.initialize must be called "
                                                             "before mxnet is imported.")

                p = mp.Process(target=__dummy_function_to_start_semaphore_tracker)
                p.start()
                p.join()
            except ValueError:
                logger.warning("Forkserver spawn method not available. Default spawn method will be used.")
                __context = mp.get_context()


def get_context():
    assert __context is not None, ("Multiprocessing context not initialized. Please call "
                                   "sockeye.multiprocessing_utils.initialize() right after interpreter startup.")
    return __context

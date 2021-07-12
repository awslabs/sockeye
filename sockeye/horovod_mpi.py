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
"""Optional Horovod and MPI support"""

# Import MPI-related packages once and in order.  Horovod should be initialized
# once and mpi4py should not auto-initialize.

# Import Horovod but do not call `init()` yet.  Initialization should be called
# as part of the main program after all modules (including Sockeye modules) have
# been imported.
try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

# Import mpi4py.MPI but do not automatically initialize/finalize the MPI
# environment.  Horovod already initializes the environment and running multiple
# initializations causes errors.  Finalization causes errors with other
# processes.
try:
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
except ImportError:
    mpi4py = None
    MPI = None


def using_horovod():
    """
    Returns true if the MPI environment is initialized, indicating that
    `hvd.init()` has been called.
    """
    if MPI is not None:
        return MPI.Is_initialized()
    return False

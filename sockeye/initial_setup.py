# Copyright 2019--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
'''
Module for setting up the initial environment before importing anything else.
'''

import logging
import os
import sys


ENV_ARG = '--env'


logger = logging.getLogger(__name__)


def handle_env_cli_arg():
    '''
    Call this before importing/initializing any modules that only read
    environment variables once at import/init time.
    '''
    for i, arg in enumerate(sys.argv):
        if arg.startswith(ENV_ARG):
            if arg.startswith(ENV_ARG + '='):
                argval = arg.split("=", 1)[1]
            else:
                argval = sys.argv[i + 1]
            for var_val in argval.split(','):
                var, val = var_val.split('=', 1)
                logger.warning('Setting %s=%s', var, val)
                os.environ[var] = val

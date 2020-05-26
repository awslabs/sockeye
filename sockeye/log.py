# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import logging.config
import sys
from typing import Optional

FORMATTERS = {
    'verbose': {
        'format': '[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s] %(message)s',
        'datefmt': "%Y-%m-%d:%H:%M:%S",
    },
    'simple': {
        'format': '[%(levelname)s:%(name)s] %(message)s'
    },
}

FILE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'rotating': {
            'level': 'INFO',
            'formatter': 'verbose',
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 10000000,
            'backupCount': 5,
            'filename': 'sockeye.log',
        }
    },
    'root': {
        'handlers': ['rotating'],
        'level': 'DEBUG',
    }
}

CONSOLE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'simple',
            'class': 'logging.StreamHandler',
            'stream': None
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    }
}

FILE_CONSOLE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'simple',
            'class': 'logging.StreamHandler',
            'stream': None
        },
        'rotating': {
            'level': 'INFO',
            'formatter': 'verbose',
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 10000000,
            'backupCount': 5,
            'filename': 'sockeye.log',
        }
    },
    'root': {
        'handlers': ['console', 'rotating'],
        'level': 'DEBUG',
    }
}

NO_LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
}

LOGGING_CONFIGS = {
    "file_only": FILE_LOGGING,
    "console_only": CONSOLE_LOGGING,
    "file_console": FILE_CONSOLE_LOGGING,
    "none": NO_LOGGING,
}


def setup_main_logger(file_logging=True, console=True, path: Optional[str] = None, level=logging.INFO):
    """
    Configures logging for the main application.

    :param file_logging: Whether to log to a file.
    :param console: Whether to log to the console.
    :param path: Optional path to write logfile to.
    :param level: Log level. Default: INFO.
    """
    if file_logging and console:
        log_config = LOGGING_CONFIGS["file_console"]  # type: ignore
    elif file_logging:
        log_config = LOGGING_CONFIGS["file_only"]
    elif console:
        log_config = LOGGING_CONFIGS["console_only"]
    else:
        log_config = LOGGING_CONFIGS["none"]

    if file_logging:
        assert path is not None, "Must provide a logfile path"
        log_config["handlers"]["rotating"]["filename"] = path  # type: ignore

    for _, handler_config in log_config['handlers'].items():  # type: ignore
        handler_config['level'] = level

    logging.config.dictConfig(log_config)  # type: ignore

    def exception_hook(exc_type, exc_value, exc_traceback):
        logging.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_hook


def log_sockeye_version(logger):
    from sockeye import __version__, __file__
    try:
        from sockeye.git_version import git_hash
    except ImportError:
        git_hash = "unknown"
    logger.info("Sockeye version %s, commit %s, path %s", __version__, git_hash, __file__)


def log_mxnet_version(logger):
    from mxnet import __version__, __file__
    logger.info("MXNet version %s, path %s", __version__, __file__)

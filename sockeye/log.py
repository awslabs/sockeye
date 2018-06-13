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

LOGGING_CONFIGS = {
    "file_only": FILE_LOGGING,
    "console_only": CONSOLE_LOGGING,
    "file_console": FILE_CONSOLE_LOGGING,
}


def is_python34() -> bool:
    version = sys.version_info
    return version[0] == 3 and version[1] == 4


def setup_main_logger(name: str, file_logging=True, console=True, path: Optional[str] = None) -> logging.Logger:
    """
    Return a logger that configures logging for the main application.

    :param name: Name of the returned logger.
    :param file_logging: Whether to log to a file.
    :param console: Whether to log to the console.
    :param path: Optional path to write logfile to.
    """
    if file_logging and console:
        log_config = LOGGING_CONFIGS["file_console"]
    elif file_logging:
        log_config = LOGGING_CONFIGS["file_only"]
    else:
        log_config = LOGGING_CONFIGS["console_only"]

    if path:
        log_config["handlers"]["rotating"]["filename"] = path  # type: ignore

    logging.config.dictConfig(log_config)
    logger = logging.getLogger(name)

    def exception_hook(exc_type, exc_value, exc_traceback):
        if is_python34():
            # Python3.4 does not seem to handle logger.exception() well
            import traceback
            traceback = "".join(traceback.format_tb(exc_traceback)) + exc_type.name
            logger.error("Uncaught exception\n%s", traceback)
        else:
            logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_hook

    return logger


def log_sockeye_version(logger):
    from sockeye import __version__
    try:
        from sockeye.git_version import git_hash
    except ImportError:
        git_hash = "unknown"
    logger.info("Sockeye version %s commit %s", __version__, git_hash)


def log_mxnet_version(logger):
    from mxnet import __version__
    logger.info("MXNet version %s", __version__)

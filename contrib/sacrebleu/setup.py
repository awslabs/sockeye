#!/usr/bin/env python3

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

"""
A setuptools based setup module.

See:
- https://packaging.python.org/en/latest/distributing.html
- https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup
import re
import os

def get_version():
    VERSION_RE = re.compile(r'''VERSION\s+=\s+['"]([0-9.]+)['"]''')
    with open(os.path.join(os.path.dirname(__file__), 'sacrebleu.py'), encoding='utf-8') as fin:
        return VERSION_RE.search(fin.read()).group(1)

setup(
    name = 'sacrebleu',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version = get_version(),

    description = 'Hassle-free computation of shareable, comparable, and reproducible BLEU scores',
    long_description = 'SacreBLEU is a standard BLEU implementation that downloads and manages WMT datasets, produces scores on detokenized outputs, and reports a string encapsulating BLEU parameters, facilitating the production of shareable, comparable BLEU scores.',

    # The project's main homepage.
    url = 'https://github.com/awslabs/sockeye',

    author = 'Amazon',
    author_email='sockeye-dev@amazon.com',
    maintainer_email='sockeye-dev@amazon.com',

    license = 'Apache License 2.0',

    python_requires = '>=3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3 :: Only',
    ],

    # What does your project relate to?
    keywords = ['machine translation, evaluation, NLP, natural language processing, computational linguistics'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment this:
    py_modules = ["sacrebleu"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires = ['typing'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require = {},

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'sacrebleu = sacrebleu:main',
        ],
    },
)

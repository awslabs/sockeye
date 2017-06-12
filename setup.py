import os
import re
import logging
from setuptools import setup, find_packages

ROOT = os.path.dirname(__file__)


def get_long_description():
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
        markdown_txt = f.read()
    try:
        import pypandoc
        long_description = pypandoc.convert(markdown_txt, 'rst', format='md')
        print(long_description)
    except(IOError, ImportError):
        logging.warning("Could not import package 'pypandoc'. Will not convert markdown readme to rst for PyPI.")
        long_description = markdown_txt
    return long_description


def get_version():
    VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')
    init = open(os.path.join(ROOT, 'sockeye', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)


def get_requirements(filename):
    with open(os.path.join(ROOT, filename)) as f:
        return [line.rstrip() for line in f]

try:
    from sphinx.setup_command import BuildDoc
    cmdclass = {'build_sphinx': BuildDoc}
except:
    logging.warning("Package 'sphinx' not found. You will not be able to build docs.")
    cmdclass = {}

args = dict(
    name='sockeye',

    version=get_version(),

    description='Sequence-to-Sequence framework for Neural Machine Translation',
    long_description=get_long_description(),

    url='https://github.com/awslabs/sockeye',

    author='Amazon',
    author_email='sockeye-dev@amazon.com',
    maintainer_email='sockeye-dev@amazon.com',

    license='Apache License 2.0',
    
    python_requires='>=3',

    packages=find_packages(exclude=("test",)),

    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],

    extras_require={
        'optional': ['tensorboard'],
        'dev': get_requirements('requirements.dev.txt')
    },

    install_requires=get_requirements('requirements.txt'),

    entry_points={
        'console_scripts': [
            'sockeye-train = sockeye.train:main',
            'sockeye-translate = sockeye.translate:main',
            'sockeye-average = sockeye.average:main',
            'sockeye-embeddings = sockeye.embeddings:main'
        ],
    },

    cmdclass=cmdclass,

)

setup(**args)

# Developer Documentation

## Requirements

There are three types of dependencies: core dependencies, development dependencies and dependencies for generating the documentation.

Install them via

```bash
> pip install -r requirements/requirements.txt
> pip install -r requirements/requirements.dev.txt
> pip install -r requirements/requirements.docs.txt
```

## Developer Guidelines

We welcome contributions to sockeye in form of pull requests on Github.
If you want to develop sockeye, please adhere to the following development guidelines.

- Write Python 3.7, PEP8 compatible code.

- Functions should be documented with Sphinx-style docstrings and
   should include type hints for static code analyzers.

```python
def foo(bar: <type of bar>) -> <returnType>:
    """
    <Docstring for foo method, followed by a period>.

    :param bar: <Description of bar argument followed by a period>.
    :return: <Description of the return value followed by a period>.
    """
```

- The desired line length of Python modules should not exceed 120 characters.

- Make sure to pass unit tests before submitting a pull request.

- Whenever reasonable, write py.test unit tests covering your contribution.

- When importing other sockeye modules import the entire module instead of individual functions and classes using relative imports:

```python
from . import attention
```

## Unit & Integration Tests

Unit & integration tests are written using py.test.
They can be run with:

```bash
> python setup.py test
```

or:

```bash
> pytest
```

Integration tests run Sockeye CLI tools on small, synthetic data to test for functional correctness.

## System Tests

System tests test Sockeye CLI tools on synthetic tasks (digit sequence copying & sorting) for functional correctness and successful learning. They assert on validation metrics (perplexity) and BLEU scores from decoding.
A subset of the system tests are run as part of Github workflows for every commit/pull request.
You can manually run the system tests with:

```bash
> pytest test/system
```

## Submitting a New Version to PyPI

Before starting make sure you have the [TestPyPI](https://wiki.python.org/moin/TestPyPI) and PyPI accounts and the
corresponding `~/.pypirc` set up.

1. Build source distribution:
   ``` bash
   > python setup.py sdist bdist_wheel
   ```
1. Upload to PyPITest:
   ```bash
   > twine upload dist/sockeye-${VERSION}.tar.gz dist/sockeye-${VERSION}-py3-none-any.whl -r pypitest
   ```
1. In a new python environment check that the package is installable
   ```bash
   > pip install -i https://testpypi.python.org/pypi sockeye
   ```
1. Upload to PyPI
   ```bash
   > twine upload dist/sockeye-${VERSION}.tar.gz dist/sockeye-${VERSION}-py3-none-any.whl
   ```
When pushing a new git tag to the repository, it is automatically built and deployed to PyPI as a new version via Travis.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Licensing

See the [LICENSE](https://github.com/awslabs/sockeye/blob/main/LICENSE) file for our project's licensing. We will ask you confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.

# Developer Documentation

## Requirements



There are three types of dependencies: core dependencies, development dependencies and dependencies for generating the documentation.
 
Install them via
```bash
> pip install -r requirements.txt
> pip install -r requirements.dev.txt
> pip install -r requirements.docs.txt
```


## Developer Guidelines

We welcome contributions to sockeye in form of pull requests on Github.
If you want to develop sockeye, please adhere to the following development guidelines.


 * Write Python 3.5, PEP8 compatible code.
 
 * Functions should be documented with Sphinx-style docstrings and
   should include type hints for static code analyzers.
 
    ```python
    def foo(bar: <type of bar>) -> <returnType>:
        """
        <Docstring for foo method, followed by a period>.
        
        :param bar: <Description of bar argument followed by a period>.
        :return: <Description of the return value followed by a period>.
        """
    ```

 * When using MXNet operators, preceding symbolic statements
   in the code with the resulting, expected shape of the tensor greatly improves readability of the code:
    ```python
    # (batch_size, num_hidden)
    data = mx.sym.Variable('data')
    # (batch_size * num_hidden,)
    data = mx.sym.reshape(data=data, shape=(-1))
    ```

 * The desired line length of Python modules should not exceed 120 characters.
 
 * When writing symbol-generating classes (such as encoders/decoders), initialize variables in the constructor of the 
   class and re-use them in the class methods.
   
 * Make sure to pass unit tests before submitting a pull request.
 
 * Whenever reasonable, write py.test unit tests covering your contribution.
 
 * When importing other sockeye modules import the entire module instead of individual functions and classes using
   relative imports:
    ```python
    from . import attention
    ```  
   

## Building the Documentation
Full documentation, including a code reference, can be generated using Sphinx with the following command:
```bash
> python setup.py docs
```
The results are written to ```docs/_build/html/index.html```.


## Unit tests
Unit tests are written using py.test.
They can be run like this:
```bash
> python setup.py test
```

## Submitting a new version to PyPI

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

 

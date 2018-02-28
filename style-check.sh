#!/bin/sh

# run this to perform code-style checking.

# Run pylint on the sockeye package, failing on any reported errors.
pylint --rcfile=pylintrc sockeye -E
SOCKEYE_LINT_RESULT=$?

# Run pylint on test package, failing on any reported errors.
pylint --rcfile=pylintrc test -E
TESTS_LINT_RESULT=$?

# Run mypy, we are currently limiting to modules in typechecked-files
mypy --ignore-missing-imports --follow-imports=silent @typechecked-files
MYPY_RESULT=$?

[ $SOCKEYE_LINT_RESULT -ne 0 ] && echo 'pylint found errors in the sockeye package' && exit 1
[ $TESTS_LINT_RESULT -ne 0 ] && echo 'pylint found errors in the test package' && exit 1
[ $MYPY_RESULT -ne 0 ] && echo 'mypy found incorrect type usage' && exit 1

echo 'all style checks passed'
exit 0

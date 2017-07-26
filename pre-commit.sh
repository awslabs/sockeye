#!/bin/sh
#
# A pre-commit script that will run before every commit to Sockeye.
# This script contains the same tests that will evenutally run in CI.
# Install by running ln -s ../../pre-commit.sh .git/hooks/pre-commit
# You can remove these checks at any time by running rm .git/hooks/pre-commit
# You can commit bypassing these changes by running git commit --no-verify

# Stash all non-commited files
STASH_NAME="pre-commit-$(date +%s)"
git stash save -q --keep-index $STASH_NAME

# Run unit and integration tests
python3 setup.py test
TEST_RESULT=$?

# Run pylint on the sockeye package, failing on any reported errors.
pylint --rcfile=pylintrc sockeye -E
SOCKEYE_LINT_RESULT=$?

# Run pylint on test package, failing on any reported errors.
pylint --rcfile=pylintrc test -E
TESTS_LINT_RESULT=$?

# Run mypy, we are currently limiting to modules that pass
# Please feel free to fix mypy issues in other modules and add them to typechecked-files
mypy --ignore-missing-imports --follow-imports=silent @typechecked-files
MYPY_RESULT=$?

# Run system tests
python3 -m pytest test/system
SYSTEM_RESULT=$?

# Pop our stashed files
STASHES=$(git stash list)
if [[ $STASHES == "$STASH_NAME" ]]; then
  git stash pop -q
fi

[ $TEST_RESULT -ne 0 ] && echo 'Unit or integration tests failed' && exit 1
[ $SOCKEYE_LINT_RESULT -ne 0 ] && echo 'pylint found errors in the sockeye package' && exit 1
[ $TESTS_LINT_RESULT -ne 0 ] && echo 'pylint found errors in the test package' && exit 1
[ $MYPY_RESULT -ne 0 ] && echo 'mypy found incorrect type usage' && exit 1
[ $SYSTEM_RESULT -ne 0 ] && echo 'System tests failed' && exit 1

echo 'all pre-commit checks passed'
exit 0

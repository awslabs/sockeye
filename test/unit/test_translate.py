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
import io
import unittest
import unittest.mock

import pytest

import sockeye.inference
import sockeye.output_handler
import sockeye.translate

TEST_DATA = "Test file line 1\n" \
            "Test file line 2\n"


@pytest.fixture
def mock_translator():
    return unittest.mock.Mock(spec=sockeye.inference.Translator)


@pytest.fixture
def mock_output_handler():
    return unittest.mock.Mock(spec=sockeye.output_handler.OutputHandler)


def mock_open(*args, **kargs):
    f_open = unittest.mock.mock_open(*args, **kargs)
    f_open.return_value.__iter__ = lambda self: iter(self.readline, '')
    return f_open


@unittest.mock.patch("builtins.open", new_callable=mock_open, read_data=TEST_DATA)
def test_translate_by_file(mock_file, mock_translator, mock_output_handler):
    mock_translator.translate.return_value = ['', '']
    mock_translator.batch_size = 1
    mock_file.return_value = TEST_DATA.splitlines()
    sockeye.translate.read_and_translate(translator=mock_translator, output_handler=mock_output_handler,
                                         chunk_size=2, source='/dev/null')

    # Ensure that our translator has the correct input passed to it.
    mock_translator.make_input.assert_any_call(1, "Test file line 1")
    mock_translator.make_input.assert_any_call(2, "Test file line 2")

    # Ensure translate gets called once.  Input here will be a dummy mocked result, so we'll ignore it.
    assert mock_translator.translate.call_count == 1


@unittest.mock.patch("sys.stdin", io.StringIO(TEST_DATA))
def test_translate_by_stdin_chunk2(mock_translator, mock_output_handler):
    mock_translator.translate.return_value = ['', '']
    mock_translator.batch_size = 1
    sockeye.translate.read_and_translate(translator=mock_translator, output_handler=mock_output_handler,
                                         chunk_size=2)

    # Ensure that our translator has the correct input passed to it.
    mock_translator.make_input.assert_any_call(1, "Test file line 1\n")
    mock_translator.make_input.assert_any_call(2, "Test file line 2\n")

    # Ensure translate gets called once.  Input here will be a dummy mocked result, so we'll ignore it.
    assert mock_translator.translate.call_count == 1

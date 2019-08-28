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
import io
import os
import unittest
import unittest.mock
from tempfile import TemporaryDirectory

import sockeye.constants
import sockeye.inference
import sockeye.output_handler
import sockeye.translate

TEST_DATA = "Test file line 1\n" \
            "Test file line 2\n"


def mock_open(*args, **kargs):
    # work-around for [MagicMock objects not being iterable](http://bugs.python.org/issue21258)
    # cf. http://stackoverflow.com/questions/24779893/customizing-unittest-mock-mock-open-for-iteration
    f_open = unittest.mock.mock_open(*args, **kargs)
    f_open.return_value.__iter__ = lambda self: iter(self.readline, '')
    return f_open


def test_translate_by_file():
    mock_output_handler = unittest.mock.Mock(spec=sockeye.output_handler.OutputHandler)
    mock_translator = unittest.mock.Mock(spec=sockeye.inference.Translator)
    mock_translator.translate.return_value = ['', '']
    mock_translator.num_source_factors = 1
    mock_translator.max_batch_size = 1

    mock_translator.nbest_size = 1
    sockeye.translate.read_and_translate(translator=mock_translator, output_handler=mock_output_handler,
                                         chunk_size=2, input_file='/dev/null', input_factors=None)

    with TemporaryDirectory() as temp:
        input_filename = os.path.join(temp, 'input')
        with open(input_filename, 'w') as f:
            f.write(TEST_DATA)

        sockeye.translate.read_and_translate(translator=mock_translator, output_handler=mock_output_handler,
                                             chunk_size=2, input_file=input_filename, input_factors=None)

        # Ensure translate gets called once.  Input here will be a dummy mocked result, so we'll ignore it.
        assert mock_translator.translate.call_count == 1


@unittest.mock.patch("sys.stdin", io.StringIO(TEST_DATA))
def test_translate_by_stdin_chunk2():
    mock_output_handler = unittest.mock.Mock(spec=sockeye.output_handler.OutputHandler)
    mock_translator = unittest.mock.Mock(spec=sockeye.inference.Translator)
    mock_translator.translate.return_value = ['', '']
    mock_translator.num_source_factors = 1
    mock_translator.max_batch_size = 1
    mock_translator.nbest_size = 1
    sockeye.translate.read_and_translate(translator=mock_translator,
                                         output_handler=mock_output_handler,
                                         chunk_size=2)

    # Ensure translate gets called once.  Input here will be a dummy mocked result, so we'll ignore it.
    assert mock_translator.translate.call_count == 1

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

import pytest
import io
import numpy as np
from sockeye.inference import TranslatorInput, TranslatorOutput
import sockeye.output_handler

stream_handler_tests = [(sockeye.output_handler.StringOutputHandler(io.StringIO()),
                         TranslatorInput(id=0, sentence="a test", tokens=None),
                         TranslatorOutput(id=0, translation="ein Test", tokens=None,
                                          attention_matrix=None,
                                          score=0.),
                         "ein Test\n"),
                        (sockeye.output_handler.StringOutputHandler(io.StringIO()),
                         TranslatorInput(id=0, sentence="", tokens=None),
                         TranslatorOutput(id=0, translation="", tokens=None,
                                          attention_matrix=None,
                                          score=0.),
                         "\n"),
                        (sockeye.output_handler.StringWithAlignmentsOutputHandler(io.StringIO(), threshold=0.5),
                         TranslatorInput(id=0, sentence="a test", tokens=None),
                         TranslatorOutput(id=0, translation="ein Test", tokens=None,
                                          attention_matrix=np.asarray([[1, 0],
                                                                       [0, 1]]),
                                          score=0.),
                         "ein Test\t0-0 1-1\n"),
                        (sockeye.output_handler.StringWithAlignmentsOutputHandler(io.StringIO(), threshold=0.5),
                         TranslatorInput(id=0, sentence="a test", tokens=None),
                         TranslatorOutput(id=0, translation="ein Test !", tokens=None,
                                          attention_matrix=np.asarray([[0.4, 0.6],
                                                                       [0.8, 0.2],
                                                                       [0.5, 0.5]]),
                                          score=0.),
                         "ein Test !\t0-1 1-0\n"),
                        ]


@pytest.mark.parametrize("handler, translation_input, translation_output, expected_string", stream_handler_tests)
def test_stream_output_handler(handler, translation_input, translation_output, expected_string):
    handler.handle(translation_input, translation_output)
    assert handler.stream.getvalue() == expected_string

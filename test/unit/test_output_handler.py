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

import pytest

import sockeye.output_handler
from sockeye.inference_pt import TranslatorInput, TranslatorOutput

stream_handler_tests = [(sockeye.output_handler.StringOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=[], factors=[], constraints=[]),
                         TranslatorOutput(sentence_id=0, translation="ein Test", tokens=None,
                                          score=0.),
                         0.,
                         "ein Test\n"),
                        (sockeye.output_handler.StringOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=[], factors=[]),
                         TranslatorOutput(sentence_id=0, translation="", tokens=None,
                                          score=0.),
                         0.,
                         "\n"),
                        (sockeye.output_handler.BenchmarkOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=["a", "test"], factors=[]),
                         TranslatorOutput(sentence_id=0, translation="ein Test", tokens=["ein", "Test"],
                                          score=0.),
                         0.5,
                         "input=a test\toutput=ein Test\tinput_tokens=2\toutput_tokens=2\ttranslation_time=0.5000\n"),
                        (sockeye.output_handler.JSONOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=[], factors=[], constraints=[]),
                         TranslatorOutput(sentence_id=0, translation="ein Test", tokens=['ein', 'Test'],
                                          score=0.,
                                          pass_through_dict={'pass_through_test': 'success!'},
                                          nbest_translations=["ein Test", "der Test"],
                                          nbest_tokens=[["ein", "Test"], ["der", "Test"]],
                                          nbest_scores=[0., 0.1],
                                          factor_translations=['f11 f12', 'f21 f22'],
                                          factor_scores=[0.1, 0.2],
                                          nbest_factor_translations=[['f11 f12', 'f21 f22'],
                                                                     ['f11 f12', 'f21 f22']]),
                         0.5,
                         '{"factor1": "f11 f12", "factor1_score": 0.1, "factor2": "f21 f22", "factor2_score": 0.2, "pass_through_test": "success!", "score": 0.0, "scores": [0.0, 0.1], "sentence_id": 0, "translation": "ein Test", "translations": ["ein Test", "der Test"], "translations_factors": [{"factor1": "f11 f12", "factor2": "f21 f22"}, {"factor1": "f11 f12", "factor2": "f21 f22"}]}\n')]


@pytest.mark.parametrize("handler, translation_input, translation_output, translation_walltime, expected_string", stream_handler_tests)
def test_stream_output_handler(handler, translation_input, translation_output, translation_walltime, expected_string):
    handler.handle(translation_input, translation_output, translation_walltime)
    assert handler.stream.getvalue() == expected_string

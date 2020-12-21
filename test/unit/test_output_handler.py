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

import numpy as np
import pytest

import sockeye.output_handler
from sockeye.inference import TranslatorInput, TranslatorOutput

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
                        (sockeye.output_handler.BeamStoringHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=["What"]),
                         TranslatorOutput(sentence_id=0, translation="Was", tokens=["Was"],
                                          score=0.,
                                          beam_histories=[
                                              {"predicted_ids": [[258, 137, 31],
                                                                 [0, 0, 3]],
                                               "predicted_tokens": [["Was", "Wie", "Wo"],
                                                                    ["<pad>", "<pad>", "</s>"]],
                                               "parent_ids": [[0, 0, 0],
                                                              [0, 0, 1]],
                                               "scores": [[0.05599012225866318, 4.394228935241699, 4.426244735717773],
                                                          [2.2783169746398926, 3.5674173831939697, 3.648634195327759]],
                                               "normalized_scores": [[0.05599012225866318, 4.394228935241699, 4.426244735717773],
                                                                     [0.17525514960289001, 0.2744167149066925, 0.2806641757488251]]}
                                          ]),
                         0.5,
                         '{"id": 0, "normalized_scores": [[0.05599012225866318, 4.394228935241699, 4.426244735717773], [0.17525514960289001, 0.2744167149066925, 0.2806641757488251]], "number_steps": 2, "parent_ids": [[0, 0, 0], [0, 0, 1]], "predicted_ids": [[258, 137, 31], [0, 0, 3]], "predicted_tokens": [["Was", "Wie", "Wo"], ["<pad>", "<pad>", "</s>"]], "scores": [[0.05599012225866318, 4.394228935241699, 4.426244735717773], [2.2783169746398926, 3.5674173831939697, 3.648634195327759]]}\n'),
                        (sockeye.output_handler.JSONOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=[], factors=[], constraints=[]),
                         TranslatorOutput(sentence_id=0, translation="ein Test", tokens=None,
                                          score=0.,
                                          pass_through_dict={'pass_through_test': 'success!'},
                                          nbest_translations=["ein Test", "der Test"],
                                          nbest_tokens=[None, None],
                                          nbest_scores=[0., 0.1]),
                         0.5,
                         '{"pass_through_test": "success!", "score": 0.0, "scores": [0.0, 0.1], "sentence_id": 0, "translation": "ein Test", "translations": ["ein Test", "der Test"]}\n')]


@pytest.mark.parametrize("handler, translation_input, translation_output, translation_walltime, expected_string", stream_handler_tests)
def test_stream_output_handler(handler, translation_input, translation_output, translation_walltime, expected_string):
    handler.handle(translation_input, translation_output, translation_walltime)
    assert handler.stream.getvalue() == expected_string

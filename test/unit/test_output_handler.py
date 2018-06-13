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
import pytest
import numpy as np
from sockeye.inference import TranslatorInput, TranslatorOutput
import sockeye.output_handler

stream_handler_tests = [(sockeye.output_handler.StringOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=[], factors=[], constraints=[]),
                         TranslatorOutput(id=0, translation="ein Test", tokens=None,
                                          attention_matrix=None,
                                          score=0.),
                         0.,
                         "ein Test\n"),
                        (sockeye.output_handler.StringOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=[], factors=[]),
                         TranslatorOutput(id=0, translation="", tokens=None,
                                          attention_matrix=None,
                                          score=0.),
                         0.,
                         "\n"),
                        (sockeye.output_handler.StringWithAlignmentsOutputHandler(io.StringIO(), threshold=0.5),
                         TranslatorInput(sentence_id=0, tokens="a test".split(), factors=[]),
                         TranslatorOutput(id=0, translation="ein Test", tokens=None,
                                          attention_matrix=np.asarray([[1, 0],
                                                                       [0, 1]]),
                                          score=0.),
                         0.,
                         "ein Test\t0-0 1-1\n"),
                        (sockeye.output_handler.StringWithAlignmentsOutputHandler(io.StringIO(), threshold=0.5),
                         TranslatorInput(sentence_id=0, tokens="a test".split(), factors=[]),
                         TranslatorOutput(id=0, translation="ein Test !", tokens=None,
                                          attention_matrix=np.asarray([[0.4, 0.6],
                                                                       [0.8, 0.2],
                                                                       [0.5, 0.5]]),
                                          score=0.),
                         0.,
                         "ein Test !\t0-1 1-0\n"),
                        (sockeye.output_handler.BenchmarkOutputHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=["a", "test"], factors=[]),
                         TranslatorOutput(id=0, translation="ein Test", tokens=["ein", "Test"],
                                          attention_matrix=None,
                                          score=0.),
                         0.5,
                         "input=a test\toutput=ein Test\tinput_tokens=2\toutput_tokens=2\ttranslation_time=0.5000\n"),
                        (sockeye.output_handler.BeamStoringHandler(io.StringIO()),
                         TranslatorInput(sentence_id=0, tokens=["What"]),
                         TranslatorOutput(id=0, translation="Was", tokens=["Was"],
                                          attention_matrix=None, score=0.,
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
                        ]


@pytest.mark.parametrize("handler, translation_input, translation_output, translation_walltime, expected_string", stream_handler_tests)
def test_stream_output_handler(handler, translation_input, translation_output, translation_walltime, expected_string):
    handler.handle(translation_input, translation_output, translation_walltime)
    assert handler.stream.getvalue() == expected_string

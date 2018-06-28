# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sockeye import custom_seq_parser

descriptions_with_parses = [
    ("rnn", [{'name': 'rnn', 'params': None}]),
    ("rnn->rnn", [{'name': 'rnn', 'params': None}, {'name': 'rnn', 'params': None}]),
    # Transformer
    ("pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff(2048)->linear(512)))->norm",
     [
         {"name": 'pos', 'params': None},
         {"name": "repeat", "num": 6, 'layers': [
            {'name': 'res', 'layers': [{'name': 'norm', 'params': None}, {'name': 'mh_dot_self_att', 'params': None}]},
            {'name': 'res', 'layers': [{'name': 'norm', 'params': None}, {'name': 'mh_dot_att', 'params': None}]},
            {'name': 'res', 'layers': [{'name': 'norm', 'params': None},
                                       {'name': 'ff', 'params': [2048]},
                                       {'name': 'linear', 'params': [512]}]}]},
         {'name': 'norm', 'params': None}
     ]),
    # keyword args
    ("ff(1,2,key1=3,key2=4)", [{"name": "ff", "params": [1, 2, ('key1', 3), ('key2', 4)]}])
]
# TODO: add tests for parallel layers and other meta layers


@pytest.mark.parametrize("description, expected_parsed_layers", descriptions_with_parses)
def test_parser(description, expected_parsed_layers):
    parsed_layers = custom_seq_parser.parse(description)
    print(parsed_layers)

    assert parsed_layers == expected_parsed_layers

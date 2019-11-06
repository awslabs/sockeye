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

"""
Integration tests for lexical constraints.
"""
import json
import os
import sys
from typing import Dict, List, Any
from unittest.mock import patch

import sockeye.translate
from sockeye.test_utils import collect_translate_output_and_scores, TRANSLATE_PARAMS_COMMON

_TRAIN_LINE_COUNT = 20
_TRAIN_LINE_COUNT_EMPTY = 1
_DEV_LINE_COUNT = 5
_TEST_LINE_COUNT = 5
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20

TEST_CONFIGS = [
    # beam prune
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01",
     "--batch-size 3 --beam-size 9 --beam-prune 1"),
    # no beam prune
    ("--encoder transformer --decoder transformer"
     " --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8"
     " --transformer-feed-forward-num-hidden 16"
     " --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr"
     " --weight-tying-type src_trg_softmax"
     " --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg"
     " --batch-size 2 --max-updates 4 --batch-type sentence --decode-and-evaluate 0"
     " --checkpoint-interval 4 --optimizer adam --initial-learning-rate 0.01",
     "--batch-size 1 --beam-size 10")
]

# TODO(fhieber): Disabled due to brittleness of constrained decoding tests with Transformer models. Requires investigation.
# @pytest.mark.parametrize("train_params, translate_params", TEST_CONFIGS)
# def test_constraints(train_params: str, translate_params: str):
#     with tmp_digits_dataset(prefix="test_constraints",
#                             train_line_count=_TRAIN_LINE_COUNT,
#                             train_line_count_empty=_TRAIN_LINE_COUNT_EMPTY,
#                             train_max_length=_LINE_MAX_LENGTH,
#                             dev_line_count=_DEV_LINE_COUNT,
#                             dev_max_length=_LINE_MAX_LENGTH,
#                             test_line_count=_TEST_LINE_COUNT,
#                             test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
#                             test_max_length=_TEST_MAX_LENGTH,
#                             sort_target=False) as data:
#         # train a minimal default model
#         data = run_train_translate(train_params=train_params, translate_params=translate_params, data=data,
#                                    max_seq_len=_LINE_MAX_LENGTH + C.SPACE_FOR_XOS)
#
#         # 'constraint' = positive constraints (must appear), 'avoid' = negative constraints (must not appear)
#         for constraint_type in ["constraints", "avoid"]:
#             _test_constrained_type(constraint_type=constraint_type, data=data, translate_params=translate_params)


def _test_constrained_type(constraint_type: str, data: Dict[str, Any], translate_params: str):
    constrained_inputs = _create_constrained_inputs(constraint_type, data['test_inputs'], data['test_outputs'])
    new_test_source_path = os.path.join(data['work_dir'], "test_constrained.txt")
    with open(new_test_source_path, 'w') as out:
        for json_line in constrained_inputs:
            print(json_line, file=out)
    out_path_constrained = os.path.join(data['work_dir'], "out_constrained.txt")
    params = "{} {} {} --json-input --output-type translation_with_score".format(
        sockeye.translate.__file__,
        TRANSLATE_PARAMS_COMMON.format(model=data['model'],
                                       input=new_test_source_path,
                                       output=out_path_constrained),
        translate_params)
    with patch.object(sys, "argv", params.split()):
        sockeye.translate.main()
    constrained_outputs, constrained_scores = collect_translate_output_and_scores(out_path_constrained)
    assert len(constrained_outputs) == len(data['test_outputs']) == len(constrained_inputs)
    for json_source, constrained_out, unconstrained_out in zip(constrained_inputs,
                                                               constrained_outputs,
                                                               data['test_outputs']):
        jobj = json.loads(json_source)
        if jobj.get(constraint_type) is None:
            # if there were no constraints, make sure the output is the same as the unconstrained output
            assert constrained_out == unconstrained_out
        else:
            restriction = jobj[constraint_type][0]
            if constraint_type == 'constraints':
                # for positive constraints, ensure the constraint is in the constrained output
                assert restriction in constrained_out
            else:
                # for negative constraints, ensure the constraints is *not* in the constrained output
                assert restriction not in constrained_out


def _create_constrained_inputs(constraint_type: str,
                               translate_inputs: List[str],
                               translate_outputs: List[str]) -> List[str]:
    constrained_inputs = []  # type: List[str]
    for sentno, (source, translate_output) in enumerate(zip(translate_inputs, translate_outputs)):
        target_words = translate_output.split()
        target_len = len(target_words)
        new_source = {'text': source}
        # From the odd-numbered sentences that are not too long, create constraints. We do
        # only odds to ensure we get batches with mixed constraints / lack of constraints.
        if target_len > 0 and sentno % 2 == 0:
            start_pos = 0
            end_pos = min(target_len, 3)
            constraint = ' '.join(target_words[start_pos:end_pos])
            new_source[constraint_type] = [constraint]
        constrained_inputs.append(json.dumps(new_source))
    return constrained_inputs

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
import unittest
import unittest.mock

import pytest

from collections import defaultdict

import sockeye.scoring
import sockeye.output_handler
import sockeye.score
import sockeye.data_io


# TODO: read from actual files?
TEST_DATA_SRC = "Test file src line 1\n" \
                "Test file src line 2\n"

TEST_DATA_TRG = "Test file trg line 1\n" \
                "Test file trg line 2\n"


@pytest.fixture
def mock_scorer():
    return unittest.mock.Mock(spec=sockeye.scoring.Scorer)


@pytest.fixture
def mock_score_output_handler():
    return unittest.mock.Mock(spec=sockeye.output_handler.ScoreOutputHandler)


@pytest.fixture
def mock_scoring_model():
    return unittest.mock.Mock(spec=sockeye.scoring.ScoringModel)


@pytest.fixture
def mock_data_iterator():
    return unittest.mock.Mock(spec=sockeye.data_io.BaseParallelSampleIter)


def test_score(mock_scorer, mock_score_output_handler, mock_scoring_model, mock_data_iterator):
    mock_scorer.score.return_value = ['', '']
    mock_scorer.batch_size = 1
    mock_scorer.no_bucketing = False
    sockeye.score.score(output_handler=mock_score_output_handler,
                        models=[mock_scoring_model],
                        data_iters=[mock_data_iterator],
                        mapids=[defaultdict(lambda: defaultdict(int))],
                        scorer=mock_scorer)
    # Ensure score gets called once.
    assert mock_scorer.score.call_count == 1

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

import pytest

import sockeye.average as average


@pytest.mark.parametrize(
    "test_points, expected_top_n, size, maximize", [
        ([(1.1, 3), (2.2, 2), (3.3, 1)], [(3.3, 1), (2.2, 2), (1.1, 3)], 3, True),
        ([(1.1, 3), (2.2, 2), (3.3, 1)], [(1.1, 3), (2.2, 2), (3.3, 1)], 3, False),
        ([(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], [(4.4, 1), (3.3, 2), (2.2, 3)], 3, True),
        ([(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], [(4.4, 1), (3.3, 2), (2.2, 3), (1.1, 4)], 5, True)
])
def test_strategy_best(test_points, expected_top_n, size, maximize):
    result = average._strategy_best(test_points, size, maximize)
    assert result == expected_top_n


@pytest.mark.parametrize(
    "test_points, expected_top_n, size, maximize", [
        ([(1.1, 3), (2.2, 2), (3.3, 1)], [(1.1, 3), (2.2, 2), (3.3, 1)], 3, True),
        ([(1.1, 3), (2.2, 2), (3.3, 1)], [(1.1, 3)], 3, False),
        ([(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], [(2.2, 3), (3.3, 2), (4.4, 1)], 3, True),
        ([(2.2, 4), (1.1, 3), (3.3, 2), (4.4, 1)], [(2.2, 4), (1.1, 3)], 3, False),
        ([(2.2, 4), (1.1, 3), (3.3, 2), (4.4, 1)], [(1.1, 3)], 1, False),
        ([(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], [(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], 5, True)
])
def test_strategy_last(test_points, expected_top_n, size, maximize):
    result = average._strategy_last(test_points, size, maximize)
    assert result == expected_top_n


@pytest.mark.parametrize(
    "test_points, expected_top_n, size, maximize", [
        ([(1.1, 3), (2.2, 2), (3.3, 1)], [[0, 3.3, 1], [0, 2.2, 2], [0, 1.1, 3]], 3, True),
        ([(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], [[0, 4.4, 1], [0, 3.3, 2], [0, 2.2, 3]], 3, True),
        ([(3.3, 3), (2.2, 2), (1.1, 1)], [[2, 3.3, 3], [0, 2.2, 2], [0, 1.1, 1]], 3, True),
        ([(3.3, 3), (2.2, 2), (1.1, 1)], [[0, 1.1, 1], [0, 2.2, 2], [0, 3.3, 3]], 3, False),
        ([(2.2, 4), (1.1, 3), (3.3, 2), (4.4, 1)], [[1, 2.2, 4], [0, 4.4, 1], [0, 3.3, 2]], 3, True),
        ([(2.2, 4), (1.1, 3), (3.3, 2), (4.4, 1)], [[2, 1.1, 3]], 1, False),
        ([(1.1, 4), (2.2, 3), (3.3, 2), (4.4, 1)], [[3, 1.1, 4], [0, 2.2, 3], [0, 3.3, 2], [0, 4.4, 1]], 5, False)
])
def test_strategy_lifespan(test_points, expected_top_n, size, maximize):
    result = average._strategy_lifespan(test_points, size, maximize)
    assert result == expected_top_n

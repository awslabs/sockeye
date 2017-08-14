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

"""
Tests sockeye.callback.TrainingMonitor optimization logic
"""
import os
import tempfile

import numpy as np
import pytest

from sockeye import callback
from sockeye import constants as C
from sockeye import utils

test_constants = [('perplexity', np.inf,
                   [{'perplexity': 100.0, '_': 42}, {'perplexity': 50.0}, {'perplexity': 60.0}, {'perplexity': 80.0}],
                   [{'perplexity': 200.0}, {'perplexity': 100.0}, {'perplexity': 100.001}, {'perplexity': 99.99}],
                   [True, True, False, True]),
                  ('accuracy', 0.0,
                   [{'accuracy': 100.0}, {'accuracy': 50.0}, {'accuracy': 60.0}, {'accuracy': 80.0}],
                   [{'accuracy': 200.0}, {'accuracy': 100.0}, {'accuracy': 100.001}, {'accuracy': 99.99}],
                   [True, False, False, False])]


class DummyMetric(object):
    def __init__(self, metric_dict):
        self.metric_dict = metric_dict

    def get_name_value(self):
        for metric_name, value in self.metric_dict.items():
            yield metric_name, value


@pytest.mark.parametrize("optimized_metric, initial_best, train_metrics, eval_metrics, improved_seq",
                         test_constants)
def test_callback(optimized_metric, initial_best, train_metrics, eval_metrics, improved_seq):
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 32
        monitor = callback.TrainingMonitor(batch_size=batch_size,
                                           output_folder=tmpdir,
                                           optimized_metric=optimized_metric)
        assert monitor.optimized_metric == optimized_metric
        assert monitor.get_best_validation_score() == initial_best
        metrics_fname = os.path.join(tmpdir, C.METRICS_NAME)

        for checkpoint, (train_metric, eval_metric, expected_improved) in enumerate(
                zip(train_metrics, eval_metrics, improved_seq), 1):
            monitor.checkpoint_callback(checkpoint, DummyMetric(train_metric))
            assert len(monitor.metrics) == checkpoint
            assert monitor.metrics[-1] == {k + "-train": v for k, v in train_metric.items()}
            improved, best_checkpoint = monitor.eval_end_callback(checkpoint, DummyMetric(eval_metric))
            assert {k + "-val" for k in eval_metric.keys()} <= monitor.metrics[-1].keys()
            assert improved == expected_improved
            assert os.path.exists(metrics_fname)
            metrics = utils.read_metrics_file(metrics_fname)
            _compare_metrics(metrics, monitor.metrics)


def _compare_metrics(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert len(x.items()) == len(y.items())
        for (xk, xv), (yk, yv) in zip(sorted(x.items()), sorted(y.items())):
            assert xk == yk
            assert pytest.approx(xv, yv)


def test_bleu_requires_checkpoint_decoder():
    with pytest.raises(utils.SockeyeError) as e, tempfile.TemporaryDirectory() as tmpdir:
        callback.TrainingMonitor(batch_size=1,
                                 output_folder=tmpdir,
                                 optimized_metric='bleu',
                                 cp_decoder=None)
    assert "bleu requires CheckpointDecoder" == str(e.value)

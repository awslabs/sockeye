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
import pytest
import numpy as np
import sockeye.callback
import tempfile
import os

test_constants = [('perplexity', np.inf, True,
                   [{'perplexity': 100.0, '_': 42}, {'perplexity': 50.0}, {'perplexity': 60.0}, {'perplexity': 80.0}],
                   [{'perplexity': 200.0}, {'perplexity': 100.0}, {'perplexity': 100.001}, {'perplexity': 99.99}],
                   [True, True, False, True]),
                  ('accuracy', -np.inf, False,
                   [{'accuracy': 100.0}, {'accuracy': 50.0}, {'accuracy': 60.0}, {'accuracy': 80.0}],
                   [{'accuracy': 200.0}, {'accuracy': 100.0}, {'accuracy': 100.001}, {'accuracy': 99.99}],
                   [True, False, False, False])]


class DummyMetric(object):
    def __init__(self, metric_dict):
        self.metric_dict = metric_dict

    def get_name_value(self):
        for metric_name, value in self.metric_dict.items():
            yield metric_name, value


@pytest.mark.parametrize("optimized_metric, initial_best, minimize, train_metrics, eval_metrics, improved_seq",
                         test_constants)
def test_callback(optimized_metric, initial_best, minimize, train_metrics, eval_metrics, improved_seq):
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 32
        monitor = sockeye.callback.TrainingMonitor(batch_size=batch_size,
                                                   output_folder=tmpdir,
                                                   optimized_metric=optimized_metric)
        assert monitor.optimized_metric == optimized_metric
        assert monitor.get_best_validation_score() == initial_best
        assert monitor.minimize == minimize

        for checkpoint, (train_metric, eval_metric, expected_improved) in enumerate(
                zip(train_metrics, eval_metrics, improved_seq), 1):
            monitor.checkpoint_callback(checkpoint, DummyMetric(train_metric))
            assert len(monitor.metrics) == checkpoint
            assert monitor.metrics[-1] == {k + "-train": v for k, v in train_metric.items()}
            improved, best_checkpoint = monitor.eval_end_callback(checkpoint, DummyMetric(eval_metric))
            assert {k + "-val" for k in eval_metric.keys()} <= monitor.metrics[-1].keys()
            assert improved == expected_improved
            

def test_bleu_requires_checkpoint_decoder():
    with pytest.raises(AssertionError), tempfile.TemporaryDirectory() as tmpdir:
        sockeye.callback.TrainingMonitor(batch_size=1,
                                         output_folder=tmpdir,
                                         optimized_metric='bleu',
                                         checkpoint_decoder=None)

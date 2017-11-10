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
Provides functionality to track metrics on training and validation data during training and controls
early-stopping.
"""
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import time
from typing import List, Optional, Tuple, Dict

import mxnet as mx

from . import checkpoint_decoder
from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


class TrainingMonitor(object):
    """
    TrainingMonitor logs metrics on training and validation data, submits decoding processes to compute BLEU scores,
    and writes metrics to the model output folder.
    It further controls early stopping as it decides based on the specified metric to optimize, whether the model
    has improved w.r.t to the last checkpoint.
    Technically, TrainingMonitor exposes a couple of callback function that are called in the fit() method of
    TrainingModel.

    :param batch_size: Batch size during training.
    :param output_folder: Folder where model files are written to.
    :param optimized_metric: Name of the metric that controls early stopping.
    :param use_tensorboard: Whether to use Tensorboard logging of metrics.
    :param cp_decoder: Optional CheckpointDecoder instance for BLEU monitoring.
    :param num_concurrent_decodes: Number of concurrent subprocesses to decode validation data.
    """

    def __init__(self,
                 batch_size: int,
                 output_folder: str,
                 optimized_metric: str = C.PERPLEXITY,
                 use_tensorboard: bool = False,
                 cp_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None,
                 num_concurrent_decodes: int = 1) -> None:
        self.output_folder = output_folder
        # stores dicts of metric names & values for each checkpoint
        self.metrics = []  # type: List[Dict]
        self.metrics_filename = os.path.join(output_folder, C.METRICS_NAME)
        self.best_checkpoint = 0
        self.start_tic = time.time()
        self.summary_writer = None
        if use_tensorboard:
            import tensorboard  # pylint: disable=import-error
            log_dir = os.path.join(output_folder, C.TENSORBOARD_NAME)
            if os.path.exists(log_dir):
                logger.info("Deleting existing tensorboard log dir %s", log_dir)
                shutil.rmtree(log_dir)
            logger.info("Logging training events for Tensorboard at '%s'", log_dir)
            self.summary_writer = tensorboard.FileWriter(log_dir)
        self.cp_decoder = cp_decoder
        self.ctx = mp.get_context('spawn') # type: ignore
        self.num_concurrent_decodes = num_concurrent_decodes
        self.decoder_metric_queue = self.ctx.Queue()
        self.decoder_processes = []  # type: List[mp.Process]
        # TODO(fhieber): MXNet Speedometer uses root logger. How to fix this?
        self.speedometer = mx.callback.Speedometer(batch_size=batch_size,
                                                   frequent=C.MEASURE_SPEED_EVERY,
                                                   auto_reset=False)
        utils.check_condition(optimized_metric in C.METRICS, "Unsupported metric: %s" % optimized_metric)
        if optimized_metric == C.BLEU:
            utils.check_condition(self.cp_decoder is not None, "%s requires CheckpointDecoder" % C.BLEU)
        self.optimized_metric = optimized_metric
        self.validation_best = C.METRIC_WORST[self.optimized_metric]
        logger.info("Early stopping by optimizing '%s'", self.optimized_metric)
        self.tic = 0

    def get_best_checkpoint(self) -> int:
        """
        Returns current best checkpoint.
        """
        return self.best_checkpoint

    def get_best_validation_score(self) -> float:
        """
        Returns current best validation result for optimized metric.
        """
        return self.validation_best

    def _is_better(self, value: float) -> bool:
        if C.METRIC_MAXIMIZE[self.optimized_metric]:
            return value > self.validation_best
        else:
            return value < self.validation_best

    def batch_end_callback(self, epoch: int, nbatch: int, metric: mx.metric.EvalMetric):
        """
        Callback function when processing of a data bach is completed.

        :param epoch: Current epoch.
        :param nbatch: Current batch.
        :param metric: Evaluation metric for training data.
        """
        self.speedometer(
            mx.model.BatchEndParam(
                epoch=epoch, nbatch=nbatch, eval_metric=metric, locals=None))

    def checkpoint_callback(self,
                            checkpoint: int,
                            train_metric: mx.metric.EvalMetric,
                            memory_data: Optional[Dict[int, Tuple[int, int]]] = None):
        """
        Callback function when a model checkpoint is performed.
        If TrainingMonitor uses Tensorboard, training metrics are written to the Tensorboard event file.

        :param checkpoint: Current checkpoint.
        :param train_metric: Evaluation metric for training data.
        :param memory_data: Optional data about memory usage.
        """
        metrics = {}
        for name, value in train_metric.get_name_value():
            metrics[name + "-train"] = value
        if memory_data is not None:
            utils.log_gpu_memory_usage(memory_data)
            used_gpu_mem = sum(v[0] for v in memory_data.values())
            metrics['used-gpu-memory'] = used_gpu_mem  # total gpu memory used in MB
        self.metrics.append(metrics)

        if self.summary_writer:
            write_tensorboard(self.summary_writer, metrics, checkpoint)

    def eval_end_callback(self, checkpoint: int, val_metric: mx.metric.EvalMetric) -> Tuple[bool, int]:
        """
        Callback function when processing of held-out validation data is complete.
        Counts time elapsed since the start of training.
        If TrainingMonitor uses Tensorboard, validation metrics are written to the Tensorboard event file.
        If BLEU is monitored with subprocesses, this function collects result from finished decoder processes
        and starts a new one for the current checkpoint.

        :param checkpoint: Current checkpoint.
        :param val_metric: Evaluation metric for validation data.
        :return: Tuple of boolean indicating if model improved on validation data according to the.
                 optimized metric, and the (updated) best checkpoint.
        """
        metrics = {}
        for name, value in val_metric.get_name_value():
            metrics[name + "-val"] = value
        metrics['time-elapsed'] = time.time() - self.start_tic

        if self.summary_writer:
            write_tensorboard(self.summary_writer, metrics, checkpoint)

        if self.cp_decoder:
            self._empty_decoder_metric_queue()
            self._start_decode_process(checkpoint)

        self.metrics[-1].update(metrics)
        utils.write_metrics_file(self.metrics, self.metrics_filename)

        has_improved, best_checkpoint = self._find_best_checkpoint()
        return has_improved, best_checkpoint

    def _find_best_checkpoint(self):
        """
        Returns True if optimized_metric has improved since the last call of
        this function, together with the best checkpoint
        """
        has_improved = False
        for checkpoint, metric_dict in enumerate(self.metrics, 1):
            value = metric_dict.get(self.optimized_metric + "-val",
                                    self.validation_best)
            if self._is_better(value):
                self.validation_best = value
                self.best_checkpoint = checkpoint
                has_improved = True

        if has_improved:
            logger.info("Validation-%s improved to %f.", self.optimized_metric,
                        self.validation_best)
        else:
            logger.info("Validation-%s has not improved, best so far: %f",
                        self.optimized_metric, self.validation_best)
        return has_improved, self.best_checkpoint

    def _start_decode_process(self, checkpoint):
        self._wait_for_decode_slot()
        output_name = os.path.join(self.output_folder, C.DECODE_OUT_NAME % checkpoint)
        process = self.ctx.Process(
            target=_decode_and_evaluate,
            args=(self.cp_decoder,
                  checkpoint,
                  output_name,
                  self.decoder_metric_queue))
        process.name = 'Decoder-%d' % checkpoint
        logger.info("Starting process: %s", process.name)
        process.start()
        self.decoder_processes.append(process)

    def _empty_decoder_metric_queue(self):
        """
        Get metric results from decoder_process queue and optionally write to tensorboard logs
        """
        while not self.decoder_metric_queue.empty():
            decoded_checkpoint, decoder_metrics = self.decoder_metric_queue.get()
            logger.info("Checkpoint [%d]: Decoder finished (%s)",
                        decoded_checkpoint, decoder_metrics)
            self.metrics[decoded_checkpoint - 1].update(decoder_metrics)
            if self.summary_writer:
                write_tensorboard(self.summary_writer, decoder_metrics,
                                  decoded_checkpoint)

    def _wait_for_decode_slot(self, timeout: int = 5):
        while len(self.decoder_processes) == self.num_concurrent_decodes:
            self.decoder_processes = [p for p in self.decoder_processes
                                      if p.is_alive()]
            time.sleep(timeout)

    def stop_fit_callback(self):
        """
        Callback function when fitting is stopped. Collects results from decoder processes and writes their results.
        """
        for process in self.decoder_processes:
            if process.is_alive():
                logger.info("Waiting for %s process to finish." % process.name)
            process.join()
        self._empty_decoder_metric_queue()
        utils.write_metrics_file(self.metrics, self.metrics_filename)

    def save_state(self, fname: str):
        """
        Saves the state: current metrics and best checkpoint.

        :param fname: Name of the file to save the state to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self.metrics, fp)
            pickle.dump(self.best_checkpoint, fp)

    def load_state(self, fname: str):
        """
        Loads the state: current metrics and best checkpoint.

        :param fname: Name of the file to load the state from.
        """
        with open(fname, "rb") as fp:
            self.metrics = pickle.load(fp)
            self.best_checkpoint = pickle.load(fp)


def _decode_and_evaluate(checkpoint_decoder: checkpoint_decoder.CheckpointDecoder,
                         checkpoint: int,
                         output_name: str,
                         queue: mp.Queue):
    """
    Decodes and evaluates using given checkpoint_decoder and puts result in the queue,
    indexed by the checkpoint.
    """
    metrics = checkpoint_decoder.decode_and_evaluate(checkpoint, output_name)
    queue.put((checkpoint, metrics))


def write_tensorboard(summary_writer,
                      metrics: Dict[str, float],
                      checkpoint: int):
    """
    Writes a Tensorboard scalar event to the given SummaryWriter.

    :param summary_writer: A Tensorboard SummaryWriter instance.
    :param metrics: Mapping of metric names to their values.
    :param checkpoint: Current checkpoint.
    """
    from tensorboard.summary import scalar  # pylint: disable=import-error
    for name, value in metrics.items():
        summary_writer.add_summary(
            scalar(
                name=name, scalar=value), global_step=checkpoint)

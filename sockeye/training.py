# Copyright 2017--2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Code for training
"""
import glob
import logging
import os
import pickle
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Dict, List, Optional, Iterable, Tuple, Union, Set

import mxnet as mx
import numpy as onp
from mxnet import amp, np, npx, gluon

from . import average
from . import constants as C
from . import data_io
from . import horovod_mpi
from . import loss
from . import lr_scheduler
from . import parallel
from . import utils
from . import vocab
from .checkpoint_decoder import CheckpointDecoder
from .config import Config
from .model import SockeyeModel
from .optimizers import OptimizerConfig

logger = logging.getLogger(__name__)


def global_norm(ndarrays: List[np.ndarray]) -> float:
    # accumulate in a list, as item() is blocking and this way we can run the norm calculation in parallel.
    norms = [np.square(np.linalg.norm(arr)) for arr in ndarrays if arr is not None]
    return sqrt(sum(norm.item() for norm in norms))


@dataclass
class TrainerConfig(Config):
    output_dir: str
    early_stopping_metric: str
    max_params_files_to_keep: int
    keep_initializations: bool
    max_params_files_to_cache: int
    cache_strategy: str
    cache_metric: str
    checkpoint_interval: int
    max_num_checkpoint_not_improved: int
    checkpoint_improvement_threshold: float
    max_checkpoints: Optional[int] = None
    min_samples: Optional[int] = None
    max_samples: Optional[int] = None
    min_updates: Optional[int] = None
    max_updates: Optional[int] = None
    min_epochs: Optional[int] = None
    max_epochs: Optional[int] = None
    max_seconds: Optional[int] = None
    update_interval: int = 1
    stop_training_on_decoder_failure: bool = False


class TrainState:
    """
    Stores the state an EarlyStoppingTrainer instance.
    """

    _pickle_slots = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint', 'batches',
                     'updates', 'samples', 'gradient_norm', 'metrics', 'start_tic', '_tic_last_time_elapsed',
                     '_time_elapsed', 'early_stopping_metric', 'best_metric', 'best_metric_history',
                     'best_checkpoint', 'converged', 'diverged']

    __slots__ = _pickle_slots + ['gradients']

    def __init__(self, early_stopping_metric: str) -> None:
        self.num_not_improved = 0
        self.epoch = 0
        self.checkpoint = 0
        self.best_checkpoint = 0
        self.batches = 0
        self.updates = 0
        self.samples = 0
        self.gradient_norm = None  # type: Optional[float]
        self.gradients = {}  # type: Dict[str, List[np.ndarray]]
        # stores dicts of metric names & values for each checkpoint
        self.metrics = []  # type: List[Dict]
        self.start_tic = time.time()
        self._tic_last_time_elapsed = self.start_tic
        self._time_elapsed = 0.0
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = C.METRIC_WORST[early_stopping_metric]
        # List of the last N best metrics, used for threshold-based stopping
        self.best_metric_history = deque([self.best_metric])
        self.best_checkpoint = 0
        self.converged = False
        self.diverged = False

    def save(self, fname: str):
        """
        Saves this training state to fname.
        """
        self.update_time_elapsed()
        assert len(self.metrics) == self.checkpoint
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname: str) -> 'TrainState':
        """
        Loads a training state from fname.
        """
        with open(fname, "rb") as fp:
            state = pickle.load(fp)
            state._tic_last_time_elapsed = time.time()
            assert len(state.metrics) == state.checkpoint
            return state

    def update_time_elapsed(self):
        current_time = time.time()
        self._time_elapsed += current_time - self._tic_last_time_elapsed
        self._tic_last_time_elapsed = current_time

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def __getstate__(self):
        return {k: getattr(self, k) for k in self._pickle_slots}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.gradients = {}


class GluonEarlyStoppingTrainer:

    def __init__(self,
                 config: TrainerConfig,
                 optimizer_config: OptimizerConfig,
                 sockeye_model: SockeyeModel,
                 trainer: gluon.Trainer,
                 loss_functions: List[loss.Loss],
                 context: List[mx.context.Context],
                 dtype: str,
                 using_amp: bool = False,
                 custom_metrics_logger: Optional[Callable] = None,
                 checkpoint_callback: Optional[Callable] = None) -> None:
        self.config = config
        self.optimizer_config = optimizer_config
        self.model = sockeye_model
        self.trainer = trainer
        self.loss_functions = loss_functions
        self.context = context
        self.dtype = dtype
        self.using_amp = using_amp
        self._parallel = parallel.Parallel(len(context) if len(context) > 1 else 0,
                                           ParallelModel(sockeye_model,
                                                         loss_functions,
                                                         trainer,
                                                         using_amp=using_amp))
        self.state = None  # type: Optional[TrainState]
        self._speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        self._custom_metrics_logger = custom_metrics_logger
        self._tflogger = TensorboardLogger(logdir=os.path.join(self.config.output_dir, C.TENSORBOARD_NAME))
        self.checkpoint_callback = checkpoint_callback

    def fit(self,
            train_iter: data_io.BaseParallelSampleIter,
            validation_iter: data_io.BaseParallelSampleIter,
            checkpoint_decoder: Optional[CheckpointDecoder] = None):
        logger.info("Early stopping by optimizing '%s'", self.config.early_stopping_metric)

        if self.config.early_stopping_metric in C.METRICS_REQUIRING_DECODER:
            utils.check_condition(checkpoint_decoder is not None,
                                  "%s requires CheckpointDecoder" % self.config.early_stopping_metric)

        resume_training = os.path.exists(self.training_state_dirname)
        if resume_training:
            logger.info("Found partial training in '%s'. Resuming from saved state.", self.training_state_dirname)
            self._load_training_state(train_iter)
        else:
            self.state = TrainState(self.config.early_stopping_metric)
            self.model.save_config(self.config.output_dir)
            self.model.save_version(self.config.output_dir)
            # self._save_training_state(train_iter)
            # self._save_trainer_states(self.best_optimizer_states_fname)  # not saving due to deferred initialization
            logger.info("Training started.")

        tic = time.time()

        if self.config.max_checkpoints is not None:
            self.config.max_updates = self.state.updates + self.config.max_checkpoints * self.config.checkpoint_interval
            logger.info("Resetting max_updates to %d + %d * %d = %d in order to implement stopping "
                        "after (an additional) %d checkpoints.",
                        self.state.updates,
                        self.config.max_checkpoints,
                        self.config.checkpoint_interval,
                        self.config.max_updates,
                        self.config.max_checkpoints)

        checkpoint_up_to_date = False
        while True:
            if self.config.max_epochs is not None and self.state.epoch == self.config.max_epochs:
                logger.info("Maximum # of epochs (%s) reached.", self.config.max_epochs)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break

            if self.config.max_updates is not None and self.state.updates == self.config.max_updates:
                logger.info("Maximum # of updates (%s) reached.", self.config.max_updates)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break

            if self.config.max_samples is not None and self.state.samples >= self.config.max_samples:
                logger.info("Maximum # of samples (%s) reached", self.config.max_samples)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break

            did_grad_step = self._step(batch=train_iter.next())
            checkpoint_up_to_date = checkpoint_up_to_date and not did_grad_step

            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()

            if self.state.updates > 0 and self.state.batches % (
                    self.config.checkpoint_interval * self.config.update_interval) == 0:
                time_cost = time.time() - tic
                self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                checkpoint_up_to_date = True

                if self.config.max_seconds is not None and self.state.time_elapsed >= self.config.max_seconds:
                    logger.info("Maximum # of seconds (%s) reached. Training ran for %d seconds.",
                                self.config.max_seconds, self.state.time_elapsed)
                    break

                if self.state.converged or self.state.diverged:
                    break

                tic = time.time()

        logger.info("Training finished%s. Best checkpoint: %d. Best validation %s: %.6f",
                    ", can be continued later" if not self.state.converged else "",
                    self.state.best_checkpoint, self.state.early_stopping_metric, self.state.best_metric)

        # Always keep the training state to allow continuing training with
        # different stopping criteria
        self._cleanup(keep_training_state=True)
        return self.state

    def _create_checkpoint(self, checkpoint_decoder: CheckpointDecoder, time_cost: float,
                           train_iter: data_io.BaseParallelSampleIter, validation_iter: data_io.BaseParallelSampleIter):
        """
        Creates a checkpoint, which will update self.state.converged/self.state.diverged, evaluate validation
        metrics and update the best known parameters accordingly.
        """
        self.state.checkpoint += 1
        # save parameters and evaluate on validation data
        self._save_params()
        train_metrics = [lf.metric for lf in self.loss_functions]
        logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f",
                    self.state.checkpoint, self.state.updates, self.state.epoch,
                    self.state.samples, time_cost, self.config.checkpoint_interval / time_cost)
        logger.info('Checkpoint [%d]\t%s', self.state.checkpoint,
                    "\t".join("Train-%s" % str(metric) for metric in train_metrics))
        val_metrics = self._evaluate(self.state.checkpoint, validation_iter, checkpoint_decoder)
        npx.waitall()
        has_improved = self._determine_improvement(val_metrics)
        self.state.converged = self._determine_convergence()
        self.state.diverged = self._determine_divergence(val_metrics)
        self._adjust_learning_rate(has_improved)
        if has_improved:
            self._update_best_params()
            self._save_trainer_states(self.best_optimizer_states_fname)
            self._save_lr_scheduler(self.best_lr_scheduler_fname)
        self._write_and_log_metrics(train_metrics=train_metrics, val_metrics=val_metrics)
        for metric in train_metrics:
            metric.reset()
        self._save_training_state(train_iter)
        if self.checkpoint_callback:
            self.checkpoint_callback(self.state.checkpoint)

    def _forward_backward(self, batch: data_io.Batch):
        """
        Performs forward-backward pass on a batch in data-parallel mode.

        :param batch: Current data batch.
        :return: List loss outputs (tuple of loss value and number of samples) for each loss function.
        """
        # split batch into shards
        batch = batch.split_and_load(ctx=self.context)

        # send sharded inputs to the backend
        for inputs, labels in batch.shards():
            self._parallel.put((inputs, labels))

        # get outputs from parallel requests to the backend. Each shard output contains a list of tuples, one for each
        # loss function of the form: (loss_value, num_samples).
        sharded_outputs = [self._parallel.get() for _ in range(len(self.context))]

        # repack outputs into a list of loss_values (length = number of shards) for each loss function
        sharded_outputs_per_loss_function = list(zip(*sharded_outputs))

        # sum loss values (on the cpu) and number of samples for each loss function
        output_per_loss_function = [
            tuple(npx.add_n(*(s.as_in_context(mx.cpu()) for s in shard)) for shard in zip(*outs)) for outs in
            sharded_outputs_per_loss_function]
        return output_per_loss_function

    def _step(self, batch: data_io.Batch) -> bool:
        self.state.batches += 1
        loss_outputs = self._forward_backward(batch)

        did_grad_step = False
        if self.config.update_interval == 1 or self.state.batches % self.config.update_interval == 0:
            # `step` rescales the gradients for the number of batches in this
            # update.
            self.trainer.step(batch_size=self.config.update_interval)
            if self.config.update_interval > 1:
                # Multi-batch updates sum gradients for each batch instead of
                # overwriting, so gradients must be manually zeroed after each
                # update.
                self.model.zero_grad()
            self.state.updates += 1
            did_grad_step = True

        self.state.samples += batch.samples
        for loss_func, (loss_value, num_samples) in zip(self.loss_functions, loss_outputs):
            loss_func.metric.update(loss_value.item(), num_samples.item())
        self._speedometer(self.state.epoch, self.state.batches,
                          self.state.updates, batch.samples, batch.tokens, (lf.metric for lf in self.loss_functions))
        return did_grad_step

    def _evaluate(self, checkpoint: int, data_iter, checkpoint_decoder: Optional[CheckpointDecoder]) -> List[loss.LossMetric]:
        """
        Computes loss(es) on validation data and returns their metrics.
        :param data_iter: Validation data iterator.
        :return: List of validation metrics, same order as self.loss_functions.
        """
        data_iter.reset()
        val_metrics = [lf.create_metric() for lf in self.loss_functions]
        for batch in data_iter:
            batch = batch.split_and_load(ctx=self.context)
            sharded_loss_outputs = []  # type: List[List[Tuple[np.ndarray, np.ndarray]]]
            for inputs, labels in batch.shards():
                outputs = self.model(*inputs)  # type: Dict[str, np.ndarray]
                loss_outputs = [loss_function(outputs, labels) for loss_function in self.loss_functions]
                sharded_loss_outputs.append(loss_outputs)

            # repack outputs into a list of loss_values (length = number of shards) for each loss function
            sharded_loss_outputs_per_loss_function = list(zip(*sharded_loss_outputs))
            # sum loss values (on the cpu) and number of samples for each loss function
            output_per_loss_function = [tuple(npx.add_n(*(s.as_in_context(mx.cpu()) for s in shard))
                                        for shard in zip(*outs)) for outs in sharded_loss_outputs_per_loss_function]
            # update validation metrics for batch
            for loss_metric, (loss_value, num_samples) in zip(val_metrics, output_per_loss_function):
                loss_metric.update(loss_value.item(), num_samples.item())

        # Optionally run the checkpoint decoder
        if checkpoint_decoder is not None:
            output_name = os.path.join(self.config.output_dir, C.DECODE_OUT_NAME.format(checkpoint=checkpoint))
            decoder_metrics = checkpoint_decoder.decode_and_evaluate(output_name=output_name)
            for metric_name, metric_value in decoder_metrics.items():
                assert metric_name not in val_metrics, "Duplicate validation metric %s" % metric_name
                metric = loss.LossMetric(name=metric_name)
                metric.update(metric_value, num_samples=1)
                val_metrics.append(metric)

        logger.info('Checkpoint [%d]\t%s',
                    self.state.checkpoint, "\t".join("Validation-%s" % str(lm) for lm in val_metrics))

        return val_metrics

    def _determine_improvement(self, val_metrics: List[loss.LossMetric]) -> bool:
        """
        Determines whether early stopping metric on validation data improved and updates best value and checkpoint in
        the state.
        :param val_metrics: Validation metrics.
        :return: Whether model has improved on held-out data since last checkpoint.
        """
        value = None
        value_is_better = False
        for val_metric in val_metrics:
            if val_metric.name == self.config.early_stopping_metric:
                value = val_metric.get()
                # When using Horovod, the primary worker makes an authoritative
                # check of whether metric value has improved and broadcasts the
                # result to secondary workers.  Non-determinism in the order of
                # GPU operations can lead to slight numeric variation across
                # workers, causing potential desync if each worker makes its own
                # check for key training decisions (reducing learning rate,
                # early stopping, etc.).
                if not horovod_mpi.using_horovod() or horovod_mpi.hvd.rank() == 0:
                    # Horovod primary worker or not using Horovod: make
                    # authoritative metric check.
                    value_is_better = utils.metric_value_is_better(value,
                                                                   self.state.best_metric,
                                                                   self.config.early_stopping_metric)
                if horovod_mpi.using_horovod():
                    # Broadcast result across workers.
                    value_is_better = horovod_mpi.MPI.COMM_WORLD.bcast(value_is_better, root=0)
                if value_is_better:
                    logger.info("Validation-%s improved to %f (delta=%f).", self.config.early_stopping_metric,
                                value, abs(value - self.state.best_metric))
                    self.state.best_metric = value
                    self.state.best_checkpoint = self.state.checkpoint
                    self.state.num_not_improved = 0
        assert value is not None, "Early stopping metric %s not found in validation metrics." % self.config.early_stopping_metric
        if not value_is_better:
            self.state.num_not_improved += 1
            logger.info("Validation-%s has not improved for %d checkpoints, best so far: %f",
                        self.config.early_stopping_metric, self.state.num_not_improved, self.state.best_metric)
        # Update best metric history
        self.state.best_metric_history.append(self.state.best_metric)
        if (self.config.max_num_checkpoint_not_improved is not None
                and len(self.state.best_metric_history) > self.config.max_num_checkpoint_not_improved + 1):
            self.state.best_metric_history.popleft()

        return value_is_better

    def _determine_convergence(self) -> bool:
        """
        True if model has converged w.r.t early stopping criteria (patience).
        Order: first check required minimums (samples, updates, epochs), then
        check early stopping criteria (checkpoints not improved).
        """
        if self.config.min_samples is not None and self.state.samples < self.config.min_samples:
            logger.info("Minimum number of samples (%d) not reached yet: %d",
                        self.config.min_samples, self.state.samples)
            return False

        if self.config.min_updates is not None and self.state.updates < self.config.min_updates:
            logger.info("Minimum number of updates (%d) not reached yet: %d",
                        self.config.min_updates, self.state.updates)
            return False

        if self.config.min_epochs is not None and self.state.epoch < self.config.min_epochs:
            logger.info("Minimum number of epochs (%d) not reached yet: %d",
                        self.config.min_epochs, self.state.epoch)
            return False

        if (self.config.max_num_checkpoint_not_improved is not None
                and 0 <= self.config.max_num_checkpoint_not_improved
                and self.state.checkpoint >= self.config.max_num_checkpoint_not_improved):
            # When using Horovod, the primary worker makes the authoritative
            # calculation of improvement over the window for evaluating stopping
            window_improvement = 0.
            if not horovod_mpi.using_horovod() or horovod_mpi.hvd.rank() == 0:
                window_improvement = abs(self.state.best_metric - self.state.best_metric_history[0])
            if horovod_mpi.using_horovod():
                window_improvement = horovod_mpi.MPI.COMM_WORLD.bcast(window_improvement, root=0)

            # <= to correctly handle threshold == 0
            if window_improvement <= self.config.checkpoint_improvement_threshold:
                logger.info("Maximum number of not improved checkpoints reached: "
                            "improvement %f <= %f over %d checkpoints", window_improvement,
                            self.config.checkpoint_improvement_threshold, self.config.max_num_checkpoint_not_improved)
                return True
            else:
                logger.info("Sufficient improvement to continue: %f > %f over %d checkpoints", window_improvement,
                            self.config.checkpoint_improvement_threshold, self.config.max_num_checkpoint_not_improved)

        return False

    def _determine_divergence(self, val_metrics: List[loss.LossMetric]) -> bool:
        """
        True if last perplexity is infinite or >2*target_vocab_size.
        """
        # (5) detect divergence with respect to the perplexity value at the last checkpoint
        last_ppl = float('nan')
        for metric in val_metrics:
            if metric.name == C.PERPLEXITY:
                last_ppl = metric.get()
                break
        # using a double of uniform distribution's value as a threshold
        if not np.isfinite(last_ppl) or last_ppl > 2 * self.model.config.vocab_target_size:
            logger.warning("Model optimization diverged. Last checkpoint's perplexity: %f", last_ppl)
            return True
        return False

    def _adjust_learning_rate(self, has_improved: bool):
        """
        Adjusts the optimizer learning rate if required.
        """
        scheduler = self.trainer.optimizer.lr_scheduler
        if scheduler is not None:
            if issubclass(type(scheduler), lr_scheduler.AdaptiveLearningRateScheduler):
                lr_adjusted = scheduler.new_evaluation_result(has_improved)  # type: ignore
            else:
                lr_adjusted = False
            if lr_adjusted and not has_improved:
                logger.info("Loading model parameters and optimizer states from best checkpoint: %d",
                            self.state.best_checkpoint)
                adjusted_lr = self.trainer.optimizer.lr_scheduler.lr
                # trainer.load_states also reloads the parameters
                if os.path.exists(self.best_optimizer_states_fname):
                    self._load_trainer_states(self.best_optimizer_states_fname)
                # state loading replaces the lr_scheduler instance which then contains the old learning rate,
                # overwriting here. TODO: make this better...
                self.trainer.optimizer.lr_scheduler.lr = adjusted_lr

    def _write_and_log_metrics(self, train_metrics: Iterable[loss.LossMetric], val_metrics: Iterable[loss.LossMetric]):
        """
        Updates metrics for current checkpoint.
        Writes all metrics to the metrics file, optionally logs to tensorboard, and sends metrics to custom logger.
        """
        data = {"epoch": self.state.epoch,
                "learning-rate": (self.trainer.learning_rate if self.trainer.optimizer.lr_scheduler is None
                                  else self.trainer.optimizer.lr_scheduler.lr),
                "gradient-norm": self.state.gradient_norm,
                "time-elapsed": self.state.time_elapsed}
        gpu_memory_usage = utils.get_gpu_memory_usage(self.context)
        data['used-gpu-memory'] = sum(v[0] for v in gpu_memory_usage.values())
        data['converged'] = self.state.converged
        data['diverged'] = self.state.diverged

        for metric in train_metrics:
            data["%s-train" % metric.name] = metric.get()
        for metric in val_metrics:
            data["%s-val" % metric.name] = metric.get()

        self.state.metrics.append(data)
        utils.write_metrics_file(self.state.metrics, self.metrics_fname)

        self._tflogger.log_metrics(metrics=data, checkpoint=self.state.checkpoint)
        safe_custom_metrics_logger(logging_function=self._custom_metrics_logger,
                                   metrics=data,
                                   global_step=self.state.checkpoint)

    def _update_best_params(self):
        """
        Updates the params.best link to the latest best parameter file.
        """
        actual_best_params_fname = C.PARAMS_NAME % self.state.best_checkpoint
        if os.path.lexists(self.best_params_fname):
            os.remove(self.best_params_fname)
        os.symlink(actual_best_params_fname, self.best_params_fname)
        logger.info("'%s' now points to '%s'", self.best_params_fname, actual_best_params_fname)

    def _save_params(self):
        """
        Saves model parameters at current checkpoint and optionally cleans up older parameter files to save disk space.
        """
        self.model.save_parameters(self.current_params_fname)
        cleanup_params_files(self.config.output_dir, self.config.max_params_files_to_keep, self.state.checkpoint,
                             self.state.best_checkpoint, self.config.keep_initializations,
                             self.config.max_params_files_to_cache, self.config.cache_metric, self.config.cache_strategy)

    def _save_trainer_states(self, fname):
        trainer_save_states_no_dump_optimizer(self.trainer, fname)
        logger.info('Saved optimizer states to "%s"', fname)

    def _load_trainer_states(self, fname):
        self.trainer.load_states(fname)
        logger.info('Loaded optimizer states from "%s"', fname)

    def _save_lr_scheduler(self, fname):
        if self.trainer.optimizer.lr_scheduler is not None:
            with open(fname, "wb") as fp:
                pickle.dump(self.trainer.optimizer.lr_scheduler, fp)
            logger.info("Saved '%s' to '%s'", self.trainer.optimizer.lr_scheduler, fname)

    def _load_lr_scheduler(self, fname):
        if os.path.exists(fname):
            with open(fname, "rb") as fp:
                self.trainer.optimizer.lr_scheduler = pickle.load(fp)
            logger.info("Loaded '%s' from '%s'", self.trainer.optimizer.lr_scheduler, fname)
        self.trainer.optimizer.begin_num_update = self.state.updates
        self.trainer.optimizer.num_update = self.state.updates

    def _save_training_state(self, train_iter: data_io.BaseParallelSampleIter):
        """
        Saves current training state.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(self.config.output_dir, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)

        # (1) Parameters: link current file
        params_base_fname = C.PARAMS_NAME % self.state.checkpoint
        params_file = os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME)
        if os.path.exists(params_file):
            os.unlink(params_file)
        os.symlink(os.path.join("..", params_base_fname), params_file)

        # (2) Optimizer states
        opt_state_fname = os.path.join(training_state_dirname, C.OPT_STATES_LAST)
        self._save_trainer_states(opt_state_fname)

        # (3) Data iterator
        train_iter.save_state(os.path.join(training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # (4) Random generators
        # RNG states: python's random and onp.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(onp.random.get_state(), fp)

        # (5) Training state
        self.state.save(os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # (5.5) lr_scheduler
        lr_scheduler_fname = os.path.join(training_state_dirname, C.LR_SCHEDULER_LAST)
        self._save_lr_scheduler(lr_scheduler_fname)

        # (6) AMP loss scaler state
        if self.using_amp:
            with open(os.path.join(training_state_dirname, C.AMP_LOSS_SCALER_STATE_NAME), "wb") as fp:
                pickle.dump([self.trainer._amp_loss_scaler._loss_scale,
                             self.trainer._amp_loss_scaler._next_loss_scale,
                             self.trainer._amp_loss_scaler._unskipped], fp)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(self.config.output_dir, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(self.training_state_dirname):
            os.rename(self.training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, self.training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            try:
                shutil.rmtree(delete_training_state_dirname)
            except FileNotFoundError:
                # This can be occur on file systems with higher latency, such as
                # distributed file systems.  While repeated occurrences of this
                # warning may indicate a problem, seeing one or two warnings
                # during training is usually fine.
                logger.warning('Directory has already been removed: %s', delete_training_state_dirname)

    def _load_training_state(self, train_iter: data_io.BaseParallelSampleIter):
        """
        Loads the full training state from disk.
        :param train_iter: training data iterator.
        """
        # (1) Parameters
        params_fname = os.path.join(self.training_state_dirname, C.TRAINING_STATE_PARAMS_NAME)
        self.model.load_parameters(params_fname, ctx=self.context, allow_missing=False, ignore_extra=False)

        # (2) Optimizer states
        opt_state_fname = os.path.join(self.training_state_dirname, C.OPT_STATES_LAST)
        self._load_trainer_states(opt_state_fname)

        # (3) Data Iterator
        train_iter.load_state(os.path.join(self.training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # (4) Random generators
        # RNG states: python's random and onp.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(self.training_state_dirname, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            onp.random.set_state(pickle.load(fp))

        # (5) Training state
        self.state = TrainState.load(os.path.join(self.training_state_dirname, C.TRAINING_STATE_NAME))

        # (5.5) lr_scheduler
        lr_scheduler_fname = os.path.join(self.training_state_dirname, C.LR_SCHEDULER_LAST)
        self._load_lr_scheduler(lr_scheduler_fname)

        # (6) AMP loss scaler state
        if self.using_amp:
            # Load loss scaler state
            with open(os.path.join(self.training_state_dirname, C.AMP_LOSS_SCALER_STATE_NAME), "rb") as fp:
                (self.trainer._amp_loss_scaler._loss_scale,
                 self.trainer._amp_loss_scaler._next_loss_scale,
                 self.trainer._amp_loss_scaler._unskipped) = pickle.load(fp)

        logger.info("Training State: epoch=%d, checkpoint=%d batches=%d updates=%d best_metric=%.2f, " \
                    "best_checkpoint=%d time_elapsed=%d" % (
                        self.state.epoch, self.state.checkpoint, self.state.batches, self.state.updates,
                        self.state.best_metric, self.state.best_checkpoint, self.state.time_elapsed))

    def _cleanup(self, keep_training_state=False):
        """
        Cleans parameter files, training state directory and waits for remaining decoding processes.
        """
        cleanup_params_files(self.config.output_dir, self.config.max_params_files_to_keep,
                             self.state.checkpoint, self.state.best_checkpoint, self.config.keep_initializations,
                             self.config.max_params_files_to_cache, self.config.cache_metric, self.config.cache_strategy)

        if not keep_training_state:
            if os.path.exists(self.training_state_dirname):
                shutil.rmtree(self.training_state_dirname)
            if os.path.exists(self.best_optimizer_states_fname):
                os.remove(self.best_optimizer_states_fname)
            if os.path.exists(self.best_lr_scheduler_fname):
                os.remove(self.best_lr_scheduler_fname)

    @property
    def metrics_fname(self) -> str:
        return os.path.join(self.config.output_dir, C.METRICS_NAME)

    @property
    def current_params_fname(self) -> str:
        return os.path.join(self.config.output_dir, C.PARAMS_NAME % self.state.checkpoint)

    @property
    def best_params_fname(self) -> str:
        return os.path.join(self.config.output_dir, C.PARAMS_BEST_NAME)

    @property
    def training_state_dirname(self) -> str:
        return os.path.join(self.config.output_dir, C.TRAINING_STATE_DIRNAME)

    @property
    def best_optimizer_states_fname(self) -> str:
        return os.path.join(self.config.output_dir, C.OPT_STATES_BEST)

    @property
    def best_lr_scheduler_fname(self) -> str:
        return os.path.join(self.config.output_dir, C.LR_SCHEDULER_BEST)


class ParallelModel(parallel.Parallelizable):

    def __init__(self,
                 model: Callable,
                 loss_functions: List[loss.Loss],
                 trainer: gluon.Trainer,
                 using_amp: bool = False) -> None:
        self.model = model
        self.loss_functions = loss_functions
        self.trainer = trainer
        self.using_amp = using_amp

    def forward_backward(self, shard: Tuple) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Applies forward-backward pass for a single shard of a batch (data-parallel training).
        """
        inputs, labels = shard
        with mx.autograd.record():
            outputs = self.model(*inputs)  # type: Dict[str, np.ndarray]
            loss_outputs = [loss_function(outputs, labels) for loss_function in self.loss_functions]
            loss_values = (v for v, _ in loss_outputs)
            sum_losses = npx.add_n(*loss_values)
            if self.using_amp:
                # AMP applies dynamic loss scaling to the losses (scale up) and
                # the Trainer (scale down).
                with amp.scale_loss(sum_losses, self.trainer) as scaled_loss:
                    mx.autograd.backward(scaled_loss)
            else:
                # backward on the sum of losses, weights are defined in the loss blocks themselves.
                sum_losses.backward()
        return loss_outputs


class TensorboardLogger:
    """
    Thin wrapper for MXBoard API to log training events.
    Flushes logging events to disk every 60 seconds.

    :param logdir: Directory to write Tensorboard event files to.
    :param source_vocab: Optional source vocabulary to log source embeddings.
    :param target_vocab: Optional target vocabulary to log target and output embeddings.
    """

    def __init__(self,
                 logdir: str,
                 source_vocab: Optional[vocab.Vocab] = None,
                 target_vocab: Optional[vocab.Vocab] = None) -> None:
        self.logdir = logdir
        self.source_labels = vocab.get_ordered_tokens_from_vocab(source_vocab) if source_vocab is not None else None
        self.target_labels = vocab.get_ordered_tokens_from_vocab(target_vocab) if target_vocab is not None else None
        try:
            import mxboard
            logger.info("Logging training events for Tensorboard at '%s'", self.logdir)
            self._writer = mxboard.SummaryWriter(logdir=self.logdir, flush_secs=60, verbose=False)
        except ImportError:
            logger.info("mxboard not found. Consider 'pip install mxboard' to log events to Tensorboard.")
            self._writer = None

    def log_metrics(self, metrics: Dict[str, Union[float, int, np.ndarray]], checkpoint: int):
        if self._writer is None:
            return

        for name, value in metrics.items():
            if isinstance(value, np.ndarray):
                if np.isfinite(value).sum().item() == value.size:
                    self._writer.add_histogram(tag=name, values=value, bins=100, global_step=checkpoint)
                else:
                    logger.warning("Histogram of %s not logged to tensorboard because of infinite data.")
            elif value is None:
                continue
            else:
                self._writer.add_scalar(tag=name, value=value, global_step=checkpoint)
        self._writer.flush()

    def log_source_embedding(self, embedding: np.ndarray, checkpoint: int):
        if self._writer is None or self.source_labels is None:
            return
        self._writer.add_embedding(tag="source", embedding=embedding, labels=self.source_labels, global_step=checkpoint)

    def log_target_embedding(self, embedding: np.ndarray, checkpoint: int):
        if self._writer is None or self.target_labels is None:
            return
        self._writer.add_embedding(tag="target", embedding=embedding, labels=self.target_labels, global_step=checkpoint)

    def log_output_embedding(self, embedding: np.ndarray, checkpoint: int):
        if self._writer is None or self.target_labels is None:
            return
        self._writer.add_embedding(tag="output", embedding=embedding, labels=self.target_labels, global_step=checkpoint)


class Speedometer:
    """
    Custom Speedometer to log samples and words per second.
    """

    def __init__(self, frequency: int = 50, auto_reset: bool = True) -> None:
        self.frequency = frequency
        self.init = False
        self.tic = 0.0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.samples = 0
        self.tokens = 0
        self.msg = 'E=%d B=%d\ts/sec=%.2f tok/sec=%.2f u/sec=%.2f\t'

    def __call__(self, epoch: int, batches: int, updates: int, samples: int,
                 tokens: int, metrics: Optional[Iterable[loss.LossMetric]] = None):
        count = batches
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.samples += samples
        self.tokens += tokens

        if self.init:
            if count % self.frequency == 0:
                toc = (time.time() - self.tic)
                update_interval = batches / max(1, updates)
                updates_per_sec = self.frequency / update_interval / toc
                samples_per_sec = self.samples / toc
                tokens_per_sec = self.tokens / toc
                self.samples = 0
                self.tokens = 0

                if metrics is not None:
                    metric_values = []  # type: List[Tuple[str, float]]
                    for metric in metrics:
                        metric_values.append((metric.short_name, metric.get()))
                        if self.auto_reset:
                            metric.reset()
                    logger.info(self.msg + '%s=%f ' * len(metric_values),
                                epoch, count, samples_per_sec, tokens_per_sec, updates_per_sec, *sum(metric_values, ()))

                else:
                    logger.info(self.msg, epoch, count, samples_per_sec, tokens_per_sec, updates_per_sec)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def safe_custom_metrics_logger(logging_function: Callable,
                               metrics: Dict,
                               global_step: int = None):
    """
    A thin wrapper for calling a custom metrics logging function, if supplied. As it uses an external function,
    it should never throw an exception. If there is no logging_function supplied, the function does nothing.
    :param logging_function: The function supplied by a caller of sockeye.train
    :param metrics: A non-empty dict of (nonempty str, float/int/bool) pairs.
    :param global_step: Optional argument, which can be used e.g. by Tensorboard.
    """
    if logging_function is None:
        return
    try:
        logging_function(metrics, global_step)
    except Exception as e:
        logging.warning("Didn't use custom metrics logger, exception '{}' occurred".format(str(e)))


def trainer_save_states_no_dump_optimizer(trainer: gluon.Trainer, fname: str):
    """
    Otherwise exact copy of `Trainer.save_states` that does not include a
    pickled optimizer instance as part of the state.  This is compatible with
    the standard `Trainer.load_states`, which will handle a state file with no
    optimizer instance (any statements involving `self._optimizer` become
    no-ops).  This is especially important when using AMP, which patches the
    optimizer at runtime with references to a specific loss scaler instance.
    Loading a stale optimizer instance causes errors.
    """
    assert trainer._optimizer is not None

    if not trainer._kv_initialized:
        trainer._init_kvstore()
    if trainer._params_to_init:
        trainer._init_params()

    if trainer._update_on_kvstore:
        assert not trainer._params_to_init, "Cannot save trainer states when some " \
                                            "parameters are not yet initialized in kvstore."
        trainer._kvstore.save_optimizer_states(fname, dump_optimizer=False)
    else:
        with open(fname, 'wb') as fout:
            fout.write(trainer._updaters[0].get_states(dump_optimizer=False))


def cleanup_params_files(output_folder: str, max_to_keep: int, checkpoint: int, best_checkpoint: int, keep_first: bool,
                         max_params_files_to_cache: int, cache_metric: str, cache_strategy: str):
    """
    Deletes oldest parameter files from a model folder.

    :param output_folder: Folder where param files are located.
    :param max_to_keep: Maximum number of files to keep, negative to keep all.
    :param checkpoint: Current checkpoint (i.e. index of last params file created).
    :param best_checkpoint: Best checkpoint. The parameter file corresponding to this checkpoint will not be deleted.
    :param keep_first: Don't delete the first checkpoint.
    :param max_params_files_to_cache: Maximum number of best param files to cache.
    :param cache_metric: Metric to determine best param files.
    :param cache_strategy: Strategy to select 'best' param files.
    """
    if max_to_keep <= 0:
        return

    # make sure we keep N best params files from .metrics file according to strategy.
    top_n: Set[int] = set()
    metrics_path = os.path.join(output_folder, C.METRICS_NAME)

    if max_params_files_to_cache > 0 and os.path.exists(metrics_path):
        maximize = C.METRIC_MAXIMIZE[cache_metric]
        points = utils.get_validation_metric_points(model_path=output_folder, metric=cache_metric)

        if cache_strategy == C.AVERAGE_BEST:
            # N best scoring points
            top = average.strategy_best(points, max_params_files_to_cache, maximize)

        elif cache_strategy == C.AVERAGE_LAST:
            # N sequential points ending with overall best
            top = average.strategy_last(points, max_params_files_to_cache, maximize)

        elif cache_strategy == C.AVERAGE_LIFESPAN:
            # Track lifespan of every "new best" point
            # Points dominated by a previous better point have lifespan 0
            top = average.strategy_lifespan(points, max_params_files_to_cache, maximize)
        else:
            raise RuntimeError("Unknown strategy, options are: %s" % C.AVERAGE_CHOICES)

        top_n = set([x[1] for x in top])

    # get rid of params files that are neither among the latest, nor among the best
    existing_files = glob.glob(os.path.join(output_folder, C.PARAMS_PREFIX + "*"))
    params_name_with_dir = os.path.join(output_folder, C.PARAMS_NAME)

    for n in range(1 if keep_first else 0, max(1, checkpoint - max_to_keep + 1)):
        if n != best_checkpoint:
            param_fname_n = params_name_with_dir % n
            if param_fname_n in existing_files and n not in top_n:
                try:
                    os.remove(param_fname_n)
                except FileNotFoundError:
                    # This can be occur on file systems with higher latency,
                    # such as distributed file systems.  While repeated
                    # occurrences of this warning may indicate a problem, seeing
                    # one or two warnings during training is usually fine.
                    logger.warning('File has already been removed: %s', param_fname_n)

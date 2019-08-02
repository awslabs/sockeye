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
Code for training
"""
import logging
import multiprocessing
import os
import pickle
import random
import shutil
import time
from math import sqrt
from typing import Callable, Dict, List, Optional, Iterable, Tuple, Union

import mxnet as mx
import numpy as np

import sockeye.multiprocessing_utils as mp_utils
from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
from . import lr_scheduler
from . import utils
from . import vocab
from . import parallel
from .config import Config
from .model import SockeyeModel

logger = logging.getLogger(__name__)


def global_norm(ndarrays: List[mx.nd.NDArray]) -> float:
    # accumulate in a list, as asscalar is blocking and this way we can run the norm calculation in parallel.
    norms = [mx.nd.square(mx.nd.norm(arr)) for arr in ndarrays if arr is not None]
    return sqrt(sum(norm.asscalar() for norm in norms))


class TrainerConfig(Config):
    def __init__(self,
                 output_dir: str,
                 early_stopping_metric: str,
                 max_params_files_to_keep: int,
                 keep_initializations: bool,
                 checkpoint_interval: int,
                 max_num_checkpoint_not_improved: int,
                 max_checkpoints: Optional[int] = None,
                 min_samples: Optional[int] = None,
                 max_samples: Optional[int] = None,
                 min_updates: Optional[int] = None,
                 max_updates: Optional[int] = None,
                 min_epochs: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 update_interval: int = 1,
                 stop_training_on_decoder_failure: bool = False) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.early_stopping_metric = early_stopping_metric
        self.max_params_files_to_keep = max_params_files_to_keep
        self.keep_initializations = keep_initializations
        self.checkpoint_interval = checkpoint_interval
        self.max_num_checkpoint_not_improved = max_num_checkpoint_not_improved
        self.max_checkpoints = max_checkpoints
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_updates = min_updates
        self.max_updates = max_updates
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.update_interval = update_interval
        self.stop_training_on_decoder_failure = stop_training_on_decoder_failure


class TrainState:
    """
    Stores the state an EarlyStoppingTrainer instance.
    """

    __slots__ = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint', 'batches',
                 'updates', 'samples', 'gradient_norm', 'gradients', 'metrics', 'start_tic',
                 'early_stopping_metric', 'best_metric', 'best_checkpoint', 'converged', 'diverged']

    def __init__(self, early_stopping_metric: str) -> None:
        self.num_not_improved = 0
        self.epoch = 0
        self.checkpoint = 0
        self.best_checkpoint = 0
        self.batches = 0
        self.updates = 0
        self.samples = 0
        self.gradient_norm = None  # type: Optional[float]
        self.gradients = {}  # type: Dict[str, List[mx.nd.NDArray]]
        # stores dicts of metric names & values for each checkpoint
        self.metrics = []  # type: List[Dict]
        self.start_tic = time.time()
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = C.METRIC_WORST[early_stopping_metric]
        self.best_checkpoint = 0
        self.converged = False
        self.diverged = False

    def save(self, fname: str):
        """
        Saves this training state to fname.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname: str) -> 'TrainState':
        """
        Loads a training state from fname.
        """
        with open(fname, "rb") as fp:
            return pickle.load(fp)


class GluonEarlyStoppingTrainer:
    def __init__(self,
                 config: TrainerConfig,
                 sockeye_model: SockeyeModel,
                 trainer: mx.gluon.Trainer,
                 loss_functions: List[loss.Loss],
                 context: List[mx.context.Context],
                 dtype: str) -> None:
        self.config = config
        self.model = sockeye_model
        self.trainer = trainer
        self.loss_functions = loss_functions
        self.context = context
        self._parallel = parallel.Parallel(len(context) if len(context) > 1 else 0,
                                           ParallelModel(sockeye_model,
                                                         loss_functions,
                                                         rescale_factor=self.config.update_interval))
        self.dtype = dtype
        self.state = None  # type: Optional[TrainState]
        self._speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)

    def fit(self,
            train_iter: data_io.BaseParallelSampleIter,
            validation_iter: data_io.BaseParallelSampleIter,
            ck_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None):
        logger.info("Early stopping by optimizing '%s'", self.config.early_stopping_metric)

        if self.config.early_stopping_metric in C.METRICS_REQUIRING_DECODER:
            utils.check_condition(ck_decoder is not None,
                                  "%s requires CheckpointDecoder" % self.config.early_stopping_metric)

        resume_training = os.path.exists(self.training_state_dirname)
        if resume_training:
            logger.info("Found partial training in '%s'. Resuming from saved state.", self.training_state_dirname)
            self._load_training_state(train_iter)
        else:
            self.state = TrainState(self.config.early_stopping_metric)
            self.model.save_config(self.config.output_dir)
            self.model.save_version(self.config.output_dir)
            #~ self._save_training_state(train_iter)
            #self._save_trainer_states(self.best_optimizer_states_fname) # not saving due to deferred initialization
            logger.info("Training started.")

        # TODO: CheckpointDecoder
        tic = time.time()

        if self.config.max_checkpoints is not None:
            self.config.max_updates = self.state.updates + self.config.max_checkpoints * self.config.checkpoint_interval
            logger.info("Resetting max_updates to %d + %d * %d = %d in order to implement stopping after (an additional) %d checkpoints.",
                        self.state.updates,
                        self.config.max_checkpoints,
                        self.config.checkpoint_interval,
                        self.config.max_updates,
                        self.config.max_checkpoints)

        while True:
            if self.config.max_epochs is not None and self.state.epoch == self.config.max_epochs:
                logger.info("Maximum # of epochs (%s) reached.", self.config.max_epochs)
                break

            if self.config.max_updates is not None and self.state.updates == self.config.max_updates:
                logger.info("Maximum # of updates (%s) reached.", self.config.max_updates)
                break

            if self.config.max_samples is not None and self.state.samples >= self.config.max_samples:
                logger.info("Maximum # of samples (%s) reached", self.config.max_samples)
                break

            self._step(batch=train_iter.next())

            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()

            if self.state.updates > 0 and self.state.batches % (
                    self.config.checkpoint_interval * self.config.update_interval) == 0:
                time_cost = time.time() - tic
                self.state.checkpoint += 1

                # (1) save parameters and evaluate on validation data
                self._save_params()

                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f",
                            self.state.checkpoint, self.state.updates, self.state.epoch,
                            self.state.samples, time_cost, self.config.checkpoint_interval / time_cost)
                logger.info('Checkpoint [%d]\t%s',
                            self.state.checkpoint, "\t".join("Train-%s" % str(lf.metric) for lf in self.loss_functions))

                val_metrics = self._evaluate(validation_iter)

                mx.nd.waitall()

                has_improved = self._determine_improvement(val_metrics)
                self.state.converged = self._determine_convergence()
                self.state.diverged = self._determine_divergence(val_metrics)
                self._adjust_learning_rate(has_improved)
                if has_improved:
                    self._update_best_params()
                    self._save_trainer_states(self.best_optimizer_states_fname)
                self._save_training_state(train_iter)

                if self.state.converged or self.state.diverged:
                    break

                self._write_metrics_file(train_metrics=[l.metric for l in self.loss_functions], val_metrics=val_metrics)
                for lf in self.loss_functions:
                    lf.metric.reset()

                tic = time.time()

        logger.info("Training finished%s. Best checkpoint: %d. Best validation %s: %.6f",
                    ", can be continued later" if not self.state.converged else "",
                    self.state.best_checkpoint, self.state.early_stopping_metric, self.state.best_metric)

        self._cleanup(keep_training_state=not self.state.converged and not self.state.diverged)
        return self.state

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
            tuple(mx.nd.add_n(*(s.as_in_context(mx.cpu()) for s in shard)) for shard in zip(*outs)) for outs in
            sharded_outputs_per_loss_function]
        return output_per_loss_function

    def _step(self, batch: data_io.Batch):
        self.state.batches += 1
        loss_outputs = self._forward_backward(batch)
        if self.config.update_interval == 1 or self.state.batches % self.config.update_interval == 0:
            self.trainer.step(1)  # 1: We already normalized
            if self.config.update_interval > 1:
                self.model.collect_params().zero_grad()
            self.state.updates += 1

        self.state.samples += batch.samples
        for loss_func, (loss_value, num_samples) in zip(self.loss_functions, loss_outputs):
            loss_func.metric.update(loss_value.asscalar(), num_samples.asscalar())
        self._speedometer(self.state.epoch, self.state.batches,
                          self.state.updates, batch.samples, batch.tokens, (lf.metric for lf in self.loss_functions))

    def _evaluate(self, data_iter) -> List[loss.LossMetric]:
        """
        Computes loss(es) on validation data and returns their metrics.
        :param data_iter: Validation data iterator.
        :return: List of validation metrics, same order as self.loss_functions.
        """
        data_iter.reset()
        val_metrics = [lf.create_metric() for lf in self.loss_functions]
        for batch in data_iter:
            batch = batch.split_and_load(ctx=self.context)
            sharded_loss_outputs = []  # type: List[List[Tuple[mx.nd.NDArray, mx.nd.NDArray]]]
            for inputs, labels in batch.shards():
                outputs = self.model(*inputs)  # type: Dict[str, mx.nd.NDArray]
                loss_outputs = [loss_function(outputs, labels) for loss_function in self.loss_functions]
                sharded_loss_outputs.append(loss_outputs)

            # repack outputs into a list of loss_values (length = number of shards) for each loss function
            sharded_loss_outputs_per_loss_function = list(zip(*sharded_loss_outputs))
            # sum loss values and number of samples for each loss function
            output_per_loss_function = [tuple(mx.nd.add_n(*shard) for shard in zip(*outs)) for outs in
                                        sharded_loss_outputs_per_loss_function]
            # update validation metrics for batch
            for loss_metric, (loss_value, num_samples) in zip(val_metrics, output_per_loss_function):
                loss_metric.update(loss_value.asscalar(), num_samples.asscalar())

        logger.info('Checkpoint [%d]\t%s',
                    self.state.checkpoint, "\t".join("Validation-%s" % str(lm) for lm in val_metrics))

        # TODO CheckpointDecoder

        return val_metrics

    def _determine_improvement(self, val_metrics: List[loss.LossMetric]) -> bool:
        """
        Determines whether early stopping metric on validation data improved and updates best value and checkpoint in
        the state.
        :param val_metrics: Validation metrics.
        :return: Whether model has improved on held-out data since last checkpoint.
        """
        for val_metric in val_metrics:
            if val_metric.name == self.config.early_stopping_metric:
                value = val_metric.get()
                if utils.metric_value_is_better(value,
                                                self.state.best_metric,
                                                self.config.early_stopping_metric):
                    logger.info("Validation-%s improved to %f (delta=%f).", self.config.early_stopping_metric,
                                value, abs(value - self.state.best_metric))
                    self.state.best_metric = value
                    self.state.best_checkpoint = self.state.checkpoint
                    self.state.num_not_improved = 0
                    return True

        self.state.num_not_improved += 1
        logger.info("Validation-%s has not improved for %d checkpoints, best so far: %f",
                    self.config.early_stopping_metric, self.state.num_not_improved, self.state.best_metric)
        return False

    def _determine_convergence(self) -> bool:
        """
        True if model has converged w.r.t early stopping criteria (patience).
        """
        if 0 <= self.config.max_num_checkpoint_not_improved <= self.state.num_not_improved:
            logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                        self.config.max_num_checkpoint_not_improved, self.state.num_not_improved)
            return True

        if self.config.min_epochs is not None and self.state.epoch < self.config.min_epochs:
            logger.info("Minimum number of epochs (%d) not reached yet: %d",
                        self.config.min_epochs, self.state.epoch)

        if self.config.min_updates is not None and self.state.updates < self.config.min_updates:
            logger.info("Minimum number of updates (%d) not reached yet: %d",
                        self.config.min_updates, self.state.updates)

        if self.config.min_samples is not None and self.state.samples < self.config.min_samples:
            logger.info("Minimum number of samples (%d) not reached yet: %d",
                        self.config.min_samples, self.state.samples)
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
                self._load_trainer_states(self.best_optimizer_states_fname)
                # state loading replaces the lr_scheduler instance which then contains the old learning rate,
                # overwriting here. TODO: make this better...
                self.trainer.optimizer.lr_scheduler.lr = adjusted_lr

    def _write_metrics_file(self, train_metrics: List[loss.LossMetric], val_metrics: List[loss.LossMetric]):
        """
        Updates metrics for current checkpoint.
        Writes all metrics to the metrics file and optionally logs to tensorboard.
        """
        data = {"epoch": self.state.epoch,
                "learning-rate": self.trainer.optimizer.lr_scheduler.lr,
                "gradient-norm": self.state.gradient_norm,
                "time-elapsed": time.time() - self.state.start_tic}
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

        # TODO: Tensorboard logging
        # tf_metrics = data.copy()
        # tf_metrics.update({"%s_grad" % n: v for n, v in self.state.gradients.items()})
        # tf_metrics.update(self.model.params)
        #self.tflogger.log_metrics(metrics=tf_metrics, checkpoint=self.state.checkpoint)

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
        utils.cleanup_params_files(self.config.output_dir, self.config.max_params_files_to_keep, self.state.checkpoint,
                                   self.state.best_checkpoint, self.config.keep_initializations)

    def _save_trainer_states(self, fname):
        self.trainer.save_states(fname)
        logger.info('Saved optimizer states to "%s"', fname)

    def _load_trainer_states(self, fname):
        self.trainer.load_states(fname)
        logger.info('Loaded optimizer states from "%s"', fname)

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
        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp)

        # (5) Training state
        self.state.save(os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # trainer.save_states also pickles optimizers and their lr schedulers.
        # # (6) Learning rate scheduler
        # with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
        #     pickle.dump(self.trainer.optimizer.lr_scheduler, fp)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(self.config.output_dir, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(self.training_state_dirname):
            os.rename(self.training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, self.training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            shutil.rmtree(delete_training_state_dirname)

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
        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(self.training_state_dirname, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            np.random.set_state(pickle.load(fp))

        # (5) Training state
        self.state = TrainState.load(os.path.join(self.training_state_dirname, C.TRAINING_STATE_NAME))

        # trainer.save_states also pickles optimizers and their lr schedulers. additional loading not required
        # # (6) Learning rate scheduler
        # with open(os.path.join(self.training_state_dirname, C.SCHEDULER_STATE_NAME), "rb") as fp:
        #     self.trainer.optimizer.lr_scheduler = pickle.load(fp)

    def _cleanup(self, keep_training_state=False):
        """
        Cleans parameter files, training state directory and waits for remaining decoding processes.
        """
        utils.cleanup_params_files(self.config.output_dir, self.config.max_params_files_to_keep,
                                   self.state.checkpoint, self.state.best_checkpoint, self.config.keep_initializations)
        # if process_manager is not None:
        #     result = process_manager.collect_results()
        #     if result is not None:
        #         decoded_checkpoint, decoder_metrics = result
        #         self.state.metrics[decoded_checkpoint - 1].update(decoder_metrics)
        #         self.tflogger.log_metrics(decoder_metrics, decoded_checkpoint)
        #         utils.write_metrics_file(self.state.metrics, self.metrics_fname)
        #         self.state.save(os.path.join(self.training_state_dirname, C.TRAINING_STATE_NAME))

        if not keep_training_state:
            if os.path.exists(self.training_state_dirname):
                shutil.rmtree(self.training_state_dirname)
            if os.path.exists(self.best_optimizer_states_fname):
                os.remove(self.best_optimizer_states_fname)

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


class ParallelModel(parallel.Parallelizable):

    def __init__(self, model: Callable, loss_functions: List[loss.Loss], rescale_factor: float) -> None:
        self.model = model
        self.loss_functions = loss_functions
        self.rescale_factor = rescale_factor

    def forward_backward(self, shard: Tuple) -> List[Tuple[mx.nd.NDArray, mx.nd.NDArray]]:
        """
        Applies forward-backward pass for a single shard of a batch (data-parallel training).
        """
        inputs, labels = shard
        with mx.autograd.record():
            outputs = self.model(*inputs)  # type: Dict[str, mx.nd.NDArray]
            loss_outputs = [loss_function(outputs, labels) for loss_function in self.loss_functions]
            loss_values = (v for v, _ in loss_outputs)
            sum_losses = mx.nd.add_n(*loss_values) / self.rescale_factor
            # Note: rescaling works for all loss functions except softmax output, which requires grad_scale to be set
            # directly in the op call (see loss function implementation).
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
            self.sw = mxboard.SummaryWriter(logdir=self.logdir, flush_secs=60, verbose=False)
        except ImportError:
            logger.info("mxboard not found. Consider 'pip install mxboard' to log events to Tensorboard.")
            self.sw = None

    def log_metrics(self, metrics: Dict[str, Union[float, int, mx.nd.NDArray]], checkpoint: int):
        if self.sw is None:
            return

        for name, value in metrics.items():
            if isinstance(value, mx.nd.NDArray):
                if mx.nd.contrib.isfinite(value).sum().asscalar() == value.size:
                    self.sw.add_histogram(tag=name, values=value, bins=100, global_step=checkpoint)
                else:
                    logger.warning("Histogram of %s not logged to tensorboard because of infinite data.")
            else:
                self.sw.add_scalar(tag=name, value=value, global_step=checkpoint)

    def log_graph(self, symbol: mx.sym.Symbol):
        if self.sw is None:
            return
        self.sw.add_graph(symbol)

    def log_source_embedding(self, embedding: mx.nd.NDArray, checkpoint: int):
        if self.sw is None or self.source_labels is None:
            return
        self.sw.add_embedding(tag="source", embedding=embedding, labels=self.source_labels, global_step=checkpoint)

    def log_target_embedding(self, embedding: mx.nd.NDArray, checkpoint: int):
        if self.sw is None or self.target_labels is None:
            return
        self.sw.add_embedding(tag="target", embedding=embedding, labels=self.target_labels, global_step=checkpoint)

    def log_output_embedding(self, embedding: mx.nd.NDArray, checkpoint: int):
        if self.sw is None or self.target_labels is None:
            return
        self.sw.add_embedding(tag="output", embedding=embedding, labels=self.target_labels, global_step=checkpoint)


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
        self.msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec %.2f tokens/sec %.2f updates/sec'

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
                update_interval = batches / updates
                updates_per_sec = self.frequency / update_interval / toc
                samples_per_sec = self.samples / toc
                tokens_per_sec = self.tokens / toc
                self.samples = 0
                self.tokens = 0

                if metrics is not None:
                    metric_values = []  # type: List[Tuple[str, float]]
                    for metric in metrics:
                        metric_values.append((metric.name, metric.get()))
                        if self.auto_reset:
                            metric.reset()
                    logger.info(self.msg + '\t%s=%f' * len(metric_values),
                                epoch, count, samples_per_sec, tokens_per_sec, updates_per_sec, *sum(metric_values, ()))

                else:
                    logger.info(self.msg, epoch, count, samples_per_sec, tokens_per_sec, updates_per_sec)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class DecoderProcessManager(object):
    """
    Thin wrapper around a CheckpointDecoder instance to start non-blocking decodes and collect the results.

    :param output_folder: Folder where decoder outputs are written to.
    :param decoder: CheckpointDecoder instance.
    """

    def __init__(self,
                 output_folder: str,
                 decoder: checkpoint_decoder.CheckpointDecoder) -> None:
        self.output_folder = output_folder
        self.decoder = decoder
        self.ctx = mp_utils.get_context()  # type: ignore
        self.decoder_metric_queue = self.ctx.Queue()
        self.decoder_process = None  # type: Optional[multiprocessing.Process]
        self._any_process_died = False
        self._results_pending = False

    def start_decoder(self, checkpoint: int):
        """
        Starts a new CheckpointDecoder process and returns. No other process may exist.

        :param checkpoint: The checkpoint to decode.
        """
        assert self.decoder_process is None
        output_name = os.path.join(self.output_folder, C.DECODE_OUT_NAME % checkpoint)
        self.decoder_process = self.ctx.Process(target=_decode_and_evaluate,
                                                args=(self.decoder, checkpoint, output_name, self.decoder_metric_queue))
        self.decoder_process.name = 'Decoder-%d' % checkpoint
        logger.info("Starting process: %s", self.decoder_process.name)
        self.decoder_process.start()
        self._results_pending = True

    def collect_results(self) -> Optional[Tuple[int, Dict[str, float]]]:
        """
        Returns the decoded checkpoint and the decoder metrics or None if the queue is empty.
        """
        self.wait_to_finish()
        if self.decoder_metric_queue.empty():
            if self._results_pending:
                self._any_process_died = True
            self._results_pending = False
            return None
        decoded_checkpoint, decoder_metrics = self.decoder_metric_queue.get()
        assert self.decoder_metric_queue.empty()
        self._results_pending = False
        logger.info("Decoder-%d finished: %s", decoded_checkpoint, decoder_metrics)
        return decoded_checkpoint, decoder_metrics

    def wait_to_finish(self):
        if self.decoder_process is None:
            return
        if not self.decoder_process.is_alive():
            self.decoder_process = None
            return
        name = self.decoder_process.name
        logger.warning("Waiting for process %s to finish.", name)
        wait_start = time.time()
        self.decoder_process.join()
        self.decoder_process = None
        wait_time = int(time.time() - wait_start)
        logger.warning("Had to wait %d seconds for the Checkpoint %s to finish. Consider increasing the "
                       "checkpoint interval (updates between checkpoints, see %s) or reducing the size of the "
                       "validation samples that are decoded (see %s)." % (wait_time, name,
                                                                          C.TRAIN_ARGS_CHECKPOINT_INTERVAL,
                                                                          C.TRAIN_ARGS_MONITOR_BLEU))

    @property
    def any_process_died(self):
        """ Returns true if any decoder process exited and did not provide a result. """
        return self._any_process_died

    def update_process_died_status(self):
        """ Update the flag indicating whether any process exited and did not provide a result. """

        # There is a result pending, the process is no longer alive, yet there is no result in the queue
        # This means the decoder process has not succesfully produced metrics
        queue_should_hold_result = self._results_pending and self.decoder_process is not None and not self.decoder_process.is_alive()
        if queue_should_hold_result and self.decoder_metric_queue.empty():
            self._any_process_died = True


def _decode_and_evaluate(decoder: checkpoint_decoder.CheckpointDecoder,
                         checkpoint: int,
                         output_name: str,
                         queue: multiprocessing.Queue):
    """
    Decodes and evaluates using given checkpoint_decoder and puts result in the queue,
    indexed by the checkpoint.
    """
    metrics = decoder.decode_and_evaluate(checkpoint, output_name)
    queue.put((checkpoint, metrics))

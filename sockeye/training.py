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
import glob
import logging
import os
import pickle
import random
import shutil
import time
from functools import reduce
from typing import Any, AnyStr, Dict, List, Optional, Tuple, Union

import mxnet as mx
import numpy as np
from math import sqrt

from . import callback
from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
from . import lr_scheduler
from . import model
from . import utils
from .optimizers import BatchState, CheckpointState, SockeyeOptimizer, OptimizerConfig

logger = logging.getLogger(__name__)


class TrainingModel(model.SockeyeModel):
    """
    Defines an Encoder/Decoder model (with attention).
    RNN configuration (number of hidden units, number of layers, cell type)
    is shared between encoder & decoder.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param train_iter: The iterator over the training data.
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 output_dir: str,
                 train_iter: data_io.BaseParallelSampleIter,
                 bucketing: bool,
                 gradient_compression_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.context = context
        self.output_dir = output_dir
        self._bucketing = bucketing
        self._gradient_compression_params = gradient_compression_params

        self._build_model_components()
        self._initialize(train_iter)

        self._monitor = None  # type: Optional[mx.monitor.Monitor]

    def _initialize(self, train_iter: data_io.BaseParallelSampleIter):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        self.model_loss = loss.get_loss(self.config.config_loss)

        data_names = [x[0] for x in train_iter.provide_data]
        label_names = [x[0] for x in train_iter.provide_label]

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, decoder_depth)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(target_decoded)

            probs = self.model_loss.get_loss(logits, labels)

            return mx.sym.Group(probs), data_names, label_names

        if self._bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=train_iter.default_bucket_key,
                                                 context=self.context,
                                                 compression_params=self._gradient_compression_params)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_target)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            self.module = mx.mod.Module(symbol=symbol,
                                        data_names=data_names,
                                        label_names=label_names,
                                        logger=logger,
                                        context=self.context,
                                        compression_params=self._gradient_compression_params)

        self.module.bind(data_shapes=train_iter.provide_data,
                         label_shapes=train_iter.provide_label,
                         for_training=True,
                         force_rebind=True,
                         grad_req='write')

        self.module.symbol.save(os.path.join(self.output_dir, C.SYMBOL_NAME))

        self.save_version(self.output_dir)
        self.save_config(self.output_dir)

    def run_forward_backward(self, batch: mx.io.DataBatch, metric: mx.metric.EvalMetric):
        """
        Runs forward/backward pass and updates training metric(s).
        """
        self.module.forward_backward(batch)
        self.module.update_metric(metric, batch.label)

    def update(self):
        """
        Update model parameters
        """
        self.module.update()

    def get_global_gradient_norm(self) -> float:
        """
        Returns global gradient norm.
        """
        # average norm across executors:
        exec_norms = [global_norm([arr for arr in exe.grad_arrays if arr is not None]) for exe in self.executors]
        norm_val = sum(exec_norms) / float(len(exec_norms))
        norm_val *= self.optimizer.rescale_grad
        return norm_val

    def rescale_gradients(self, scale: float):
        """
        Rescales gradient arrays by scale.
        """
        for exe in self.executors:
            for arr in exe.grad_arrays:
                if arr is None:
                    continue
                arr *= scale

    def prepare_batch(self, batch: mx.io.DataBatch):
        """
        Pre-fetches the next mini-batch.
        """
        self.module.prepare(batch)

    def evaluate(self, eval_iter: data_io.BaseParallelSampleIter, eval_metric: mx.metric.EvalMetric):
        """
        Resets and recomputes evaluation metric on given data iterator.
        """
        for eval_batch in eval_iter:
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(eval_metric, eval_batch.label)

    @property
    def current_module(self) -> mx.module.Module:
        # As the BucketingModule does not expose all methods of the underlying Module we need to directly access
        # the currently active module, when we use bucketing.
        return self.module._curr_module if self._bucketing else self.module

    @property
    def executors(self):
        return self.current_module._exec_group.execs

    @property
    def loss(self) -> loss.Loss:
        return self.model_loss

    @property
    def optimizer(self) -> Union[mx.optimizer.Optimizer, SockeyeOptimizer]:
        # TODO: Push update to MXNet to expose the optimizer (Module should have a get_optimizer method)
        return self.current_module._optimizer

    def initialize_optimizer(self, config: OptimizerConfig):
        self.module.init_optimizer(kvstore=config.kvstore,
                                   optimizer=config.name,
                                   optimizer_params=config.params,
                                   force_init=True)  # force init for training resumption use case

    def save_optimizer_states(self, fname: str):
        """
        Saves optimizer states to a file.

        :param fname: File name to save optimizer states to.
        """
        self.current_module.save_optimizer_states(fname)

    def load_optimizer_states(self, fname: str):
        """
        Loads optimizer states from file.

        :param fname: File name to load optimizer states from.
        """
        self.current_module.load_optimizer_states(fname)

    def initialize_parameters(self, initializer: mx.init.Initializer, allow_missing_params: bool):
        self.module.init_params(initializer=initializer,
                                arg_params=self.params,
                                aux_params=self.aux_params,
                                allow_missing=allow_missing_params,
                                force_init=False)

    def log_parameters(self):
        """
        Logs information about model parameters.
        """
        arg_params, aux_params = self.module.get_params()
        total_parameters = 0
        info = []  # type: List[str]
        for name, array in sorted(arg_params.items()):
            info.append("%s: %s" % (name, array.shape))
            total_parameters += reduce(lambda x, y: x * y, array.shape)
        logger.info("Model parameters: %s", ", ".join(info))
        logger.info("Total # of parameters: %d", total_parameters)

    def save_params_to_file(self, fname: str):
        """
        Synchronizes parameters across devices, saves the parameters to disk, and updates self.params
        and self.aux_params.

        :param fname: Filename to write parameters to.
        """
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        self.aux_params = aux_params
        super().save_params_to_file(fname)

    def load_params_from_file(self, fname: str):
        super().load_params_from_file(fname)  # sets self.params & self.aux_params
        self.module.set_params(arg_params=self.params, aux_params=self.aux_params)

    def install_monitor(self, monitor_pattern: str, monitor_stat_func_name: str):
        self._monitor = mx.monitor.Monitor(interval=C.MEASURE_SPEED_EVERY,
                                           stat_func=C.MONITOR_STAT_FUNCS.get(monitor_stat_func_name),
                                           pattern=monitor_pattern,
                                           sort=True)
        self.module.install_monitor(self._monitor)
        logger.info("Installed MXNet monitor; pattern='%s'; statistics_func='%s'",
                    monitor_pattern, monitor_stat_func_name)

    @property
    def monitor(self) -> Optional[mx.monitor.Monitor]:
        return self._monitor


def global_norm(ndarrays: List[mx.nd.NDArray]) -> float:
    # accumulate in a list, as asscalar is blocking and this way we can run the norm calculation in parallel.
    norms = [mx.nd.square(mx.nd.norm(arr)) for arr in ndarrays if arr is not None]
    return sqrt(sum(norm.asscalar() for norm in norms))


class TrainState:
    """
    Stores the state of the training process. These are the variables that will
    be stored to disk when resuming training.
    """

    __slots__ = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint',
                 'updates', 'samples', 'gradient_norm', 'metrics', 'start_tic', 'best_metric', 'best_checkpoint']

    def __init__(self, early_stopping_metric: str) -> None:
        self.num_not_improved = 0
        self.epoch = 0
        self.checkpoint = 0
        self.best_checkpoint = 0
        self.updates = 0
        self.samples = 0
        self.gradient_norm = None  # type: Optional[float]
        # stores dicts of metric names & values for each checkpoint
        self.metrics = []  # type: List[Dict]
        self.start_tic = time.time()
        self.best_metric = C.METRIC_WORST[early_stopping_metric]
        self.best_checkpoint = 0

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


class Trainer:
    """

    :param model: Training model.
    :param optimizer_config: The optimizer configuration.
    :param output_folder: The folder in which all model artifacts will be stored in (parameters, checkpoints, etc.).
    :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
    :param kvstore: The MXNet kvstore used.
    :param log_to_tensorboard: If True write training and evaluation logs to tensorboard event files.
    """

    def __init__(self,
                 model: TrainingModel,
                 optimizer_config: OptimizerConfig,
                 max_params_files_to_keep: int,
                 log_to_tensorboard: bool = True) -> None:
        self.model = model
        self.optimizer_config = optimizer_config
        self.max_params_files_to_keep = max_params_files_to_keep
        self.tensorboard_logger = None  # type: Optional[TensorboardLogger]
        if log_to_tensorboard:
            self.tensorboard_logger = TensorboardLogger(log_dir=os.path.join(model.output_dir, C.TENSORBOARD_NAME))
        self.state = TrainState(C.PERPLEXITY)

    def fit(self,
            train_iter: data_io.BaseParallelSampleIter,
            val_iter: data_io.BaseParallelSampleIter,
            metrics: List[AnyStr],
            allow_missing_parameters: bool,
            max_updates: Optional[int],
            checkpoint_frequency: int,
            early_stopping_metric: str = "perplexity",
            max_num_not_improved: int = 3,
            min_num_epochs: Optional[int] = None,
            max_num_epochs: Optional[int] = None,
            cp_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None,
            mxmonitor_pattern: Optional[str] = None,
            mxmonitor_stat_func: Optional[str] = None,
            lr_decay_param_reset: bool = False,
            lr_decay_opt_states_reset: str = C.LR_DECAY_OPT_STATES_RESET_OFF,
            existing_parameters: Optional[str] = None):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator.
        :param val_iter: The validation data iterator.

        :param metrics: The metrics that will be evaluated during training.

        :param allow_missing_parameters: Allow missing parameters when initializing model parameters from file.
        :param max_updates: Optional maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing in number of updates.
        :param early_stopping_metric: The metric that is tracked for early stopping.

        :param max_num_not_improved: Stop training if the optimized_metric does not improve for this many checkpoints,
               -1: do not use early stopping.
        :param min_num_epochs: Optional minimum number of epochs to train, even if validation scores did not improve.
        :param max_num_epochs: Optional maximum number of epochs to train.
        :param decode_and_evaluate: Monitor BLEU during training (0: off, >=0: the number of sentences to decode for BLEU
               evaluation, -1: decode the full validation set.).
        :param cp_decoder: Optional CheckpointDecoder instance to decode and compute BLEU at avery checkpoint.

        :param mxmonitor_pattern: Optional pattern to match to monitor weights/gradients/outputs
               with MXNet's monitor. Default is None which means no monitoring.
        :param mxmonitor_stat_func: Choice of statistics function to run on monitored weights/gradients/outputs
               when using MXNEt's monitor.
        :param lr_decay_param_reset: Reset parameters to previous best after learning rate decay.
        :param lr_decay_opt_states_reset: How to reset optimizer states after learning rate decay.
        :return: Best score on validation data observed during training.
        """
        if 'dist' in self.optimizer_config.kvstore:
            self._check_dist_kvstore_requirements(lr_decay_opt_states_reset, lr_decay_param_reset, self.optimizer_config.name)

        utils.check_condition(self.optimizer_config.gradient_clipping_type in C.GRADIENT_CLIPPING_TYPES,
                              "Unknown gradient clipping type %s" % self.optimizer_config.gradient_clipping_type)

        utils.check_condition(early_stopping_metric in C.METRICS,
                              "Unsupported early-stopping metric: %s" % early_stopping_metric)
        if early_stopping_metric == C.BLEU:
            utils.check_condition(cp_decoder is not None, "%s requires CheckpointDecoder" % C.BLEU)
        logger.info("Early stopping by optimizing '%s'", early_stopping_metric)

        process_manager = callback.DecoderProcessManager(self.model.output_dir, cp_decoder=cp_decoder)

        self._initialize_parameters(existing_parameters, allow_missing_parameters)
        self._initialize_optimizer()

        resume_training = os.path.exists(self.training_state_dirname)
        if resume_training:
            logger.info("Found partial training in '%s'. Resuming from saved state.", self.training_state_dirname)
            utils.check_condition('dist' not in self.optimizer_config.kvstore,
                                  "Training continuation not supported with distributed training.")
            self._load_training_state(train_iter)
        else:
            logger.info("Training started.")
            self.state = TrainState(early_stopping_metric)
            self.state.best_metric = C.METRIC_WORST[early_stopping_metric]
            self._save_params()
            self._update_best_params_link()
            self._save_training_state(train_iter)
            self._update_best_optimizer_states(lr_decay_opt_states_reset)

        metric_train, metric_val, metric_loss = self._create_metrics(metrics, self.model.optimizer, self.model.loss)

        if mxmonitor_pattern is not None:
            self.model.install_monitor(mxmonitor_pattern, mxmonitor_stat_func)

        speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        tic = time.time()

        next_data_batch = train_iter.next()
        while True:

            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()
                if max_num_epochs is not None and self.state.epoch == max_num_epochs:
                    logger.info("Maximum # of epochs (%s) reached.", max_num_epochs)
                    break

            if max_updates is not None and self.state.updates == max_updates:
                logger.info("Maximum # of updates (%s) reached.", max_updates)
                break

            ######
            # STEP
            ######
            batch = next_data_batch
            self._step(self.model, batch, checkpoint_frequency, metric_train, metric_loss)
            if train_iter.iter_next():
                next_data_batch = train_iter.next()
                self.model.prepare_batch(next_data_batch)
            batch_num_samples = batch.data[0].shape[0]
            batch_num_tokens = batch.data[0].shape[1] * batch_num_samples
            self.state.updates += 1
            self.state.samples += batch_num_samples
            speedometer(self.state.epoch, self.state.updates, batch_num_samples, batch_num_tokens, metric_train)

            ############
            # CHECKPOINT
            ############
            if self.state.updates > 0 and self.state.updates % checkpoint_frequency == 0:
                time_cost = time.time() - tic
                self.state.checkpoint += 1
                self._save_params()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f",
                            self.state.checkpoint, self.state.updates, self.state.epoch,
                            self.state.samples, time_cost, checkpoint_frequency / time_cost)
                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', self.state.checkpoint, name, val)
                self._evaluate(val_iter, metric_val)
                for name, val in metric_val.get_name_value():
                    logger.info('Checkpoint [%d]\tValidation-%s=%f', self.state.checkpoint, name, val)
                self._add_metrics_to_state(metric_train, metric_val, process_manager)
                metric_train.reset()

                def _is_better(value: float, other: float) -> bool:
                    if C.METRIC_MAXIMIZE[early_stopping_metric]:
                        return value > other
                    else:
                        return value < other

                has_improved = False
                previous_best = self.state.best_metric
                for checkpoint, metric_dict in enumerate(self.state.metrics, 1):
                    value = metric_dict.get("%s-val" % early_stopping_metric, self.state.best_metric)
                    if _is_better(value, previous_best):
                        self.state.best_metric = value
                        self.state.best_checkpoint = checkpoint
                        has_improved = True

                if has_improved:
                    self._update_best_params_link()
                    self._update_best_optimizer_states(lr_decay_opt_states_reset)
                    self.state.num_not_improved = 0
                    logger.info("Validation-%s improved to %f (delta=%f).", early_stopping_metric,
                                self.state.best_metric, abs(self.state.best_metric - previous_best))
                else:
                    logger.info("Validation-%s has not improved for %d checkpoints, best so far: %f",
                                early_stopping_metric, self.state.num_not_improved, self.state.best_metric)
                    self.state.num_not_improved += 1

                # If using an extended optimizer, provide extra state information about the current checkpoint
                # Loss: optimized metric
                if metric_loss is not None and isinstance(self.model.optimizer, SockeyeOptimizer):
                    m_val = 0
                    for name, val in metric_val.get_name_value():
                        if name == early_stopping_metric:
                            m_val = val
                    checkpoint_state = CheckpointState(checkpoint=self.state.checkpoint, metric_val=m_val)
                    self.model.optimizer.pre_update_checkpoint(checkpoint_state)

                self._adjust_learning_rate(has_improved, lr_decay_param_reset, lr_decay_opt_states_reset)
                self._save_training_state(train_iter)

                if 0 <= max_num_not_improved <= self.state.num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, self.state.num_not_improved)
                    stop_fit = True

                    if min_num_epochs is not None and self.state.epoch < min_num_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_num_epochs, self.state.epoch)
                        stop_fit = False

                    if stop_fit:
                        break

                tic = time.time()

        self._cleanup(lr_decay_opt_states_reset)
        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.state.best_checkpoint, early_stopping_metric, self.state.best_metric)
        return self.state.best_metric

    def _step(self,
              model: TrainingModel,
              batch: mx.io.DataBatch,
              checkpoint_frequency: int,
              metric_train: mx.metric.EvalMetric,
              metric_loss: Optional[mx.metric.EvalMetric] = None):

        if model.monitor is not None:
            model.monitor.tic()

        ####################
        # Forward & Backward
        ####################
        model.run_forward_backward(batch, metric_train)

        ####################
        # Gradient rescaling
        ####################
        gradient_norm = None
        if self.state.updates > 0 and (self.state.updates + 1) % checkpoint_frequency == 0:
            # compute values for logging to metrics (before rescaling...)
            gradient_norm = self.state.gradient_norm = model.get_global_gradient_norm()

        # note: C.GRADIENT_CLIPPING_TYPE_ABS is handled by the mxnet optimizer directly
        if self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_NORM:
            if gradient_norm is None:
                gradient_norm = model.get_global_gradient_norm()
            # clip gradients
            if gradient_norm > self.optimizer_config.gradient_clipping_threshold:
                ratio = self.optimizer_config.gradient_clipping_threshold / gradient_norm
                model.rescale_gradients(ratio)

        # If using an extended optimizer, provide extra state information about the current batch
        optimizer = model.optimizer
        if metric_loss is not None and isinstance(optimizer, SockeyeOptimizer):
            # Loss for this batch
            metric_loss.reset()
            metric_loss.update(batch.label, model.module.get_outputs())
            [(_, m_val)] = metric_loss.get_name_value()
            batch_state = BatchState(metric_val=m_val)
            optimizer.pre_update_batch(batch_state)

        ########
        # UPDATE
        ########
        model.update()

        if model.monitor is not None:
            results = model.monitor.toc()
            if results:
                for _, k, v in results:
                    logger.info('Monitor: Batch [{:d}] {:s} {:s}'.format(self.state.updates, k, v))

    def _evaluate(self,
                  val_iter: data_io.BaseParallelSampleIter,
                  val_metric: mx.metric.EvalMetric):
        val_iter.reset()
        val_metric.reset()
        self.model.evaluate(val_iter, val_metric)

    def _add_metrics_to_state(self, metric_train: mx.metric.EvalMetric, metric_val: mx.metric.EvalMetric, process_manager: Optional[callback.DecoderProcessManager] = None):
        checkpoint_metrics = {"epoch": self.state.epoch,
                              "learning-rate": self.model.optimizer.learning_rate,
                              "gradient-norm": self.state.gradient_norm,
                              "time-elapsed": time.time() - self.state.start_tic}
        gpu_memory_usage = utils.get_gpu_memory_usage(self.model.context)
        if gpu_memory_usage is not None:
            # total gpu memory used in MB
            checkpoint_metrics['used-gpu-memory'] = sum(v[0] for v in gpu_memory_usage.values())

        for name, value in metric_train.get_name_value():
            checkpoint_metrics["%s-train" % name] = value
        for name, value in metric_val.get_name_value():
            checkpoint_metrics["%s-val" % name] = value

        if process_manager is not None:
            process_manager.wait_to_finish()
            result = process_manager.collect_results()
            if result is not None:
                decoded_checkpoint, decoder_metrics = result
                self.state.metrics[decoded_checkpoint - 1].update(decoder_metrics)
                if self.tensorboard_logger is not None:
                    self.tensorboard_logger.log_metrics(decoder_metrics, decoded_checkpoint)
            process_manager.spawn(self.state.checkpoint)

        self.state.metrics.append(checkpoint_metrics)
        utils.write_metrics_file(self.state.metrics, self.metrics_fname)
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_metrics(checkpoint_metrics, self.state.checkpoint)

    def _cleanup(self, lr_decay_opt_states_reset, process_manager: Optional[callback.DecoderProcessManager] = None):
        cleanup_params_files(self.model.output_dir, self.max_params_files_to_keep,
                             self.state.checkpoint, self.state.best_checkpoint)
        if process_manager is not None:
            process_manager.wait_to_finish()
            result = process_manager.collect_results()
            if result is not None:
                decoded_checkpoint, decoder_metrics = result
                self.state.metrics[decoded_checkpoint - 1].update(decoder_metrics)
                if self.tensorboard_logger is not None:
                    self.tensorboard_logger.log_metrics(decoder_metrics, decoded_checkpoint)

        final_training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(final_training_state_dirname):
            shutil.rmtree(final_training_state_dirname)
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            best_opt_states_fname = os.path.join(self.model.output_dir, C.OPT_STATES_BEST)
            if os.path.exists(best_opt_states_fname):
                os.remove(best_opt_states_fname)

    def _initialize_parameters(self, params: Optional[str], allow_missing_params: bool):
        self.model.initialize_parameters(self.optimizer_config.initializer, allow_missing_params)
        if params is not None:
            logger.info("Training will start with parameters loaded from '%s'", params)
            self.model.load_params_from_file(params)
        self.model.log_parameters()

    def _initialize_optimizer(self):
        self.model.initialize_optimizer(self.optimizer_config)

    def _adjust_learning_rate(self, has_improved: bool, lr_decay_param_reset: bool, lr_decay_opt_states_reset: str):
        if self.optimizer_config.lr_scheduler is not None:
            if issubclass(type(self.optimizer_config.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler):
                lr_adjusted = self.optimizer_config.lr_scheduler.new_evaluation_result(has_improved)
            else:
                lr_adjusted = False
            if lr_adjusted and not has_improved:
                if lr_decay_param_reset:
                    logger.info("Loading parameters from last best checkpoint: %d",
                                self.state.best_checkpoint)
                    self.model.load_params_from_file(self.best_params_fname)
                if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
                    logger.info("Resetting optimizer states to initial")
                    self.model.load_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_INITIAL))
                elif lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
                    logger.info("Resetting optimizer states to last best checkpoint: %d",
                                self.state.best_checkpoint)
                    self.model.load_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_BEST))

    @property
    def best_params_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.PARAMS_BEST_NAME)

    @property
    def current_params_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.PARAMS_NAME % self.state.checkpoint)

    @property
    def metrics_fname(self) -> str:
        return os.path.join(self.model.output_dir, C.METRICS_NAME)

    @property
    def training_state_dirname(self) -> str:
        return os.path.join(self.model.output_dir, C.TRAINING_STATE_DIRNAME)

    @staticmethod
    def _create_eval_metric(metric_name: AnyStr) -> mx.metric.EvalMetric:
        """
        Creates an EvalMetric given a metric names.
        """
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        if metric_name == C.ACCURACY:
            return utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])
        elif metric_name == C.PERPLEXITY:
            return mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])
        else:
            raise ValueError("unknown metric name")

    @staticmethod
    def _create_eval_metric_composite(metric_names: List[AnyStr]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = [Trainer._create_eval_metric(metric_name) for metric_name in metric_names]
        return mx.metric.create(metrics)

    def _create_metrics(self,
                        metrics: List[AnyStr],
                        optimizer: mx.optimizer.Optimizer,
                        loss: loss.Loss) -> Tuple[mx.metric.EvalMetric,
                                                  mx.metric.EvalMetric,
                                                  Optional[mx.metric.EvalMetric]]:
        metric_train = self._create_eval_metric_composite(metrics)
        metric_val = self._create_eval_metric_composite(metrics)
        # If optimizer requires it, track loss as metric
        if isinstance(optimizer, SockeyeOptimizer):
            if optimizer.request_optimized_metric:
                metric_loss = self._create_eval_metric(self.training_monitor.optimized_metric)
            else:
                metric_loss = loss.create_metric()
        else:
            metric_loss = None
        return metric_train, metric_val, metric_loss

    def _update_best_params_link(self):
        """
        Updates the params.best link to the latest best parameter file.
        """
        best_params_path = self.best_params_fname
        actual_best_params_fname = C.PARAMS_NAME % self.state.best_checkpoint
        if os.path.lexists(best_params_path):
            os.remove(best_params_path)
        os.symlink(actual_best_params_fname, best_params_path)

    def _update_best_optimizer_states(self, lr_decay_opt_states_reset: str):
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            self.model.save_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_BEST))

    def _save_initial_optimizer_states(self, lr_decay_opt_states_reset: str):
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
            self.model.save_optimizer_states(os.path.join(self.model.output_dir, C.OPT_STATES_INITIAL))

    def _check_dist_kvstore_requirements(self, lr_decay_opt_states_reset, lr_decay_param_reset, optimizer):
        # In distributed training the optimizer will run remotely. For eve we however need to pass information about
        # the loss, which is not possible anymore by means of accessing self.module._curr_module._optimizer.
        utils.check_condition(optimizer != C.OPTIMIZER_EVE, "Eve optimizer not supported with distributed training.")
        utils.check_condition(
            not issubclass(type(self.optimizer_config.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler),
            "Adaptive learning rate schedulers not supported with a dist kvstore. "
            "Try a fixed schedule such as %s." % C.LR_SCHEDULER_FIXED_RATE_INV_SQRT_T)
        utils.check_condition(not lr_decay_param_reset, "Parameter reset when the learning rate decays not "
                                                        "supported with distributed training.")
        utils.check_condition(not lr_decay_opt_states_reset, "Optimizer state reset when the learning rate decays "
                                                             "not supported with distributed training.")

    def _save_params(self):
        """
        Saves model parameters at current checkpoint and optionally cleans up older parameter files to save disk space.
        """
        self.model.save_params_to_file(self.current_params_fname)
        cleanup_params_files(self.model.output_dir, self.max_params_files_to_keep, self.state.checkpoint,
                             self.state.best_checkpoint)

    def _save_training_state(self, train_iter: data_io.BaseParallelSampleIter):
        """
        Saves current training state.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)

        # (1) Parameters: link current file
        params_base_fname = C.PARAMS_NAME % self.state.checkpoint
        os.symlink(os.path.join("..", params_base_fname),
                   os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME))

        # (2) Optimizer states
        opt_state_fname = os.path.join(training_state_dirname, C.OPT_STATES_LAST)
        self.model.save_optimizer_states(opt_state_fname)

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

        # (6) Learning rate scheduler
        with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
            pickle.dump(self.optimizer_config.lr_scheduler, fp)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(self.model.output_dir, C.TRAINING_STATE_TEMP_DELETENAME)
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
        self.model.load_params_from_file(params_fname)

        # (2) Optimizer states
        opt_state_fname = os.path.join(self.training_state_dirname, C.OPT_STATES_LAST)
        self.model.load_optimizer_states(opt_state_fname)

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

        # (6) Learning rate scheduler
        with open(os.path.join(self.training_state_dirname, C.SCHEDULER_STATE_NAME), "rb") as fp:
            self.optimizer_config.lr_scheduler = pickle.load(fp)
        # initialize optimizer again
        self.model.initialize_optimizer(self.optimizer_config)


def cleanup_params_files(output_folder: str, max_to_keep: int, checkpoint: int, best_checkpoint: int):
    """
    Cleanup the params files in the output folder.

    :param output_folder: folder where param files are created.
    :param max_to_keep: maximum number of files to keep, negative to keep all.
    :param checkpoint: current checkpoint (i.e. index of last params file created).
    :param best_checkpoint: best checkpoint, we will not delete its params.
    """
    if max_to_keep <= 0:  # We assume we do not want to delete all params
        return
    existing_files = glob.glob(os.path.join(output_folder, C.PARAMS_PREFIX + "*"))
    params_name_with_dir = os.path.join(output_folder, C.PARAMS_NAME)
    for n in range(0, max(1, checkpoint - max_to_keep + 1)):
        if n != best_checkpoint:
            param_fname_n = params_name_with_dir % n
            if param_fname_n in existing_files:
                os.remove(param_fname_n)


class TensorboardLogger:

    def __init__(self, log_dir: str) -> None:
        try:
            import tensorboard  # pylint: disable=import-error
            from tensorboard.summary import scalar
        except ImportError:
            raise RuntimeError("Please install tensorboard.")
        if os.path.exists(log_dir):
            logger.info("Deleting existing tensorboard log dir %s", log_dir)
            shutil.rmtree(log_dir)
        logger.info("Logging training events for Tensorboard at '%s'", log_dir)
        self.log_dir = log_dir
        self.summary_writer = tensorboard.FileWriter(log_dir)
        self.scalar = scalar

    def log_metrics(self, metrics: Dict[str, Union[float, int]], checkpoint: int):
        for name, value in metrics.items():
            self.summary_writer.add_summary(self.scalar(name=name, scalar=value), global_step=checkpoint)


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

    def __call__(self, epoch: int, updates: int, samples: int, tokens: int, metric: Optional[mx.metric.EvalMetric]):
        count = updates
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.samples += samples
        self.tokens += tokens

        if self.init:
            if count % self.frequency == 0:
                toc = (time.time() - self.tic)
                updates_per_sec = self.frequency / toc
                samples_per_sec = self.samples / toc
                tokens_per_sec = self.tokens / toc
                self.samples = 0
                self.tokens = 0

                if metric is not None:
                    name_value = metric.get_name_value()
                    if self.auto_reset:
                        metric.reset()
                    logger.info(self.msg + '\t%s=%f' * len(name_value),
                                epoch, count, samples_per_sec, tokens_per_sec, updates_per_sec, *sum(name_value, ()))
                else:
                    logger.info(self.msg, epoch, count, samples_per_sec)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

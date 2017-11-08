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
from typing import AnyStr, List, Optional

import mxnet as mx
import numpy as np

from . import callback
from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
from . import lr_scheduler
from .optimizers import BatchState, CheckpointState, SockeyeOptimizer
from . import model
from . import utils

logger = logging.getLogger(__name__)


class _TrainingState:
    """
    Stores the state of the training process. These are the variables that will
    be stored to disk when resuming training.
    """

    def __init__(self,
                 num_not_improved,
                 epoch,
                 checkpoint,
                 updates,
                 samples):
        self.num_not_improved = num_not_improved
        self.epoch = epoch
        self.checkpoint = checkpoint
        self.updates = updates
        self.samples = samples


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
    :param lr_scheduler: The scheduler that lowers the learning rate during training.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 bucketing: bool,
                 lr_scheduler) -> None:
        super().__init__(config)
        self.context = context
        self.lr_scheduler = lr_scheduler
        self.bucketing = bucketing
        self._build_model_components()
        self.module = self._build_module(train_iter)
        self.training_monitor = None  # type: Optional[callback.TrainingMonitor]

    def _build_module(self, train_iter: data_io.ParallelBucketSentenceIter):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        utils.check_condition(train_iter.pad_id == C.PAD_ID == 0, "pad id should be 0")
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = utils.compute_lengths(source)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        model_loss = loss.get_loss(self.config.config_loss)

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
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, rnn_num_hidden)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(target_decoded)

            probs = model_loss.get_loss(logits, labels)

            return mx.sym.Group(probs), data_names, label_names

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=train_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.max_seq_len_source, self.config.max_seq_len_target)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)

    @staticmethod
    def create_eval_metric(metric_name: AnyStr) -> mx.metric.EvalMetric:
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
    def create_eval_metric_composite(metric_names: List[AnyStr]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = [TrainingModel.create_eval_metric(metric_name) for metric_name in metric_names]
        return mx.metric.create(metrics)

    def fit(self,
            train_iter: data_io.ParallelBucketSentenceIter,
            val_iter: data_io.ParallelBucketSentenceIter,
            output_folder: str,
            max_params_files_to_keep: int,
            metrics: List[AnyStr],
            initializer: mx.initializer.Initializer,
            max_updates: Optional[int],
            checkpoint_frequency: int,
            optimizer: str,
            optimizer_params: dict,
            optimized_metric: str = "perplexity",
            kvstore: str = C.KVSTORE_DEVICE,
            max_num_not_improved: int = 3,
            min_num_epochs: Optional[int] = None,
            max_num_epochs: Optional[int] = None,
            monitor_bleu: int = 0,
            use_tensorboard: bool = False,
            mxmonitor_pattern: Optional[str] = None,
            mxmonitor_stat_func: Optional[str] = None,
            lr_decay_param_reset: bool = False,
            lr_decay_opt_states_reset: str = C.LR_DECAY_OPT_STATES_RESET_OFF):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator.
        :param val_iter: The validation data iterator.
        :param output_folder: The folder in which all model artifacts will be stored in (parameters, checkpoints, etc.).
        :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
        :param metrics: The metrics that will be evaluated during training.
        :param initializer: The parameter initializer.
        :param max_updates: Optional maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing in number of updates.
        :param optimizer: The MXNet optimizer that will update the parameters.
        :param optimizer_params: The parameters for the optimizer.
        :param optimized_metric: The metric that is tracked for early stopping.
        :param kvstore: The MXNet kvstore used.
        :param max_num_not_improved: Stop training if the optimized_metric does not improve for this many checkpoints,
               -1: do not use early stopping.
        :param min_num_epochs: Optional minimum number of epochs to train, even if validation scores did not improve.
        :param max_num_epochs: Optional maximum number of epochs to train.
        :param monitor_bleu: Monitor BLEU during training (0: off, >=0: the number of sentences to decode for BLEU
               evaluation, -1: decode the full validation set.).
        :param use_tensorboard: If True write tensorboard compatible logs for monitoring training and
               validation metrics.
        :param mxmonitor_pattern: Optional pattern to match to monitor weights/gradients/outputs
               with MXNet's monitor. Default is None which means no monitoring.
        :param mxmonitor_stat_func: Choice of statistics function to run on monitored weights/gradients/outputs
               when using MXNEt's monitor.
        :param lr_decay_param_reset: Reset parameters to previous best after learning rate decay.
        :param lr_decay_opt_states_reset: How to reset optimizer states after learning rate decay.
        :return: Best score on validation data observed during training.
        """
        self.save_version(output_folder)
        self.save_config(output_folder)

        if 'dist' in kvstore:
            self._check_dist_kvstore_requirements(lr_decay_opt_states_reset, lr_decay_param_reset, optimizer)

        self.module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label,
                         for_training=True, force_rebind=True, grad_req='write')
        self.module.symbol.save(os.path.join(output_folder, C.SYMBOL_NAME))

        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False)
        self._log_params()

        self.module.init_optimizer(kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

        cp_decoder = checkpoint_decoder.CheckpointDecoder(self.context[-1],
                                                          self.config.config_data.validation_source,
                                                          self.config.config_data.validation_target,
                                                          output_folder,
                                                          sample_size=monitor_bleu) \
            if monitor_bleu else None

        logger.info("Training started.")
        self.training_monitor = callback.TrainingMonitor(train_iter.batch_size, output_folder,
                                                         optimized_metric=optimized_metric,
                                                         use_tensorboard=use_tensorboard,
                                                         cp_decoder=cp_decoder)

        monitor = None
        if mxmonitor_pattern is not None:
            monitor = mx.monitor.Monitor(interval=C.MEASURE_SPEED_EVERY,
                                         stat_func=C.MONITOR_STAT_FUNCS.get(mxmonitor_stat_func),
                                         pattern=mxmonitor_pattern,
                                         sort=True)
            self.module.install_monitor(monitor)
            logger.info("Installed MXNet monitor; pattern='%s'; statistics_func='%s'",
                        mxmonitor_pattern, mxmonitor_stat_func)

        self._fit(train_iter, val_iter, output_folder,
                  kvstore=kvstore,
                  max_params_files_to_keep=max_params_files_to_keep,
                  metrics=metrics,
                  max_updates=max_updates,
                  checkpoint_frequency=checkpoint_frequency,
                  max_num_not_improved=max_num_not_improved,
                  min_num_epochs=min_num_epochs,
                  max_num_epochs=max_num_epochs,
                  mxmonitor=monitor,
                  lr_decay_param_reset=lr_decay_param_reset,
                  lr_decay_opt_states_reset=lr_decay_opt_states_reset)

        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.training_monitor.get_best_checkpoint(),
                    self.training_monitor.optimized_metric,
                    self.training_monitor.get_best_validation_score())
        return self.training_monitor.get_best_validation_score()

    def _check_dist_kvstore_requirements(self, lr_decay_opt_states_reset, lr_decay_param_reset, optimizer):
        # In distributed training the optimizer will run remotely. For eve we however need to pass information about
        # the loss, which is not possible anymore by means of accessing self.module._curr_module._optimizer.
        utils.check_condition(optimizer != C.OPTIMIZER_EVE, "Eve optimizer not supported with distributed training.")
        utils.check_condition(not issubclass(type(self.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler),
                              "Adaptive learning rate schedulers not supported with a dist kvstore. "
                              "Try a fixed schedule such as %s." % C.LR_SCHEDULER_FIXED_RATE_INV_SQRT_T)
        utils.check_condition(not lr_decay_param_reset, "Parameter reset when the learning rate decays not "
                                                        "supported with distributed training.")
        utils.check_condition(not lr_decay_opt_states_reset, "Optimizer state reset when the learning rate decays "
                                                             "not supported with distributed training.")

    def _fit(self,
             train_iter: data_io.ParallelBucketSentenceIter,
             val_iter: data_io.ParallelBucketSentenceIter,
             output_folder: str,
             kvstore: str,
             max_params_files_to_keep: int,
             metrics: List[AnyStr],
             max_updates: Optional[int],
             checkpoint_frequency: int,
             max_num_not_improved: int,
             min_num_epochs: Optional[int] = None,
             max_num_epochs: Optional[int] = None,
             mxmonitor: Optional[mx.monitor.Monitor] = None,
             lr_decay_param_reset: bool = False,
             lr_decay_opt_states_reset: str = C.LR_DECAY_OPT_STATES_RESET_OFF):
        """
        Internal fit method. Runtime determined by early stopping.

        :param train_iter: Training data iterator.
        :param val_iter: Validation data iterator.
        :param output_folder: Model output folder.
        :param kvstore: The MXNet kvstore.
        :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
        :param metrics: List of metric names to track on training and validation data.
        :param max_updates: Optional maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing.
        :param max_num_not_improved: Maximum number of checkpoints until fitting is stopped if model does not improve,
               -1 for no early stopping.
        :param min_num_epochs: Optional minimum number of epochs to train, even if validation scores did not improve.
        :param max_num_epochs: Optional maximum number of epochs to train.
        :param mxmonitor: Optional MXNet monitor instance.
        :param lr_decay_param_reset: Reset parameters to previous best after learning rate decay.
        :param lr_decay_opt_states_reset: How to reset optimizer states after learning rate decay.
        """
        # TODO: Push update to MXNet to expose the optimizer (Module should have a get_optimizer method)
        optimizer = self.module._curr_module._optimizer if self.bucketing else self.module._optimizer
        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
            self.save_optimizer_states(os.path.join(output_folder, C.OPT_STATES_INITIAL))

        metric_train = self.create_eval_metric_composite(metrics)
        metric_val = self.create_eval_metric_composite(metrics)
        # If optimizer requires it, track loss as metric
        if isinstance(optimizer, SockeyeOptimizer):
            # Select training loss or optimized metric
            if optimizer.request_optimized_metric:
                metric_loss = self.create_eval_metric(self.training_monitor.optimized_metric)
            else:
                metric_loss = loss.get_loss(self.config.config_loss).create_metric()

        tic = time.time()

        training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(training_state_dir):
            utils.check_condition('dist' not in kvstore, "Training continuation not supported with "
                                                         "distributed training.")

            train_state = self.load_checkpoint(training_state_dir, train_iter)
        else:
            train_state = _TrainingState(
                num_not_improved=0,
                epoch=0,
                checkpoint=0,
                updates=0,
                samples=0
            )

        next_data_batch = train_iter.next()

        while True:
            if not train_iter.iter_next():
                train_state.epoch += 1
                train_iter.reset()

            if (max_updates is not None and train_state.updates == max_updates) or \
                    (max_num_epochs is not None and train_state.epoch == max_num_epochs):
                logger.info("Maximum # of updates (%s) or epochs (%s) reached.", max_updates, max_num_epochs)
                break

            # process batch
            batch = next_data_batch

            if mxmonitor is not None:
                mxmonitor.tic()

            # Forward-backward to get outputs, gradients
            self.module.forward_backward(batch)

            # Update aggregate training loss
            self.module.update_metric(metric_train, batch.label)

            # If using an extended optimizer, provide extra state information about the current batch
            # Loss: training loss
            if isinstance(optimizer, SockeyeOptimizer):
                # Loss for this batch
                metric_loss.reset()
                metric_loss.update(batch.label, self.module.get_outputs())
                [(_, m_val)] = metric_loss.get_name_value()
                batch_state = BatchState(metric_val=m_val)
                optimizer.pre_update_batch(batch_state)

            # Call optimizer to update weights given gradients, current state
            self.module.update()

            if mxmonitor is not None:
                results = mxmonitor.toc()
                if results:
                    for _, k, v in results:
                        logger.info('Monitor: Batch [{:d}] {:s} {:s}'.format(train_state.updates, k, v))

            if train_iter.iter_next():
                # pre-fetch next batch
                next_data_batch = train_iter.next()
                self.module.prepare(next_data_batch)

            self.training_monitor.batch_end_callback(train_state.epoch, train_state.updates, metric_train)
            train_state.updates += 1
            train_state.samples += train_iter.batch_size

            if train_state.updates > 0 and train_state.updates % checkpoint_frequency == 0:
                train_state.checkpoint += 1
                self._save_params(output_folder, train_state.checkpoint)
                cleanup_params_files(output_folder, max_params_files_to_keep,
                                     train_state.checkpoint, self.training_monitor.get_best_checkpoint())
                self.training_monitor.checkpoint_callback(train_state.checkpoint, metric_train,
                                                          memory_data=utils.get_gpu_memory_usage(self.context))

                toc = time.time()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f",
                            train_state.checkpoint, train_state.updates, train_state.epoch,
                            train_state.samples, (toc - tic))
                tic = time.time()

                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', train_state.checkpoint, name, val)
                metric_train.reset()

                # evaluation on validation set
                has_improved, best_checkpoint = self._evaluate(train_state, val_iter, metric_val)

                # If using an extended optimizer, provide extra state information about the current checkpoint
                # Loss: optimized metric
                if isinstance(optimizer, SockeyeOptimizer):
                    m_val = 0
                    for name, val in metric_val.get_name_value():
                        if name == self.training_monitor.optimized_metric:
                            m_val = val
                    checkpoint_state = CheckpointState(checkpoint=train_state.checkpoint, metric_val=m_val)
                    optimizer.pre_update_checkpoint(checkpoint_state)

                # learning rate adjustment
                if self.lr_scheduler is not None:
                    if issubclass(type(self.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler):
                        lr_adjusted = self.lr_scheduler.new_evaluation_result(has_improved)
                    else:
                        lr_adjusted = False
                    if lr_adjusted and not has_improved:
                        if lr_decay_param_reset:
                            logger.info("Loading parameters from last best checkpoint: %d", best_checkpoint)
                            self._load_params(output_folder, best_checkpoint)
                        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_INITIAL:
                            logger.info("Resetting optimizer states to initial")
                            self.load_optimizer_states(os.path.join(output_folder, C.OPT_STATES_INITIAL))
                        elif lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
                            logger.info("Resetting optimizer states to last best checkpoint: %d", best_checkpoint)
                            self.load_optimizer_states(os.path.join(output_folder, C.OPT_STATES_BEST))

                if has_improved:
                    best_params_path = os.path.join(output_folder, C.PARAMS_BEST_NAME)
                    if os.path.lexists(best_params_path):
                        os.remove(best_params_path)
                    actual_best_params_fname = C.PARAMS_NAME % best_checkpoint
                    os.symlink(actual_best_params_fname, best_params_path)
                    if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
                        best_opt_states_fname = os.path.join(output_folder, C.OPT_STATES_BEST)
                        if os.path.exists(best_opt_states_fname):
                            os.remove(best_opt_states_fname)
                        self.save_optimizer_states(best_opt_states_fname)
                    train_state.num_not_improved = 0
                else:
                    train_state.num_not_improved += 1
                    logger.info("Model has not improved for %d checkpoints", train_state.num_not_improved)

                if max_num_not_improved >= 0 and train_state.num_not_improved >= max_num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, train_state.num_not_improved)
                    stop_fit = True

                    if min_num_epochs is not None and train_state.epoch < min_num_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_num_epochs,
                                    train_state.epoch)
                        stop_fit = False

                    if stop_fit:
                        break

                self._checkpoint(train_state, output_folder, train_iter)

        cleanup_params_files(output_folder, max_params_files_to_keep,
                             train_state.checkpoint, self.training_monitor.get_best_checkpoint())

        logger.info('Training stopped')
        self.training_monitor.stop_fit_callback()
        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(final_training_state_dirname):
            shutil.rmtree(final_training_state_dirname)

        if lr_decay_opt_states_reset == C.LR_DECAY_OPT_STATES_RESET_BEST:
            best_opt_states_fname = os.path.join(output_folder, C.OPT_STATES_BEST)
            if os.path.exists(best_opt_states_fname):
                os.remove(best_opt_states_fname)

    def _log_params(self):
        """
        Logs information about model parameters.
        """
        arg_params, aux_params = self.module.get_params()
        total_parameters = 0
        info = []
        for name, array in sorted(arg_params.items()):
            info.append("%s: %s" % (name, array.shape))
            total_parameters += reduce(lambda x, y: x*y, array.shape)
        logger.info("Model parameters: %s", ", ".join(info))
        logger.info("Total # of parameters: %d", total_parameters)

    def _save_params(self, output_folder: str, checkpoint: int):
        """
        Synchronizes parameters across devices, saves the parameters to disk, and updates self.params.
        """
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        params_base_fname = C.PARAMS_NAME % checkpoint
        self.save_params_to_file(os.path.join(output_folder, params_base_fname))

    def _load_params(self, output_folder: str, checkpoint: int):
        """
        Loads parameters from disk, sets self.params and module's parameters.
        """
        params_fname = os.path.join(output_folder, C.PARAMS_NAME % checkpoint)
        self.load_params_from_file(params_fname)  # sets self.params
        self.module.set_params(arg_params=self.params, aux_params={})

    def _evaluate(self, training_state, val_iter, val_metric):
        """
        Computes val_metric on val_iter. Returns whether model improved or not.
        """
        val_iter.reset()
        val_metric.reset()

        for nbatch, eval_batch in enumerate(val_iter):
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(val_metric, eval_batch.label)

        for name, val in val_metric.get_name_value():
            logger.info('Checkpoint [%d]\tValidation-%s=%f', training_state.checkpoint, name, val)

        return self.training_monitor.eval_end_callback(training_state.checkpoint, val_metric)

    def _checkpoint(self, training_state: _TrainingState, output_folder: str,
                    train_iter: data_io.ParallelBucketSentenceIter):
        """
        Saves checkpoint. Note that the parameters are saved in _save_params.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)
        # Link current parameter file
        params_base_fname = C.PARAMS_NAME % training_state.checkpoint
        os.symlink(os.path.join("..", params_base_fname),
                   os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME))

        # Save current optimizer states
        opt_state_fname = os.path.join(training_state_dirname, C.OPT_STATES_LAST)
        self.save_optimizer_states(opt_state_fname)

        # State of the bucket iterator
        train_iter.save_state(os.path.join(training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp)  # Yes, one uses _, the other does not

        # Monitor state, in order to get the full information about the metrics
        self.training_monitor.save_state(os.path.join(training_state_dirname, C.MONITOR_STATE_NAME))

        # Our own state
        self.save_state(training_state, os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # The lr scheduler
        with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
            pickle.dump(self.lr_scheduler, fp)

        # We are now finished with writing. Rename the temporary directory to
        # the actual directory
        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(final_training_state_dirname):
            os.rename(final_training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, final_training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            shutil.rmtree(delete_training_state_dirname)

    @staticmethod
    def save_state(training_state: _TrainingState, fname: str):
        """
        Saves the state (of the TrainingModel class) to disk.

        :param training_state: The training state to save.
        :param fname: File name to save the state to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(training_state, fp)

    @staticmethod
    def load_state(fname: str) -> _TrainingState:
        """
        Loads the training state (of the TrainingModel class) from disk.

        :param fname: File name to load the state from.
        :return: Training state.
        """
        training_state = None
        with open(fname, "rb") as fp:
            training_state = pickle.load(fp)
        return training_state

    def save_optimizer_states(self, fname: str):
        """
        Saves optimizer states to a file.

        :param fname: File name to save optimizer states to.
        """
        if self.bucketing:
            # This is a bit hacky, as BucketingModule does not provide a
            # save_optimizer_states call. We take the current active module and
            # save its state. This should work, as the individual modules in
            # BucketingModule share the same optimizer through
            # borrow_optimizer.
            self.module._curr_module.save_optimizer_states(fname)
        else:
            self.module.save_optimizer_states(fname)

    def load_optimizer_states(self, fname: str):
        """
        Loads optimizer states from file.

        :param fname: File name to load optimizer states from.
        """
        if self.bucketing:
            # Same hacky solution as for saving the state
            self.module._curr_module.load_optimizer_states(fname)
        else:
            self.module.load_optimizer_states(fname)

    def load_checkpoint(self, directory: str, train_iter: data_io.ParallelBucketSentenceIter) -> _TrainingState:
        """
        Loads the full training state from disk. This includes optimizer,
        random number generators and everything needed.  Note that params
        should have been loaded already by the initializer.

        :param directory: directory where the state has been saved.
        :param train_iter: training data iterator.
        """

        # Optimzer state (from mxnet)
        opt_state_fname = os.path.join(directory, C.OPT_STATES_LAST)
        self.load_optimizer_states(opt_state_fname)

        # State of the bucket iterator
        train_iter.load_state(os.path.join(directory, C.BUCKET_ITER_STATE_NAME))

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(directory, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            np.random.set_state(pickle.load(fp))

        # Monitor state, in order to get the full information about the metrics
        self.training_monitor.load_state(os.path.join(directory, C.MONITOR_STATE_NAME))

        # And our own state
        return self.load_state(os.path.join(directory, C.TRAINING_STATE_NAME))


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
    for n in range(1, max(1, checkpoint - max_to_keep + 1)):
        if n != best_checkpoint:
            param_fname_n = params_name_with_dir % n
            if param_fname_n in existing_files:
                os.remove(param_fname_n)

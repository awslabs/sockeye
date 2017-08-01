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
from typing import AnyStr, List, Optional, Tuple

import mxnet as mx
import numpy as np

from . import callback
from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
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


class BaseTrainingModel(model.SockeyeModel):
    """
    Defines an Encoder/Decoder model (with attention).
    RNN configuration (number of hidden units, number of layers, cell type)
    is shared between encoder & decoder.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param train_iter: The iterator over the training data.
    :param fused: If True fused RNN cells will be used (should be slightly more efficient, but is only available
            on GPUs).
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param lr_scheduler: The scheduler that lowers the learning rate during training.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 state_names: Optional[List[str]] = None,
                 grad_req: Optional[str] = 'write') -> None:
        super().__init__(config)
        self.context = context
        self.lr_scheduler = lr_scheduler
        self.bucketing = bucketing
        self.state_names = state_names if state_names is not None else []
        self.grad_req = grad_req

        self._build_model_components(self.config.max_seq_len, fused)
        self.module = self._build_module(train_iter, self.config.max_seq_len)
        self.training_monitor = None

    def _build_module(self,
                      train_iter: data_io.ParallelBucketSentenceIter,
                      max_seq_len: int):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        default_bucket_key = train_iter.default_bucket_key
        model_symbols = self._define_symbols()
        in_out_names = self._define_names(train_iter)
        sym_gen = self._create_computation_graph(model_symbols, in_out_names)

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          state_names=self.state_names,
                                          logger=logger,
                                          default_bucket_key=train_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to max_seq_len=%s", max_seq_len)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 state_names=self.state_names,
                                 logger=logger,
                                 context=self.context)

    def _define_symbols(self):
        raise NotImplementedError()

    def _define_names(self, train_iter):
        raise NotImplementedError()

    def _create_computation_graph(self, model_symbols, in_out_names):
        raise NotImplementedError()

    def _get_shapes(self, train_iter: data_io.ParallelBucketSentenceIter):
        raise NotImplementedError()

    def _get_data_batch(self, data_iter):
        raise NotImplementedError()

    def _prepare_metric_update(self, batch):
        raise NotImplementedError()

    def _compute_gradients(self, batch: mx.io.DataBatch, train_state):
        raise NotImplementedError()

    def _clear_gradients(self):
        for grad in self.module._curr_module._exec_group.grad_arrays:
            grad[0][:] = 0

    @staticmethod
    def _create_eval_metric(metric_names: List[AnyStr]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = []
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        for metric_name in metric_names:
            if metric_name == C.ACCURACY:
                metrics.append(utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            elif metric_name == C.sBLEU:
                metrics.append(utils.SentenceBleu(output_names=[C.GEN_WORDS_OUTPUT_NAME]))
            elif metric_name == C.PERPLEXITY:
                metrics.append(mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            else:
                raise ValueError("unknown metric name")
        return mx.metric.create(metrics)

    def _set_states(self, state_names, values):
        states = self.module.get_states(merge_multi_context=False)
        for state_name, state, value in zip(state_names, states, values):
            for state_per_dev in state:
                state_per_dev[:] = value

    def fit(self,
            train_iter: data_io.ParallelBucketSentenceIter,
            val_iter: data_io.ParallelBucketSentenceIter,
            output_folder: str,
            max_params_files_to_keep: int,
            metrics: List[AnyStr],
            initializer: mx.initializer.Initializer,
            max_updates: int,
            checkpoint_frequency: int,
            optimizer: str,
            optimizer_params: dict,
            optimized_metric: str = "perplexity",
            max_num_not_improved: int = 3,
            min_num_epochs: Optional[int] = None,
            monitor_bleu: int = 0,
            use_tensorboard: bool = False,
            mxmonitor_pattern: Optional[str] = None,
            mxmonitor_stat_func: Optional[str] = None):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator.
        :param val_iter: The validation data iterator.
        :param output_folder: The folder in which all model artifacts will be stored in (parameters, checkpoints, etc.).
        :param max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
        :param metrics: The metrics that will be evaluated during training.
        :param initializer: The parameter initializer.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing in number of updates.
        :param optimizer: The MXNet optimizer that will update the parameters.
        :param optimizer_params: The parameters for the optimizer.
        :param optimized_metric: The metric that is tracked for early stopping.
        :param max_num_not_improved: Stop training if the optimized_metric does not improve for this many checkpoints.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        :param monitor_bleu: Monitor BLEU during training (0: off, >=0: the number of sentences to decode for BLEU
               evaluation, -1: decode the full validation set.).
        :param use_tensorboard: If True write tensorboard compatible logs for monitoring training and
               validation metrics.
        :param mxmonitor_pattern: Optional pattern to match to monitor weights/gradients/outputs
               with MXNet's monitor. Default is None which means no monitoring.
        :param mxmonitor_stat_func: Choice of statistics function to run on monitored weights/gradients/outputs
               when using MXNEt's monitor.
        :return: Best score on validation data observed during training.
        """
        self.save_version(output_folder)
        self.save_config(output_folder)

        data_shapes, label_shapes = self._get_shapes(train_iter)

        self.module.bind(data_shapes=data_shapes, label_shapes=label_shapes,
                         for_training=True, force_rebind=True, grad_req=self.grad_req)
        self.module.symbol.save(os.path.join(output_folder, C.SYMBOL_NAME))

        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False)

        self.module.init_optimizer(kvstore='device', optimizer=optimizer, optimizer_params=optimizer_params)

        cp_decoder = checkpoint_decoder.CheckpointDecoder(self.context[-1],
                                                          self.config.config_data.validation_source,
                                                          self.config.config_data.validation_target,
                                                          output_folder, self.config.max_seq_len,
                                                          limit=monitor_bleu) \
            if monitor_bleu else None

        logger.info("Training started.")
        self.training_monitor = callback.TrainingMonitor(train_iter.batch_size, output_folder,
                                                         optimized_metric=optimized_metric,
                                                         use_tensorboard=use_tensorboard,
                                                         checkpoint_decoder=cp_decoder)

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
                  max_params_files_to_keep,
                  metrics=metrics,
                  max_updates=max_updates,
                  checkpoint_frequency=checkpoint_frequency,
                  max_num_not_improved=max_num_not_improved,
                  min_num_epochs=min_num_epochs,
                  mxmonitor=monitor)

        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.training_monitor.get_best_checkpoint(),
                    self.training_monitor.optimized_metric,
                    self.training_monitor.get_best_validation_score())
        return self.training_monitor.get_best_validation_score()

    def _fit(self,
             train_iter: data_io.ParallelBucketSentenceIter,
             val_iter: data_io.ParallelBucketSentenceIter,
             output_folder: str,
             max_params_files_to_keep: int,
             metrics: List[AnyStr],
             max_updates: int,
             checkpoint_frequency: int,
             max_num_not_improved: int,
             min_num_epochs: Optional[int] = None,
             mxmonitor: Optional[mx.monitor.Monitor] = None):
        """
        Internal fit method. Runtime determined by early stopping.

        :param train_iter: Training data iterator.
        :param val_iter: Validation data iterator.
        :param output_folder: Model output folder.
        :params max_params_files_to_keep: Maximum number of params files to keep in the output folder (last n are kept).
        :param metrics: List of metric names to track on training and validation data.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing.
        :param max_num_not_improved: Maximum number of checkpoints until fitting is stopped if model does not improve.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        :param mxmonitor: Optional MXNet monitor instance.
        """
        metric_train = self._create_eval_metric(metrics)
        metric_val = self._create_eval_metric(metrics)

        tic = time.time()

        training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(training_state_dir):
            train_state = self.load_checkpoint(training_state_dir, train_iter)
        else:
            train_state = _TrainingState(
                num_not_improved=0,
                epoch=0,
                checkpoint=0,
                updates=0,
                samples=0
            )

        next_data_batch = self._get_data_batch(train_iter)

        while max_updates == -1 or train_state.updates < max_updates:
            if not train_iter.iter_next():
                train_state.epoch += 1
                train_iter.reset()

            # process batch
            batch = next_data_batch

            if mxmonitor is not None:
                mxmonitor.tic()

            # compute the gradients
            self._compute_gradients(batch, train_state)

            # update the model parameters usinge the gradients
            self.module.update()

            if mxmonitor is not None:
                results = mxmonitor.toc()
                if results:
                    for _, k, v in results:
                        logger.info('Monitor: Batch [{:d}] {:s} {:s}'.format(train_state.updates, k, v))

            # clear the gradients if necessary
            if self.grad_req == 'add':
                self._clear_gradients()

            if train_iter.iter_next():
                # pre-fetch next batch
                next_data_batch = self._get_data_batch(train_iter)
                self.module.prepare(next_data_batch)

            batch = self._prepare_metric_update(batch)
            self.module.update_metric(metric_train, batch.label)

            self.training_monitor.batch_end_callback(train_state.epoch, train_state.updates, metric_train)
            train_state.updates += 1
            train_state.samples += train_iter.batch_size

            if train_state.updates > 0 and train_state.updates % checkpoint_frequency == 0:
                train_state.checkpoint += 1
                self._save_params(output_folder, train_state.checkpoint)
                cleanup_params_files(output_folder, max_params_files_to_keep,
                                     train_state.checkpoint, self.training_monitor.get_best_checkpoint())
                self.training_monitor.checkpoint_callback(train_state.checkpoint, metric_train)

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
                if self.lr_scheduler is not None:
                    self.lr_scheduler.new_evaluation_result(has_improved)

                if has_improved:
                    best_path = os.path.join(output_folder, C.PARAMS_BEST_NAME)
                    if os.path.lexists(best_path):
                        os.remove(best_path)
                    actual_best_fname = C.PARAMS_NAME % best_checkpoint
                    os.symlink(actual_best_fname, best_path)
                    train_state.num_not_improved = 0
                else:
                    train_state.num_not_improved += 1
                    logger.info("Model has not improved for %d checkpoints", train_state.num_not_improved)

                if train_state.num_not_improved >= max_num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, train_state.num_not_improved)
                    stop_fit = True

                    if min_num_epochs is not None and train_state.epoch < min_num_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_num_epochs,
                                    train_state.epoch)
                        stop_fit = False

                    if stop_fit:
                        logger.info("Stopping fit")
                        self.training_monitor.stop_fit_callback()
                        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
                        if os.path.exists(final_training_state_dirname):
                            shutil.rmtree(final_training_state_dirname)
                        break

                self._checkpoint(train_state, output_folder, train_iter)
        cleanup_params_files(output_folder, max_params_files_to_keep,
                             train_state.checkpoint, self.training_monitor.get_best_checkpoint())

    def _save_params(self, output_folder: str, checkpoint: int):
        """
        Saves the parameters to disk.
        """
        arg_params, aux_params = self.module.get_params()  # sync aux params across devices
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        params_base_fname = C.PARAMS_NAME % checkpoint
        self.save_params_to_file(os.path.join(output_folder, params_base_fname))

    def _toggle_states(self, is_train=False):
        raise NotImplementedError()

    def _evaluate(self, training_state, val_iter, val_metric):
        """
        Computes val_metric on val_iter. Returns whether model improved or not.
        """
        val_iter.reset()
        val_metric.reset()

        # set the internal states for making predictions
        self._toggle_states(is_train=False)

        while val_iter.iter_next():
            eval_batch = self._get_data_batch(val_iter)
            self.module.forward(eval_batch, is_train=False)

            eval_batch = self._prepare_metric_update(eval_batch)
            self.module.update_metric(val_metric, eval_batch.label)

        # restore the states back
        self._toggle_states(is_train=True)

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

        # Optimizer state (from mxnet)
        opt_state_fname = os.path.join(training_state_dirname, C.MODULE_OPT_STATE_NAME)
        if self.bucketing:
            # This is a bit hacky, as BucketingModule does not provide a
            # save_optimizer_states call. We take the current active module and
            # save its state. This should work, as the individual modules in
            # BucketingModule share the same optimizer through
            # borrow_optimizer.
            self.module._curr_module.save_optimizer_states(opt_state_fname)
        else:
            self.module.save_optimizer_states(opt_state_fname)

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

    def save_state(self, training_state: _TrainingState, fname: str):
        """
        Saves the state (of the TrainingModel class) to disk.

        :param fname: file name to save the state to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(training_state, fp)

    def load_state(self, fname: str) -> _TrainingState:
        """
        Loads the training state (of the TrainingModel class) from disk.

        :param fname: file name to load the state from.
        """
        training_state = None
        with open(fname, "rb") as fp:
            training_state = pickle.load(fp)
        return training_state

    def load_checkpoint(self, directory: str, train_iter: data_io.ParallelBucketSentenceIter) -> _TrainingState:
        """
        Loads the full training state from disk. This includes optimizer,
        random number generators and everything needed.  Note that params
        should have been loaded already by the initializer.

        :param directory: directory where the state has been saved.
        :param train_iter: training data iterator.
        """

        # Optimzer state (from mxnet)
        opt_state_fname = os.path.join(directory, C.MODULE_OPT_STATE_NAME)
        if self.bucketing:
            # Same hacky solution as for saving the state
            self.module._curr_module.load_optimizer_states(opt_state_fname)
        else:
            self.module.load_optimizer_states(opt_state_fname)

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


class MLETrainingModel(BaseTrainingModel):
    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 state_names: Optional[List[str]] = None,
                 grad_req: Optional[str]='write') -> None:
        super().__init__(config, context, train_iter, fused, bucketing, lr_scheduler, state_names, grad_req)

    def _define_symbols(self) -> List[mx.sym.Symbol]:
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        model_loss = loss.get_loss(self.config.config_loss)

        return [source, source_length, target, labels, model_loss]

    def _define_names(self, train_iter) -> Tuple[List['str'], List['str']]:
        data_names = [x[0] for x in train_iter.provide_data]
        label_names = [x[0] for x in train_iter.provide_label]

        return (data_names, label_names)

    def _create_computation_graph(self, model_symbols, in_out_names) -> Tuple[mx.sym.Symbol, List['str'], List['str']]:
        source, source_length, target, labels, model_loss = model_symbols
        data_names, label_names = in_out_names

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            if self.config.config_decoder.scheduled_sampling_type is not None:
                # scheduled sampling
                probs, threshold = self.decoder.decode_iter(source_encoded, source_encoded_seq_len, source_encoded_length,
                                                            target, target_seq_len, source_lexicon)

                cross_entropy = mx.sym.one_hot(indices=mx.sym.cast(data=labels, dtype='int32'),
                                               depth=self.config.vocab_target_size)

                # zero out pad symbols (0)
                cross_entropy = mx.sym.where(labels,
                                             cross_entropy, mx.sym.zeros((0, self.config.vocab_target_size)))

                # compute cross_entropy
                cross_entropy = - mx.sym.log(data=probs + 1e-10) * cross_entropy
                cross_entropy = mx.sym.sum(data=cross_entropy, axis=1)

                cross_entropy = mx.sym.MakeLoss(cross_entropy, name=C.CROSS_ENTROPY)

                probs = mx.sym.BlockGrad(probs, name=C.SOFTMAX_NAME)
                threshold = mx.sym.BlockGrad(threshold)

                outputs = [cross_entropy, probs, threshold]
            else:
                logits = self.decoder.decode(source_encoded, source_encoded_seq_len, source_encoded_length,
                                            target, target_seq_len, source_lexicon)

                outputs = model_loss.get_loss(logits, labels)

            return mx.sym.Group(outputs), data_names, label_names

        return sym_gen

    def _get_shapes(self, train_iter) -> Tuple[mx.io.DataDesc, mx.io.DataDesc]:
        data_shapes = train_iter.provide_data
        label_shapes = train_iter.provide_label

        return (data_shapes, label_shapes)

    def _get_data_batch(self, data_iter) -> mx.io.DataBatch:
        next_data_batch = data_iter.next()

        return next_data_batch

    def _prepare_metric_update(self, batch) -> mx.io.DataBatch:
        return batch

    def _compute_gradients(self, batch: mx.io.DataBatch, train_state):
        if self.state_names is not None:
            # XXX update the internal states which are used for the scheduled sampling
            self._set_states(self.state_names, [eval('train_state.updates')])

        self.module.forward_backward(batch)

    def _toggle_states(self, is_train=False):
        # if prediction is requested by setting 'is_train = False', the model's internal states are all zero (initial values).
        self._set_states(self.state_names, [int(is_train)])


class MRTrainingModel(BaseTrainingModel):
    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 state_names,
                 grad_req) -> None:
        self.mrt_metric = config.config_decoder.mrt_metric
        self.mrt_entropy_reg = config.config_decoder.mrt_entropy_reg
        self.mrt_num_samples = config.config_decoder.mrt_num_samples
        self.mrt_sup_grad_scale = config.config_decoder.mrt_sup_grad_scale
        self.mrt_max_target_seq_ratio = config.config_decoder.mrt_max_target_len_ratio

        super().__init__(config, context, train_iter, fused, bucketing, lr_scheduler, state_names, grad_req)

    def _define_symbols(self) -> List[mx.sym.Symbol]:
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.Variable(C.TARGET_LABEL_NAME)
        baseline = mx.sym.Variable('baseline')
        is_sample = mx.sym.Variable('is_sample', shape=(1,), dtype='int32')

        return [source, source_length, target, labels, baseline, is_sample]

    def _define_names(self, train_iter) -> Tuple[List['str'], List['str']]:
        batch_size = train_iter.batch_size
        data_shapes = train_iter.provide_data + [mx.io.DataDesc(name='baseline', shape=(batch_size,), layout=C.BATCH_MAJOR)]
        label_shapes = train_iter.provide_label

        data_names = [x[0] for x in data_shapes]
        label_names = [x[0] for x in label_shapes]

        return data_names, label_names

    def _create_computation_graph(self, model_symbols, in_out_names) -> Tuple[mx.sym.Symbol, List['str'], List['str']]:
        source, source_length, target, labels, baseline, is_sample = model_symbols
        data_names, label_names = in_out_names

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            outputs = self.decoder.decode_mrt(source_encoded, source_seq_len, source_length,
                                              labels, target_seq_len, self.mrt_max_target_seq_ratio, is_sample,
                                              self.mrt_entropy_reg, self.mrt_sup_grad_scale, source_lexicon)
            sampled_words, sample_probs = outputs

            mrt_loss = mx.sym.Custom(data=sampled_words, label=labels,
                                     baseline=baseline, is_sampled=is_sample,
                                     metric=self.mrt_metric,
                                     ignore_ids=[C.PAD_ID],
                                     eos_id=C.EOS_ID,
                                     name=C.TARGET_NAME, op_type='Risk')

            softmax_output = mx.sym.reshape(sample_probs,
                                            shape=(-1, self.config.vocab_target_size))

            losses = mx.sym.Group([mrt_loss,
                                   mx.sym.BlockGrad(sampled_words, name=C.GEN_WORDS_NAME),
                                   mx.sym.BlockGrad(softmax_output, name=C.SOFTMAX_NAME),
                                   mx.sym.BlockGrad(target)])


            return losses, data_names, label_names

        return sym_gen

    def _get_shapes(self, train_iter) -> Tuple[mx.io.DataDesc, mx.io.DataDesc]:
        data_shapes = self._create_mrt_data_shapes(train_iter.provide_data, train_iter.batch_size)
        label_shapes = train_iter.provide_label

        return (data_shapes, label_shapes)

    def _get_data_batch(self, data_iter) -> mx.io.DataBatch:
        next_data_batch = data_iter.next()
        next_data_batch = self._create_mrt_batch(next_data_batch)

        return next_data_batch

    def _prepare_metric_update(self, batch) -> mx.io.DataBatch:
        batch.label = self._expand_label_seq(batch.label, max_target_seq_len_ratio=self.mrt_max_target_seq_ratio)

        return batch

    def _compute_gradients(self, batch: mx.io.DataBatch, train_state):
        self.module.forward_backward(batch)

        # XXX run multiple forward passes to collect the average reward
        self._set_states(self.state_names, [1]) # is_sample = 1

        if self.mrt_num_samples > 0:
            baseline = batch.data[-1]
            for i in range(self.mrt_num_samples):
                self.module.forward(batch, is_train=False)
                risk_output = self.module.get_outputs()[0].as_in_context(baseline.context)
                baseline[:] += risk_output
            baseline[:] += 1
            baseline[:] /= (self.mrt_num_samples + 1)

        self.module.forward_backward(batch)

        self._set_states(self.state_names, [0]) # is_sample = 0

    def _toggle_states(self, is_train=False):
        # if prediction is requested by setting 'is_train = False', the model will generate sample sequences.
        # That is, the model generates a sequence of words following word distribution (, or the policy).

        self._set_states(self.state_names, [1-int(is_train)])


    @staticmethod
    def _create_mrt_data_shapes(data_shapes, batch_size):
        new_data_shapes = data_shapes + \
            [mx.io.DataDesc(name='baseline', shape=(batch_size,),
                            layout=C.BATCH_MAJOR)]

        return new_data_shapes

    @staticmethod
    def _create_mrt_batch(data_batch: mx.io.DataBatch):
        batch_size = data_batch.label[0].shape[0]

        data_batch.data = data_batch.data + [mx.nd.zeros((batch_size,))]
        data_batch.provide_data = data_batch.provide_data + \
            [mx.io.DataDesc(name='baseline', shape=(batch_size,),
                            layout=C.BATCH_MAJOR)]

        return data_batch

    @staticmethod
    def _expand_label_seq(labels: List[mx.nd.NDArray],
                          max_target_seq_len_ratio: float=1.5):

        for label_idx in range(len(labels)):
            batch_size, target_seq_len = labels[label_idx].shape
            diff_target_seq_len = int(target_seq_len*max_target_seq_len_ratio) - target_seq_len
            labels[label_idx] = mx.nd.concat(*[labels[label_idx],
                                               mx.nd.zeros((batch_size, diff_target_seq_len))],
                                             dim=1)

        return labels


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

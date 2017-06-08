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
import os
import time
from typing import List, AnyStr

import mxnet as mx

import sockeye.callback
import sockeye.checkpoint_decoder
import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
import sockeye.loss
import sockeye.lr_scheduler
import sockeye.model
import sockeye.utils

logger = logging.getLogger(__name__)


class TrainingModel(sockeye.model.SockeyeModel):
    """
    Defines an Encoder/Decoder model (with attention).
    RNN configuration (number of hidden units, number of layers, cell type)
    is shared between encoder & decoder.

    :param model_config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param train_iter: The iterator over the training data.
    :param fused: If True fused RNN cells will be used (should be slightly more efficient, but is only available
            on GPUs).
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param lr_scheduler: The scheduler that lowers the learning rate during training.
    :param rnn_forget_bias: Initial value of the RNN forget biases.
    """

    def __init__(self,
                 model_config: sockeye.model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 rnn_forget_bias: float) -> None:
        super().__init__(model_config)
        self.context = context
        self.lr_scheduler = lr_scheduler
        self._build_model_components(self.config.max_seq_len, fused, rnn_forget_bias)
        self.module = self._build_module(train_iter, self.config.max_seq_len, bucketing)

    def _build_module(self,
                      train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                      max_seq_len: int,
                      bucketing: bool):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        loss = sockeye.loss.get_loss(self.config)

        data_names = [x[0] for x in train_iter.provide_data]
        label_names = [x[0] for x in train_iter.provide_label]

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            source_encoded = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            logits = self.decoder.decode(source_encoded, source_seq_len, source_length,
                                         target, target_seq_len, source_lexicon)

            outputs = loss.get_loss(logits, labels)

            return mx.sym.Group(outputs), data_names, label_names

        if bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=train_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to max_seq_len=%s", max_seq_len)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)

    @staticmethod
    def _create_eval_metric(metric_names: List[AnyStr]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = []
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        for metric_name in metric_names:
            if metric_name == C.ACCURACY:
                metrics.append(sockeye.utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            elif metric_name == C.PERPLEXITY:
                metrics.append(mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            else:
                raise ValueError("unknown metric name")
        return mx.metric.create(metrics)

    def fit(self,
            train_iter: sockeye.data_io.ParallelBucketSentenceIter,
            val_iter: sockeye.data_io.ParallelBucketSentenceIter,
            output_folder: str,
            metrics: List[AnyStr],
            initializer: mx.initializer.Initializer,
            max_updates: int,
            checkpoint_frequency: int,
            optimizer: str,
            optimizer_params: dict,
            optimized_metric: str = "perplexity",
            max_num_not_improved: int = 3,
            monitor_bleu: int = 0,
            use_tensorboard: bool = False):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator.
        :param val_iter: The validation data iterator.
        :param output_folder: The folder in which all model artifacts will be stored in (parameters, checkpoints, etc.).
        :param metrics: The metrics that will be evaluated during training.
        :param initializer: The parameter initializer.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing in number of updates.
        :param optimizer: The MXNet optimizer that will update the parameters.
        :param optimizer_params: The parameters for the optimizer.
        :param optimized_metric: The metric that is tracked for early stopping.
        :param max_num_not_improved: Stop training if the optimized_metric does not improve for this many checkpoints.
        :param monitor_bleu: Monitor BLEU during training (0: off, >=0: the number of sentences to decode for BLEU
               evaluation, -1: decode the full validation set.).
        :param use_tensorboard: If True write tensorboard compatible logs for monitoring training and
               validation metrics.
        :return: Best score on validation data observed during training.
        """
        self.save_config(output_folder)

        self.module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label,
                         for_training=True, force_rebind=True, grad_req='write')
        self.module.symbol.save(os.path.join(output_folder, C.SYMBOL_NAME))

        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False)

        self.module.init_optimizer(kvstore='device', optimizer=optimizer, optimizer_params=optimizer_params)

        checkpoint_decoder = sockeye.checkpoint_decoder.CheckpointDecoder(self.context[-1],
                                                                          self.config.data_info.validation_source,
                                                                          self.config.data_info.validation_target,
                                                                          output_folder, self.config.max_seq_len,
                                                                          limit=monitor_bleu) \
            if monitor_bleu else None

        logger.info("Training started.")
        training_monitor = sockeye.callback.TrainingMonitor(train_iter.batch_size, output_folder,
                                                            optimized_metric=optimized_metric,
                                                            use_tensorboard=use_tensorboard,
                                                            checkpoint_decoder=checkpoint_decoder)
        self._fit(train_iter, val_iter, output_folder,
                  training_monitor,
                  metrics=metrics,
                  max_updates=max_updates,
                  checkpoint_frequency=checkpoint_frequency,
                  max_num_not_improved=max_num_not_improved)

        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    training_monitor.get_best_checkpoint(),
                    training_monitor.optimized_metric,
                    training_monitor.get_best_validation_score())
        return training_monitor.get_best_validation_score()

    def _fit(self,
             train_iter: sockeye.data_io.ParallelBucketSentenceIter,
             val_iter: sockeye.data_io.ParallelBucketSentenceIter,
             output_folder: str,
             training_monitor: sockeye.callback.TrainingMonitor,
             metrics: List[AnyStr],
             max_updates: int,
             checkpoint_frequency: int,
             max_num_not_improved: int):
        """
        Internal fit method. Runtime determined by early stopping.
        
        :param train_iter: Training data iterator.
        :param val_iter: Validation data iterator.
        :param output_folder: Model output folder.
        :param metrics: List of metric names to track on training and validation data.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing.
        :param max_num_not_improved: Maximum number of checkpoints until fitting is stopped if model does not improve.
        """
        metric_train = self._create_eval_metric(metrics)
        metric_val = self._create_eval_metric(metrics)
        num_not_improved = 0
        tic = time.time()
        epoch = 0
        checkpoint = 0
        updates = 0
        samples = 0
        next_data_batch = train_iter.next()
        while max_updates == -1 or updates < max_updates:
            if not train_iter.iter_next():
                epoch += 1
                train_iter.reset()

            # process batch
            batch = next_data_batch
            self.module.forward_backward(batch)
            self.module.update()

            if train_iter.iter_next():
                # pre-fetch next batch
                next_data_batch = train_iter.next()
                self.module.prepare(next_data_batch)

            self.module.update_metric(metric_train, batch.label)
            training_monitor.batch_end_callback(epoch, updates, metric_train)
            updates += 1
            samples += train_iter.batch_size

            if updates > 0 and updates % checkpoint_frequency == 0:
                checkpoint += 1
                self._checkpoint(checkpoint, output_folder)
                training_monitor.checkpoint_callback(checkpoint, metric_train)

                toc = time.time()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f",
                            checkpoint, updates, epoch, samples, (toc - tic))
                tic = time.time()

                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', checkpoint, name, val)
                metric_train.reset()

                # evaluation on validation set
                has_improved, best_checkpoint = self._evaluate(checkpoint, val_iter, metric_val, training_monitor)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.new_evaluation_result(has_improved)

                if has_improved:
                    best_path = os.path.join(output_folder, C.PARAMS_BEST_NAME)
                    if os.path.lexists(best_path):
                        os.remove(best_path)
                    actual_best_fname = C.PARAMS_NAME % best_checkpoint
                    os.symlink(actual_best_fname, best_path)
                    num_not_improved = 0
                else:
                    num_not_improved += 1

                if num_not_improved == max_num_not_improved:
                    logger.info("Model has not improved for %d checkpoints. Stopping fit.", num_not_improved)
                    training_monitor.stop_fit_callback()
                    break

    def _evaluate(self, checkpoint, val_iter, val_metric, training_monitor):
        """
        Computes val_metric on val_iter. Returns whether model improved or not.
        """
        val_iter.reset()
        val_metric.reset()

        for nbatch, eval_batch in enumerate(val_iter):
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(val_metric, eval_batch.label)

        for name, val in val_metric.get_name_value():
            logger.info('Checkpoint [%d]\tValidation-%s=%f', checkpoint, name, val)

        return training_monitor.eval_end_callback(checkpoint, val_metric)

    def _checkpoint(self, checkpoint, output_folder):
        """
        Saves checkpoint.
        """
        # sync aux params across devices
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        self.save_params_to_file(os.path.join(output_folder, C.PARAMS_NAME % checkpoint))

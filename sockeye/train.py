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
Simple Training CLI.
"""
import argparse
import os
import random
import sys
from contextlib import ExitStack
from typing import Optional, Dict

import mxnet as mx
import numpy as np

import sockeye.arguments as arguments
import sockeye.attention
import sockeye.constants as C
import sockeye.data_io
import sockeye.decoder
import sockeye.encoder
import sockeye.initializer
import sockeye.lexicon
import sockeye.lr_scheduler
import sockeye.model
import sockeye.training
import sockeye.utils
import sockeye.vocab
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpu, get_num_gpus


def none_if_negative(val):
    return None if val < 0 else val


def _build_or_load_vocab(existing_vocab_path: Optional[str], data_path: str, num_words: int,
                         word_min_count: int) -> Dict:
    if existing_vocab_path is None:
        vocabulary = sockeye.vocab.build_from_path(data_path,
                                                   num_words=num_words,
                                                   min_count=word_min_count)
    else:
        vocabulary = sockeye.vocab.vocab_from_json(existing_vocab_path)
    return vocabulary


def main():
    params = argparse.ArgumentParser(description='CLI to train sockeye sequence-to-sequence models.')
    params = arguments.add_io_args(params)
    params = arguments.add_model_parameters(params)
    params = arguments.add_training_args(params)
    params = arguments.add_device_args(params)
    args = params.parse_args()

    # seed the RNGs
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    if args.use_fused_rnn:
        assert not args.use_cpu, "GPU required for FusedRNN cells"

    if args.rnn_residual_connections:
        assert args.rnn_num_layers > 2, "Residual connections require at least 3 RNN layers"

    assert args.optimized_metric == C.BLEU or args.optimized_metric in args.metrics, \
        "Must optimize either BLEU or one of tracked metrics (--metrics)"

    output_folder = os.path.abspath(args.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logger = setup_main_logger(__name__, console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))

    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)

    with ExitStack() as exit_stack:
        # context
        if args.use_cpu:
            context = [mx.cpu()]
        else:
            num_gpus = get_num_gpus()
            assert num_gpus > 0, "No GPUs found, consider running on the CPU with --use-cpu " \
                                 "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi " \
                                 "binary isn't on the path)."
            context = []
            for gpu_id in args.device_ids:
                if gpu_id < 0:
                    # get an automatic gpu id:
                    gpu_id = exit_stack.enter_context(acquire_gpu())
                context.append(mx.gpu(gpu_id))

        # create vocabs
        vocab_source = _build_or_load_vocab(args.source_vocab, args.source, args.num_words, args.word_min_count)
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX)

        vocab_target = _build_or_load_vocab(args.target_vocab, args.target, args.num_words, args.word_min_count)
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX)

        vocab_source_size = len(vocab_source)
        vocab_target_size = len(vocab_target)
        logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)

        data_info = sockeye.data_io.DataInfo(os.path.abspath(args.source),
                                             os.path.abspath(args.target),
                                             os.path.abspath(args.validation_source),
                                             os.path.abspath(args.validation_target),
                                             args.source_vocab,
                                             args.target_vocab)

        # create data iterators
        train_iter, eval_iter = sockeye.data_io.get_training_data_iters(source=data_info.source,
                                                                        target=data_info.target,
                                                                        validation_source=data_info.validation_source,
                                                                        validation_target=data_info.validation_target,
                                                                        vocab_source=vocab_source,
                                                                        vocab_target=vocab_target,
                                                                        batch_size=args.batch_size,
                                                                        fill_up=args.fill_up,
                                                                        max_seq_len=args.max_seq_len,
                                                                        bucketing=not args.no_bucketing,
                                                                        bucket_width=args.bucket_width)

        # learning rate scheduling
        learning_rate_half_life = none_if_negative(args.learning_rate_half_life)
        lr_scheduler = sockeye.lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type,
                                                             args.checkpoint_frequency,
                                                             learning_rate_half_life,
                                                             args.learning_rate_reduce_factor,
                                                             args.learning_rate_reduce_num_not_improved)

        # model configuration
        num_embed_source = args.num_embed if args.num_embed_source is None else args.num_embed_source
        num_embed_target = args.num_embed if args.num_embed_target is None else args.num_embed_target
        attention_num_hidden = args.rnn_num_hidden if not args.attention_num_hidden else args.attention_num_hidden
        model_config = sockeye.model.ModelConfig(max_seq_len=args.max_seq_len,
                                                 vocab_source_size=vocab_source_size,
                                                 vocab_target_size=vocab_target_size,
                                                 num_embed_source=num_embed_source,
                                                 num_embed_target=num_embed_target,
                                                 attention_type=args.attention_type,
                                                 attention_num_hidden=attention_num_hidden,
                                                 attention_coverage_type=args.attention_coverage_type,
                                                 attention_coverage_num_hidden=args.attention_coverage_num_hidden,
                                                 attention_use_prev_word=args.attention_use_prev_word,
                                                 dropout=args.dropout,
                                                 rnn_cell_type=args.rnn_cell_type,
                                                 rnn_num_layers=args.rnn_num_layers,
                                                 rnn_num_hidden=args.rnn_num_hidden,
                                                 rnn_residual_connections=args.rnn_residual_connections,
                                                 weight_tying=args.weight_tying,
                                                 context_gating=args.context_gating,
                                                 lexical_bias=args.lexical_bias,
                                                 learn_lexical_bias=args.learn_lexical_bias,
                                                 data_info=data_info,
                                                 loss=args.loss,
                                                 normalize_loss=args.normalize_loss,
                                                 smoothed_cross_entropy_alpha=args.smoothed_cross_entropy_alpha)

        # create training model
        model = sockeye.training.TrainingModel(model_config=model_config,
                                               context=context,
                                               train_iter=train_iter,
                                               fused=args.use_fused_rnn,
                                               bucketing=not args.no_bucketing,
                                               lr_scheduler=lr_scheduler,
                                               rnn_forget_bias=args.rnn_forget_bias)

        if args.params:
            model.load_params_from_file(args.params)
            logger.info("Training will continue from parameters loaded from '%s'", args.params)

        lexicon = sockeye.lexicon.initialize_lexicon(args.lexical_bias,
                                                     vocab_source, vocab_target) if args.lexical_bias else None

        initializer = sockeye.initializer.get_initializer(args.rnn_h2h_init, lexicon=lexicon)

        optimizer = args.optimizer
        optimizer_params = {'wd': args.weight_decay,
                            "learning_rate": args.initial_learning_rate}
        if lr_scheduler is not None:
            optimizer_params["lr_scheduler"] = lr_scheduler
        clip_gradient = none_if_negative(args.clip_gradient)
        if clip_gradient is not None:
            optimizer_params["clip_gradient"] = clip_gradient
        if args.momentum is not None:
            optimizer_params["momentum"] = args.momentum
        logger.info("Optimizer: %s", optimizer)
        logger.info("Optimizer Parameters: %s", optimizer_params)

        model.fit(train_iter, eval_iter,
                  output_folder=output_folder,
                  metrics=args.metrics,
                  initializer=initializer,
                  max_updates=args.max_updates,
                  checkpoint_frequency=args.checkpoint_frequency,
                  optimizer=optimizer, optimizer_params=optimizer_params,
                  optimized_metric=args.optimized_metric,
                  max_num_not_improved=args.max_num_checkpoint_not_improved,
                  monitor_bleu=args.monitor_bleu,
                  use_tensorboard=args.use_tensorboard)


if __name__ == "__main__":
    main()

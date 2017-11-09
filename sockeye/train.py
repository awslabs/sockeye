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
import json
import os
import pickle
import shutil
import sys
from contextlib import ExitStack
from typing import Optional, Dict, List, Tuple

import mxnet as mx

from sockeye.config import Config
from sockeye.log import setup_main_logger
from sockeye.utils import check_condition
from . import arguments
from . import rnn_attention
from . import constants as C
from . import coverage
from . import data_io
from . import decoder
from . import encoder
from . import initializer
from . import loss
from . import lr_scheduler
from . import model
from . import rnn
from . import convolution
from . import training
from . import transformer
from . import utils
from . import vocab

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = setup_main_logger(__name__, file_logging=False, console=True)


def none_if_negative(val):
    return None if val < 0 else val


def _build_or_load_vocab(existing_vocab_path: Optional[str], data_paths: List[str], num_words: int,
                         word_min_count: int) -> Dict:
    if existing_vocab_path is None:
        vocabulary = vocab.build_from_paths(paths=data_paths,
                                            num_words=num_words,
                                            min_count=word_min_count)
    else:
        vocabulary = vocab.vocab_from_json(existing_vocab_path)
    return vocabulary


def _list_to_tuple(v):
    """Convert v to a tuple if it is a list."""
    if isinstance(v, list):
        return tuple(v)
    return v


def _dict_difference(dict1: Dict, dict2: Dict):
    diffs = set()
    for k, v in dict1.items():
        # Note: A list and a tuple with the same values is considered equal
        # (this is due to json deserializing former tuples as list).
        if k not in dict2 or _list_to_tuple(dict2[k]) != _list_to_tuple(v):
            diffs.add(k)
    return diffs


def check_arg_compatibility(args: argparse.Namespace):
    """
    Check if some arguments are incompatible with each other.

    :param args: Arguments as returned by argparse.
    """
    check_condition(args.optimized_metric == C.BLEU or args.optimized_metric in args.metrics,
                    "Must optimize either BLEU or one of tracked metrics (--metrics)")

    if args.encoder == C.TRANSFORMER_TYPE:
        check_condition(args.transformer_model_size == args.num_embed[0],
                        "Source embedding size must match transformer model size: %s vs. %s"
                        % (args.transformer_model_size, args.num_embed[0]))
    if args.decoder == C.TRANSFORMER_TYPE:
        check_condition(args.transformer_model_size == args.num_embed[1],
                        "Target embedding size must match transformer model size: %s vs. %s"
                        % (args.transformer_model_size, args.num_embed[1]))



def check_resume(args: argparse.Namespace, output_folder: str) -> Tuple[bool, str]:
    """
    Check if we should resume a broken training run.

    :param args: Arguments as returned by argparse.
    :param output_folder: Main output folder for the model.
    :return: Flag signaling if we are resuming training and the directory with
        the training status.
    """
    resume_training = False
    training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
    if os.path.exists(output_folder):
        if args.overwrite_output:
            logger.info("Removing existing output folder %s.", output_folder)
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        elif os.path.exists(training_state_dir):
            with open(os.path.join(output_folder, C.ARGS_STATE_NAME), "r") as fp:
                old_args = json.load(fp)
            arg_diffs = _dict_difference(vars(args), old_args) | _dict_difference(old_args, vars(args))
            # Remove args that may differ without affecting the training.
            arg_diffs -= set(C.ARGS_MAY_DIFFER)
            # allow different device-ids provided their total count is the same
            if 'device_ids' in arg_diffs and len(old_args['device_ids']) == len(vars(args)['device_ids']):
                arg_diffs.discard('device_ids')
            if not arg_diffs:
                resume_training = True
            else:
                # We do not have the logger yet
                logger.error("Mismatch in arguments for training continuation.")
                logger.error("Differing arguments: %s.", ", ".join(arg_diffs))
                sys.exit(1)
        elif os.path.exists(os.path.join(output_folder, C.PARAMS_BEST_NAME)):
            logger.error("Refusing to overwrite model folder %s as it seems to contain a trained model.", output_folder)
            sys.exit(1)
        else:
            logger.info("The output folder %s already exists, but no training state or parameter file was found. "
                        "Will start training from scratch.", output_folder)
    else:
        os.makedirs(output_folder)

    return resume_training, training_state_dir


def determine_context(args: argparse.Namespace, exit_stack: ExitStack) -> List[mx.Context]:
    """
    Determine the context we should run on (CPU or GPU).

    :param args: Arguments as returned by argparse.
    :param exit_stack: An ExitStack from contextlib.
    :return: A list with the context(s) to run on.
    """
    if args.use_cpu:
        logger.info("Device: CPU")
        context = [mx.cpu()]
    else:
        num_gpus = utils.get_num_gpus()
        check_condition(num_gpus >= 1,
                        "No GPUs found, consider running on the CPU with --use-cpu "
                        "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi "
                        "binary isn't on the path).")
        if args.disable_device_locking:
            context = utils.expand_requested_device_ids(args.device_ids)
        else:
            context = exit_stack.enter_context(utils.acquire_gpus(args.device_ids, lock_dir=args.lock_dir))
        if args.batch_type == C.BATCH_TYPE_SENTENCE:
            check_condition(args.batch_size % len(context) == 0, "When using multiple devices the batch size must be "
                                                                 "divisible by the number of devices. Choose a batch "
                                                                 "size that is a multiple of %d." % len(context))
        logger.info("Device(s): GPU %s", context)
        context = [mx.gpu(gpu_id) for gpu_id in context]
    return context


def load_or_create_vocabs(args: argparse.Namespace, resume_training: bool, output_folder: str) -> Tuple[Dict, Dict]:
    """
    Load the vocabularies from disks if given, create them if not.

    :param args: Arguments as returned by argparse.
    :param resume_training: When True, the vocabulary will be loaded from an existing output folder.
    :param output_folder: Main output folder for the training.
    :return: The source and target vocabularies.
    """
    if resume_training:
        vocab_source = vocab.vocab_from_json_or_pickle(os.path.join(output_folder, C.VOCAB_SRC_NAME))
        vocab_target = vocab.vocab_from_json_or_pickle(os.path.join(output_folder, C.VOCAB_TRG_NAME))
    else:
        num_words_source, num_words_target = args.num_words
        word_min_count_source, word_min_count_target = args.word_min_count

        # if the source and target embeddings are tied we build a joint vocabulary:
        if args.weight_tying and C.WEIGHT_TYING_SRC in args.weight_tying_type \
                and C.WEIGHT_TYING_TRG in args.weight_tying_type:
            vocab_source = vocab_target = _build_or_load_vocab(args.source_vocab,
                                                               [args.source, args.target],
                                                               num_words_source,
                                                               word_min_count_source)
        else:
            vocab_source = _build_or_load_vocab(args.source_vocab, [args.source],
                                                num_words_source, word_min_count_source)
            vocab_target = _build_or_load_vocab(args.target_vocab, [args.target],
                                                num_words_target, word_min_count_target)

        # write vocabularies
        vocab.vocab_to_json(vocab_source, os.path.join(output_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX)
        vocab.vocab_to_json(vocab_target, os.path.join(output_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX)

    return vocab_source, vocab_target


def create_data_iters(args: argparse.Namespace,
                      vocab_source: Dict,
                      vocab_target: Dict) -> Tuple['data_io.ParallelBucketSentenceIter',
                                                   'data_io.ParallelBucketSentenceIter',
                                                   'data_io.DataConfig']:
    """
    Create the data iterators.

    :param args: Arguments as returned by argparse.
    :param vocab_source: The source vocabulary.
    :param vocab_target: The target vocabulary.
    :return: The data iterators (train, validation, config_data).
    """
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    return data_io.get_training_data_iters(source=os.path.abspath(args.source),
                                           target=os.path.abspath(args.target),
                                           validation_source=os.path.abspath(
                                               args.validation_source),
                                           validation_target=os.path.abspath(
                                               args.validation_target),
                                           vocab_source=vocab_source,
                                           vocab_target=vocab_target,
                                           vocab_source_path=args.source_vocab,
                                           vocab_target_path=args.target_vocab,
                                           batch_size=args.batch_size,
                                           batch_by_words=args.batch_type == C.BATCH_TYPE_WORD,
                                           batch_num_devices=batch_num_devices,
                                           fill_up=args.fill_up,
                                           max_seq_len_source=max_seq_len_source,
                                           max_seq_len_target=max_seq_len_target,
                                           bucketing=not args.no_bucketing,
                                           bucket_width=args.bucket_width,
                                           sequence_limit=args.limit)


def create_lr_scheduler(args: argparse.Namespace, resume_training: bool,
                        training_state_dir: str) -> lr_scheduler.LearningRateScheduler:
    """
    Create the learning rate scheduler.

    :param args: Arguments as returned by argparse.
    :param resume_training: When True, the scheduler will be loaded from disk.
    :param training_state_dir: Directory where the training state is stored.
    :return: The learning rate scheduler.
    """
    learning_rate_half_life = none_if_negative(args.learning_rate_half_life)
    # TODO: The loading for continuation of the scheduler is done separately from the other parts
    if not resume_training:
        lr_scheduler_instance = lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type,
                                                              args.checkpoint_frequency,
                                                              learning_rate_half_life,
                                                              args.learning_rate_reduce_factor,
                                                              args.learning_rate_reduce_num_not_improved,
                                                              args.learning_rate_schedule,
                                                              args.learning_rate_warmup)
    else:
        with open(os.path.join(training_state_dir, C.SCHEDULER_STATE_NAME), "rb") as fp:
            lr_scheduler_instance = pickle.load(fp)
    return lr_scheduler_instance


def create_encoder_config(args: argparse.Namespace,
                          config_conv: Optional[encoder.ConvolutionalEmbeddingConfig]) -> Tuple[Config, int]:
    """
    Create the encoder config.

    :param args: Arguments as returned by argparse.
    :param config_conv: The config for the convolutional encoder (optional).
    :return: The encoder config and the number of hidden units of the encoder.
    """
    encoder_num_layers, _ = args.num_layers
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    num_embed_source, _ = args.num_embed
    config_encoder = None  # type: Optional[Config]

    if args.encoder in (C.TRANSFORMER_TYPE, C.TRANSFORMER_WITH_CONV_EMBED_TYPE):
        encoder_transformer_preprocess, _ = args.transformer_preprocess
        encoder_transformer_postprocess, _ = args.transformer_postprocess
        config_encoder = transformer.TransformerConfig(
            model_size=args.transformer_model_size,
            attention_heads=args.transformer_attention_heads,
            feed_forward_num_hidden=args.transformer_feed_forward_num_hidden,
            num_layers=encoder_num_layers,
            dropout_attention=args.transformer_dropout_attention,
            dropout_relu=args.transformer_dropout_relu,
            dropout_prepost=args.transformer_dropout_prepost,
            positional_embedding_type=args.transformer_positional_embedding_type,
            preprocess_sequence=encoder_transformer_preprocess,
            postprocess_sequence=encoder_transformer_postprocess,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            conv_config=config_conv)
        encoder_num_hidden = args.transformer_model_size
    elif args.encoder == C.CONVOLUTION_TYPE:
        cnn_kernel_width_encoder, _ = args.cnn_kernel_width
        cnn_config = convolution.ConvolutionConfig(kernel_width=cnn_kernel_width_encoder,
                                                   num_hidden=args.cnn_num_hidden,
                                                   act_type=args.cnn_activation_type,
                                                   weight_normalization=args.weight_normalization)
        config_encoder = encoder.ConvolutionalEncoderConfig(num_embed=num_embed_source,
                                                            max_seq_len_source=max_seq_len_source,
                                                            cnn_config=cnn_config,
                                                            num_layers=encoder_num_layers,
                                                            positional_embedding_type=args.cnn_positional_embedding_type)

        encoder_num_hidden = args.cnn_num_hidden
    else:
        encoder_rnn_dropout_inputs, _ = args.rnn_dropout_inputs
        encoder_rnn_dropout_states, _ = args.rnn_dropout_states
        encoder_rnn_dropout_recurrent, _ = args.rnn_dropout_recurrent
        config_encoder = encoder.RecurrentEncoderConfig(
            rnn_config=rnn.RNNConfig(cell_type=args.rnn_cell_type,
                                     num_hidden=args.rnn_num_hidden,
                                     num_layers=encoder_num_layers,
                                     dropout_inputs=encoder_rnn_dropout_inputs,
                                     dropout_states=encoder_rnn_dropout_states,
                                     dropout_recurrent=encoder_rnn_dropout_recurrent,
                                     residual=args.rnn_residual_connections,
                                     first_residual_layer=args.rnn_first_residual_layer,
                                     forget_bias=args.rnn_forget_bias),
            conv_config=config_conv,
            reverse_input=args.rnn_encoder_reverse_input)
        encoder_num_hidden = args.rnn_num_hidden

    return config_encoder, encoder_num_hidden


def create_decoder_config(args: argparse.Namespace,  encoder_num_hidden: int) -> Config:
    """
    Create the config for the decoder.

    :param args: Arguments as returned by argparse.
    :return: The config for the decoder.
    """
    _, decoder_num_layers = args.num_layers
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    _, num_embed_target = args.num_embed

    config_decoder = None  # type: Optional[Config]

    if args.decoder == C.TRANSFORMER_TYPE:
        _, decoder_transformer_preprocess = args.transformer_preprocess
        _, decoder_transformer_postprocess = args.transformer_postprocess
        config_decoder = transformer.TransformerConfig(
            model_size=args.transformer_model_size,
            attention_heads=args.transformer_attention_heads,
            feed_forward_num_hidden=args.transformer_feed_forward_num_hidden,
            num_layers=decoder_num_layers,
            dropout_attention=args.transformer_dropout_attention,
            dropout_relu=args.transformer_dropout_relu,
            dropout_prepost=args.transformer_dropout_prepost,
            positional_embedding_type=args.transformer_positional_embedding_type,
            preprocess_sequence=decoder_transformer_preprocess,
            postprocess_sequence=decoder_transformer_postprocess,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            conv_config=None)

    elif args.decoder == C.CONVOLUTION_TYPE:
        _, cnn_kernel_width_decoder = args.cnn_kernel_width
        convolution_config = convolution.ConvolutionConfig(kernel_width=cnn_kernel_width_decoder,
                                                           num_hidden=args.cnn_num_hidden,
                                                           act_type=args.cnn_activation_type,
                                                           weight_normalization=args.weight_normalization)
        config_decoder = decoder.ConvolutionalDecoderConfig(cnn_config=convolution_config,
                                                            max_seq_len_target=max_seq_len_target,
                                                            num_embed=num_embed_target,
                                                            encoder_num_hidden=encoder_num_hidden,
                                                            num_layers=decoder_num_layers,
                                                            positional_embedding_type=args.cnn_positional_embedding_type,
                                                            hidden_dropout=args.cnn_hidden_dropout)

    else:
        rnn_attention_num_hidden = args.rnn_num_hidden if args.rnn_attention_num_hidden is None else args.rnn_attention_num_hidden
        config_coverage = None
        if args.rnn_attention_type == C.ATT_COV:
            config_coverage = coverage.CoverageConfig(type=args.rnn_attention_coverage_type,
                                                      num_hidden=args.rnn_attention_coverage_num_hidden,
                                                      layer_normalization=args.layer_normalization)
        config_attention = rnn_attention.AttentionConfig(type=args.rnn_attention_type,
                                                         num_hidden=rnn_attention_num_hidden,
                                                         input_previous_word=args.rnn_attention_use_prev_word,
                                                         source_num_hidden=encoder_num_hidden,
                                                         query_num_hidden=args.rnn_num_hidden,
                                                         layer_normalization=args.layer_normalization,
                                                         config_coverage=config_coverage,
                                                         num_heads=args.rnn_attention_mhdot_heads)

        _, decoder_rnn_dropout_inputs = args.rnn_dropout_inputs
        _, decoder_rnn_dropout_states = args.rnn_dropout_states
        _, decoder_rnn_dropout_recurrent = args.rnn_dropout_recurrent

        config_decoder = decoder.RecurrentDecoderConfig(
            max_seq_len_source=max_seq_len_source,
            rnn_config=rnn.RNNConfig(cell_type=args.rnn_cell_type,
                                     num_hidden=args.rnn_num_hidden,
                                     num_layers=decoder_num_layers,
                                     dropout_inputs=decoder_rnn_dropout_inputs,
                                     dropout_states=decoder_rnn_dropout_states,
                                     dropout_recurrent=decoder_rnn_dropout_recurrent,
                                     residual=args.rnn_residual_connections,
                                     first_residual_layer=args.rnn_first_residual_layer,
                                     forget_bias=args.rnn_forget_bias),
            attention_config=config_attention,
            hidden_dropout=args.rnn_decoder_hidden_dropout,
            state_init=args.rnn_decoder_state_init,
            context_gating=args.rnn_context_gating,
            layer_normalization=args.layer_normalization,
            attention_in_upper_layers=args.rnn_attention_in_upper_layers)

    return config_decoder


def check_encoder_decoder_args(args) -> None:
    """
    Check possible encoder-decoder argument conflicts.

    :param args: Arguments as returned by argparse.
    """
    encoder_embed_dropout, decoder_embed_dropout = args.embed_dropout
    encoder_rnn_dropout_inputs, decoder_rnn_dropout_inputs = args.rnn_dropout_inputs
    encoder_rnn_dropout_states, decoder_rnn_dropout_states = args.rnn_dropout_states
    if encoder_embed_dropout > 0 and encoder_rnn_dropout_inputs > 0:
        logger.warning("Setting encoder RNN AND source embedding dropout > 0 leads to "
                       "two dropout layers on top of each other.")
    if decoder_embed_dropout > 0 and decoder_rnn_dropout_inputs > 0:
        logger.warning("Setting encoder RNN AND source embedding dropout > 0 leads to "
                       "two dropout layers on top of each other.")
    encoder_rnn_dropout_recurrent, decoder_rnn_dropout_recurrent = args.rnn_dropout_recurrent
    if encoder_rnn_dropout_recurrent > 0 or decoder_rnn_dropout_recurrent > 0:
        check_condition(args.rnn_cell_type == C.LSTM_TYPE,
                        "Recurrent dropout without memory loss only supported for LSTMs right now.")


def create_model_config(args: argparse.Namespace,
                        vocab_source_size: int, vocab_target_size: int,
                        config_data: data_io.DataConfig) -> model.ModelConfig:
    """
    Create a ModelConfig from the argument given in the command line.

    :param args: Arguments as returned by argparse.
    :param vocab_source_size: The size of the source vocabulary.
    :param vocab_target_size: The size of the target vocabulary.
    :param config_data: Data config.
    :return: The model configuration.
    """
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    num_embed_source, num_embed_target = args.num_embed
    embed_dropout_source, embed_dropout_target = args.embed_dropout

    check_encoder_decoder_args(args)

    config_conv = None
    if args.encoder == C.RNN_WITH_CONV_EMBED_NAME:
        config_conv = encoder.ConvolutionalEmbeddingConfig(num_embed=num_embed_source,
                                                           max_filter_width=args.conv_embed_max_filter_width,
                                                           num_filters=args.conv_embed_num_filters,
                                                           pool_stride=args.conv_embed_pool_stride,
                                                           num_highway_layers=args.conv_embed_num_highway_layers,
                                                           dropout=args.conv_embed_dropout)

    config_encoder, encoder_num_hidden = create_encoder_config(args, config_conv)
    config_decoder = create_decoder_config(args, encoder_num_hidden)

    config_embed_source = encoder.EmbeddingConfig(vocab_size=vocab_source_size,
                                                  num_embed=num_embed_source,
                                                  dropout=embed_dropout_source)
    config_embed_target = encoder.EmbeddingConfig(vocab_size=vocab_target_size,
                                                  num_embed=num_embed_target,
                                                  dropout=embed_dropout_target)

    config_loss = loss.LossConfig(name=args.loss,
                                  vocab_size=vocab_target_size,
                                  normalization_type=args.loss_normalization_type,
                                  label_smoothing=args.label_smoothing)

    model_config = model.ModelConfig(config_data=config_data,
                                     max_seq_len_source=max_seq_len_source,
                                     max_seq_len_target=max_seq_len_target,
                                     vocab_source_size=vocab_source_size,
                                     vocab_target_size=vocab_target_size,
                                     config_embed_source=config_embed_source,
                                     config_embed_target=config_embed_target,
                                     config_encoder=config_encoder,
                                     config_decoder=config_decoder,
                                     config_loss=config_loss,
                                     weight_tying=args.weight_tying,
                                     weight_tying_type=args.weight_tying_type if args.weight_tying else None,
                                     weight_normalization=args.weight_normalization)
    return model_config


def create_training_model(model_config: model.ModelConfig,
                          args: argparse.Namespace,
                          context: List[mx.Context],
                          train_iter: data_io.ParallelBucketSentenceIter,
                          lr_scheduler_instance: lr_scheduler.LearningRateScheduler,
                          resume_training: bool,
                          training_state_dir: str) -> training.TrainingModel:
    """
    Create a training model and load the parameters from disk if needed.

    :param model_config: The configuration for the model.
    :param args: Arguments as returned by argparse.
    :param context: The context(s) to run on.
    :param train_iter: The training data iterator.
    :param lr_scheduler: The learning rate scheduler.
    :param resume_training: When True, the model will be loaded from disk.
    :param training_state_dir: Directory where the training state is stored.
    :return: The training model.
    """
    training_model = training.TrainingModel(config=model_config,
                                            context=context,
                                            train_iter=train_iter,
                                            bucketing=not args.no_bucketing,
                                            lr_scheduler=lr_scheduler_instance)

    # We may consider loading the params in TrainingModule, for consistency
    # with the training state saving
    if resume_training:
        logger.info("Found partial training in directory %s. Resuming from saved state.", training_state_dir)
        training_model.load_params_from_file(os.path.join(training_state_dir, C.TRAINING_STATE_PARAMS_NAME))
    elif args.params:
        logger.info("Training will initialize from parameters loaded from '%s'", args.params)
        training_model.load_params_from_file(args.params)

    return training_model


def define_optimizer(args, lr_scheduler_instance) -> Tuple[str, Dict, str]:
    """
    Defines the optimizer to use and its parameters.

    :param args: Arguments as returned by argparse.
    :param lr_scheduler: The learning rate scheduler.
    :return: The optimizer type and its parameters as well as the kvstore.
    """
    optimizer = args.optimizer
    optimizer_params = {'wd': args.weight_decay,
                        "learning_rate": args.initial_learning_rate}
    if lr_scheduler_instance is not None:
        optimizer_params["lr_scheduler"] = lr_scheduler_instance
    clip_gradient = none_if_negative(args.clip_gradient)
    if clip_gradient is not None:
        optimizer_params["clip_gradient"] = clip_gradient
    if args.momentum is not None:
        optimizer_params["momentum"] = args.momentum
    if args.loss_normalization_type == C.LOSS_NORM_VALID:
        # When we normalize by the number of non-PAD symbols in a batch we need to disable rescale_grad.
        optimizer_params["rescale_grad"] = 1.0
    elif args.loss_normalization_type == C.LOSS_NORM_BATCH:
        # Making MXNet module API's default scaling factor explicit
        optimizer_params["rescale_grad"] = 1.0 / args.batch_size
    # Manually specified params
    if args.optimizer_params:
        optimizer_params.update(args.optimizer_params)
    logger.info("Optimizer: %s", optimizer)
    logger.info("Optimizer Parameters: %s", optimizer_params)
    logger.info("kvstore: %s", args.kvstore)

    return optimizer, optimizer_params, args.kvstore


def main():
    params = argparse.ArgumentParser(description='CLI to train sockeye sequence-to-sequence models.')
    arguments.add_train_cli_args(params)
    args = params.parse_args()

    utils.seedRNGs(args)

    check_arg_compatibility(args)
    output_folder = os.path.abspath(args.output)
    resume_training, training_state_dir = check_resume(args, output_folder)

    global logger
    logger = setup_main_logger(__name__,
                               file_logging=True,
                               console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    utils.log_basic_info(args)
    with open(os.path.join(output_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)
        vocab_source, vocab_target = load_or_create_vocabs(args, resume_training, output_folder)
        vocab_source_size = len(vocab_source)
        vocab_target_size = len(vocab_target)
        logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)
        train_iter, eval_iter, config_data = create_data_iters(args, vocab_source, vocab_target)
        lr_scheduler_instance = create_lr_scheduler(args, resume_training, training_state_dir)

        model_config = create_model_config(args, vocab_source_size, vocab_target_size, config_data)
        model_config.freeze()

        training_model = create_training_model(model_config, args,
                                               context, train_iter, lr_scheduler_instance,
                                               resume_training, training_state_dir)

        weight_initializer = initializer.get_initializer(
            default_init_type=args.weight_init,
            default_init_scale=args.weight_init_scale,
            default_init_xavier_factor_type=args.weight_init_xavier_factor_type,
            embed_init_type=args.embed_weight_init,
            embed_init_sigma=vocab_source_size ** -0.5,  # TODO
            rnn_init_type=args.rnn_h2h_init)

        optimizer, optimizer_params, kvstore = define_optimizer(args, lr_scheduler_instance)

        # Handle options that override training settings
        max_updates = args.max_updates
        max_num_checkpoint_not_improved = args.max_num_checkpoint_not_improved
        min_num_epochs = args.min_num_epochs
        max_num_epochs = args.max_num_epochs
        if min_num_epochs is not None and max_num_epochs is not None:
            check_condition(min_num_epochs <= max_num_epochs,
                            "Minimum number of epochs must be smaller than maximum number of epochs")
        # Fixed training schedule always runs for a set number of updates
        if args.learning_rate_schedule:
            max_updates = sum(num_updates for (_, num_updates) in args.learning_rate_schedule)
            max_num_checkpoint_not_improved = -1
            min_num_epochs = None
            max_num_epochs = None

        monitor_bleu = args.monitor_bleu
        # Turn on BLEU monitoring when the optimized metric is BLEU and it hasn't been enabled yet
        if args.optimized_metric == C.BLEU and monitor_bleu == 0:
            logger.info("You chose BLEU as the optimized metric, will turn on BLEU monitoring during training. "
                        "To control how many validation sentences are used for calculating bleu use "
                        "the --monitor-bleu argument.")
            monitor_bleu = -1

        training_model.fit(train_iter, eval_iter,
                           output_folder=output_folder,
                           max_params_files_to_keep=args.keep_last_params,
                           metrics=args.metrics,
                           initializer=weight_initializer,
                           max_updates=max_updates,
                           checkpoint_frequency=args.checkpoint_frequency,
                           optimizer=optimizer, optimizer_params=optimizer_params,
                           optimized_metric=args.optimized_metric,
                           kvstore=kvstore,
                           max_num_not_improved=max_num_checkpoint_not_improved,
                           min_num_epochs=min_num_epochs,
                           max_num_epochs=max_num_epochs,
                           monitor_bleu=monitor_bleu,
                           use_tensorboard=args.use_tensorboard,
                           mxmonitor_pattern=args.monitor_pattern,
                           mxmonitor_stat_func=args.monitor_stat_func,
                           lr_decay_param_reset=args.learning_rate_decay_param_reset,
                           lr_decay_opt_states_reset=args.learning_rate_decay_optimizer_states_reset)


if __name__ == "__main__":
    main()

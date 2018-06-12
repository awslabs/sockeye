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
import shutil
import sys
import tempfile
from contextlib import ExitStack
from typing import Any, cast, Optional, Dict, List, Tuple

import mxnet as mx

from . import arguments
from . import checkpoint_decoder
from . import constants as C
from . import convolution
from . import coverage
from . import data_io
from . import decoder
from . import encoder
from . import initializer
from . import loss
from . import lr_scheduler
from . import model
from . import rnn
from . import rnn_attention
from . import training
from . import transformer
from . import utils
from . import vocab
from .config import Config
from .log import setup_main_logger
from .optimizers import OptimizerConfig
from .utils import check_condition

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = setup_main_logger(__name__, file_logging=False, console=True)


def none_if_negative(val):
    return None if val < 0 else val


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
        check_condition(args.transformer_model_size[0] == args.num_embed[0],
                        "Source embedding size must match transformer model size: %s vs. %s"
                        % (args.transformer_model_size, args.num_embed[0]))

        total_source_factor_size = sum(args.source_factors_num_embed)
        if total_source_factor_size > 0:
            adjusted_transformer_encoder_model_size = args.num_embed[0] + total_source_factor_size
            check_condition(adjusted_transformer_encoder_model_size % 2 == 0 and
                            adjusted_transformer_encoder_model_size % args.transformer_attention_heads[0] == 0,
                            "Sum of source factor sizes, i.e. num-embed plus source-factors-num-embed, (%d) "
                            "has to be even and a multiple of encoder attention heads (%d)" % (
                                adjusted_transformer_encoder_model_size, args.transformer_attention_heads[0]))

    if args.decoder == C.TRANSFORMER_TYPE:
        check_condition(args.transformer_model_size[1] == args.num_embed[1],
                        "Target embedding size must match transformer model size: %s vs. %s"
                        % (args.transformer_model_size, args.num_embed[1]))

    if args.lhuc is not None:
        # Actually this check is a bit too strict
        check_condition(args.encoder != C.CONVOLUTION_TYPE or args.decoder != C.CONVOLUTION_TYPE,
                        "LHUC is not supported for convolutional models yet.")
        check_condition(args.decoder != C.TRANSFORMER_TYPE or C.LHUC_STATE_INIT not in args.lhuc,
                        "The %s options only applies to RNN models" % C.LHUC_STATE_INIT)


def check_resume(args: argparse.Namespace, output_folder: str) -> bool:
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
            old_args = vars(arguments.load_args(os.path.join(output_folder, C.ARGS_STATE_NAME)))
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

    return resume_training


def determine_context(args: argparse.Namespace, exit_stack: ExitStack) -> List[mx.Context]:
    """
    Determine the context we should run on (CPU or GPU).

    :param args: Arguments as returned by argparse.
    :param exit_stack: An ExitStack from contextlib.
    :return: A list with the context(s) to run on.
    """
    if args.use_cpu:
        logger.info("Training Device: CPU")
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
        logger.info("Training Device(s): GPU %s", context)
        context = [mx.gpu(gpu_id) for gpu_id in context]
    return context


def create_checkpoint_decoder(args: argparse.Namespace,
                              exit_stack: ExitStack,
                              train_context: List[mx.Context]) -> Optional[checkpoint_decoder.CheckpointDecoder]:
    """
    Returns a checkpoint decoder or None.

    :param args: Arguments as returned by argparse.
    :param exit_stack: An ExitStack from contextlib.
    :param train_context: Context for training.
    :return: A CheckpointDecoder if --decode-and-evaluate != 0, else None.
    """
    sample_size = args.decode_and_evaluate
    if args.optimized_metric == C.BLEU and sample_size == 0:
        logger.info("You chose BLEU as the optimized metric, will turn on BLEU monitoring during training. "
                    "To control how many validation sentences are used for calculating bleu use "
                    "the --decode-and-evaluate argument.")
        sample_size = -1

    if sample_size == 0:
        return None

    if args.use_cpu or args.decode_and_evaluate_use_cpu:
        context = mx.cpu()
    elif args.decode_and_evaluate_device_id is not None:
        # decode device is defined from the commandline
        num_gpus = utils.get_num_gpus()
        check_condition(num_gpus >= 1,
                        "No GPUs found, consider running on the CPU with --use-cpu "
                        "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi "
                        "binary isn't on the path).")

        if args.disable_device_locking:
            context = utils.expand_requested_device_ids([args.decode_and_evaluate_device_id])
        else:
            context = exit_stack.enter_context(utils.acquire_gpus([args.decode_and_evaluate_device_id],
                                                                  lock_dir=args.lock_dir))
        context = mx.gpu(context[0])

    else:
        # default decode context is the last training device
        context = train_context[-1]

    return checkpoint_decoder.CheckpointDecoder(context=context,
                                                inputs=[args.validation_source] + args.validation_source_factors,
                                                references=args.validation_target,
                                                model=args.output,
                                                sample_size=sample_size)


def use_shared_vocab(args: argparse.Namespace) -> bool:
    """
    True if arguments entail a shared source and target vocabulary.

    :param: args: Arguments as returned by argparse.
    """
    weight_tying = args.weight_tying
    weight_tying_type = args.weight_tying_type
    shared_vocab = args.shared_vocab
    if weight_tying and C.WEIGHT_TYING_SRC in weight_tying_type and C.WEIGHT_TYING_TRG in weight_tying_type:
        if not shared_vocab:
            logger.info("A shared source/target vocabulary will be used as weight tying source/target weight tying "
                        "is enabled")
        shared_vocab = True
    return shared_vocab


def create_data_iters_and_vocabs(args: argparse.Namespace,
                                 max_seq_len_source: int,
                                 max_seq_len_target: int,
                                 shared_vocab: bool,
                                 resume_training: bool,
                                 output_folder: str) -> Tuple['data_io.BaseParallelSampleIter',
                                                              'data_io.BaseParallelSampleIter',
                                                              'data_io.DataConfig',
                                                              List[vocab.Vocab], vocab.Vocab]:
    """
    Create the data iterators and the vocabularies.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Source maximum sequence length.
    :param max_seq_len_target: Target maximum sequence length.
    :param shared_vocab: Whether to create a shared vocabulary.
    :param resume_training: Whether to resume training.
    :param output_folder: Output folder.
    :return: The data iterators (train, validation, config_data) as well as the source and target vocabularies.
    """
    num_words_source, num_words_target = args.num_words
    word_min_count_source, word_min_count_target = args.word_min_count
    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    batch_by_words = args.batch_type == C.BATCH_TYPE_WORD

    validation_sources = [args.validation_source] + args.validation_source_factors
    validation_sources = [str(os.path.abspath(source)) for source in validation_sources]

    either_raw_or_prepared_error_msg = "Either specify a raw training corpus with %s and %s or a preprocessed corpus " \
                                       "with %s." % (C.TRAINING_ARG_SOURCE,
                                                     C.TRAINING_ARG_TARGET,
                                                     C.TRAINING_ARG_PREPARED_DATA)
    if args.prepared_data is not None:
        utils.check_condition(args.source is None and args.target is None, either_raw_or_prepared_error_msg)
        if not resume_training:
            utils.check_condition(args.source_vocab is None and args.target_vocab is None,
                                  "You are using a prepared data folder, which is tied to a vocabulary. "
                                  "To change it you need to rerun data preparation with a different vocabulary.")
        train_iter, validation_iter, data_config, source_vocabs, target_vocab = data_io.get_prepared_data_iters(
            prepared_data_dir=args.prepared_data,
            validation_sources=validation_sources,
            validation_target=str(os.path.abspath(args.validation_target)),
            shared_vocab=shared_vocab,
            batch_size=args.batch_size,
            batch_by_words=batch_by_words,
            batch_num_devices=batch_num_devices,
            fill_up=args.fill_up)

        check_condition(len(source_vocabs) == len(args.source_factors_num_embed) + 1,
                        "Data was prepared with %d source factors, but only provided %d source factor dimensions." % (
                            len(source_vocabs), len(args.source_factors_num_embed) + 1))

        if resume_training:
            # resuming training. Making sure the vocabs in the model and in the prepared data match up
            model_source_vocabs = vocab.load_source_vocabs(output_folder)
            for i, (v, mv) in enumerate(zip(source_vocabs, model_source_vocabs)):
                utils.check_condition(vocab.are_identical(v, mv),
                                      "Prepared data and resumed model source vocab %d do not match." % i)
            model_target_vocab = vocab.load_target_vocab(output_folder)
            utils.check_condition(vocab.are_identical(target_vocab, model_target_vocab),
                                  "Prepared data and resumed model target vocabs do not match.")

            check_condition(len(args.source_factors) == len(args.validation_source_factors),
                            'Training and validation data must have the same number of factors: %d vs. %d.' % (
                                len(args.source_factors), len(args.validation_source_factors)))

        return train_iter, validation_iter, data_config, source_vocabs, target_vocab

    else:
        utils.check_condition(args.prepared_data is None and args.source is not None and args.target is not None,
                              either_raw_or_prepared_error_msg)

        if resume_training:
            # Load the existing vocabs created when starting the training run.
            source_vocabs = vocab.load_source_vocabs(output_folder)
            target_vocab = vocab.load_target_vocab(output_folder)

            # Recover the vocabulary path from the data info file:
            data_info = cast(data_io.DataInfo, Config.load(os.path.join(output_folder, C.DATA_INFO)))
            source_vocab_paths = data_info.source_vocabs
            target_vocab_path = data_info.target_vocab

        else:
            # Load or create vocabs
            source_vocab_paths = [args.source_vocab] + [None] * len(args.source_factors)
            target_vocab_path = args.target_vocab
            source_vocabs, target_vocab = vocab.load_or_create_vocabs(
                source_paths=[args.source] + args.source_factors,
                target_path=args.target,
                source_vocab_paths=source_vocab_paths,
                target_vocab_path=target_vocab_path,
                shared_vocab=shared_vocab,
                num_words_source=num_words_source,
                num_words_target=num_words_target,
                word_min_count_source=word_min_count_source,
                word_min_count_target=word_min_count_target)

        check_condition(len(args.source_factors) == len(args.source_factors_num_embed),
                        "Number of source factor data (%d) differs from provided source factor dimensions (%d)" % (
                            len(args.source_factors), len(args.source_factors_num_embed)))

        sources = [args.source] + args.source_factors
        sources = [str(os.path.abspath(source)) for source in sources]

        train_iter, validation_iter, config_data, data_info = data_io.get_training_data_iters(
            sources=sources,
            target=os.path.abspath(args.target),
            validation_sources=validation_sources,
            validation_target=os.path.abspath(args.validation_target),
            source_vocabs=source_vocabs,
            target_vocab=target_vocab,
            source_vocab_paths=source_vocab_paths,
            target_vocab_path=target_vocab_path,
            shared_vocab=shared_vocab,
            batch_size=args.batch_size,
            batch_by_words=batch_by_words,
            batch_num_devices=batch_num_devices,
            fill_up=args.fill_up,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            bucketing=not args.no_bucketing,
            bucket_width=args.bucket_width)

        data_info_fname = os.path.join(output_folder, C.DATA_INFO)
        logger.info("Writing data config to '%s'", data_info_fname)
        data_info.save(data_info_fname)

        return train_iter, validation_iter, config_data, source_vocabs, target_vocab


def create_encoder_config(args: argparse.Namespace,
                          max_seq_len_source: int,
                          max_seq_len_target: int,
                          config_conv: Optional[encoder.ConvolutionalEmbeddingConfig]) -> Tuple[encoder.EncoderConfig,
                                                                                                int]:
    """
    Create the encoder config.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param config_conv: The config for the convolutional encoder (optional).
    :return: The encoder config and the number of hidden units of the encoder.
    """
    encoder_num_layers, _ = args.num_layers
    num_embed_source, _ = args.num_embed
    config_encoder = None  # type: Optional[Config]

    if args.encoder in (C.TRANSFORMER_TYPE, C.TRANSFORMER_WITH_CONV_EMBED_TYPE):
        encoder_transformer_preprocess, _ = args.transformer_preprocess
        encoder_transformer_postprocess, _ = args.transformer_postprocess
        encoder_transformer_model_size = args.transformer_model_size[0]

        total_source_factor_size = sum(args.source_factors_num_embed)
        if total_source_factor_size > 0:
            logger.info("Encoder transformer-model-size adjusted to account source factor embeddings: %d -> %d" % (
                encoder_transformer_model_size, num_embed_source + total_source_factor_size))
            encoder_transformer_model_size = num_embed_source + total_source_factor_size
        config_encoder = transformer.TransformerConfig(
            model_size=encoder_transformer_model_size,
            attention_heads=args.transformer_attention_heads[0],
            feed_forward_num_hidden=args.transformer_feed_forward_num_hidden[0],
            act_type=args.transformer_activation_type,
            num_layers=encoder_num_layers,
            dropout_attention=args.transformer_dropout_attention,
            dropout_act=args.transformer_dropout_act,
            dropout_prepost=args.transformer_dropout_prepost,
            positional_embedding_type=args.transformer_positional_embedding_type,
            preprocess_sequence=encoder_transformer_preprocess,
            postprocess_sequence=encoder_transformer_postprocess,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            conv_config=config_conv,
            lhuc=args.lhuc is not None and (C.LHUC_ENCODER in args.lhuc or C.LHUC_ALL in args.lhuc))
        encoder_num_hidden = encoder_transformer_model_size
    elif args.encoder == C.CONVOLUTION_TYPE:
        cnn_kernel_width_encoder, _ = args.cnn_kernel_width
        cnn_config = convolution.ConvolutionConfig(kernel_width=cnn_kernel_width_encoder,
                                                   num_hidden=args.cnn_num_hidden,
                                                   act_type=args.cnn_activation_type,
                                                   weight_normalization=args.weight_normalization)
        cnn_num_embed = num_embed_source + sum(args.source_factors_num_embed)
        config_encoder = encoder.ConvolutionalEncoderConfig(num_embed=cnn_num_embed,
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
                                     forget_bias=args.rnn_forget_bias,
                                     lhuc=args.lhuc is not None and (C.LHUC_ENCODER in args.lhuc or C.LHUC_ALL in args.lhuc)),
            conv_config=config_conv,
            reverse_input=args.rnn_encoder_reverse_input)
        encoder_num_hidden = args.rnn_num_hidden

    return config_encoder, encoder_num_hidden


def create_decoder_config(args: argparse.Namespace, encoder_num_hidden: int,
                          max_seq_len_source: int, max_seq_len_target: int) -> decoder.DecoderConfig:
    """
    Create the config for the decoder.

    :param args: Arguments as returned by argparse.
    :param encoder_num_hidden: Number of hidden units of the Encoder.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :return: The config for the decoder.
    """
    _, decoder_num_layers = args.num_layers
    _, num_embed_target = args.num_embed

    config_decoder = None  # type: Optional[Config]

    if args.decoder == C.TRANSFORMER_TYPE:
        _, decoder_transformer_preprocess = args.transformer_preprocess
        _, decoder_transformer_postprocess = args.transformer_postprocess
        config_decoder = transformer.TransformerConfig(
            model_size=args.transformer_model_size[1],
            attention_heads=args.transformer_attention_heads[1],
            feed_forward_num_hidden=args.transformer_feed_forward_num_hidden[1],
            act_type=args.transformer_activation_type,
            num_layers=decoder_num_layers,
            dropout_attention=args.transformer_dropout_attention,
            dropout_act=args.transformer_dropout_act,
            dropout_prepost=args.transformer_dropout_prepost,
            positional_embedding_type=args.transformer_positional_embedding_type,
            preprocess_sequence=decoder_transformer_preprocess,
            postprocess_sequence=decoder_transformer_postprocess,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            conv_config=None,
            lhuc=args.lhuc is not None and (C.LHUC_DECODER in args.lhuc or C.LHUC_ALL in args.lhuc))

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
                                                            project_qkv=args.cnn_project_qkv,
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
                                                         num_heads=args.rnn_attention_mhdot_heads,
                                                         is_scaled=args.rnn_scale_dot_attention)

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
                                     forget_bias=args.rnn_forget_bias,
                                     lhuc=args.lhuc is not None and (C.LHUC_DECODER in args.lhuc or C.LHUC_ALL in args.lhuc)),
            attention_config=config_attention,
            hidden_dropout=args.rnn_decoder_hidden_dropout,
            state_init=args.rnn_decoder_state_init,
            context_gating=args.rnn_context_gating,
            layer_normalization=args.layer_normalization,
            attention_in_upper_layers=args.rnn_attention_in_upper_layers,
            state_init_lhuc=args.lhuc is not None and (C.LHUC_STATE_INIT in args.lhuc or C.LHUC_ALL in args.lhuc),
            enc_last_hidden_concat_to_embedding=args.rnn_enc_last_hidden_concat_to_embedding)

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
                        source_vocab_sizes: List[int],
                        target_vocab_size: int,
                        max_seq_len_source: int,
                        max_seq_len_target: int,
                        config_data: data_io.DataConfig) -> model.ModelConfig:
    """
    Create a ModelConfig from the argument given in the command line.

    :param args: Arguments as returned by argparse.
    :param source_vocab_sizes: The size of the source vocabulary (and source factors).
    :param target_vocab_size: The size of the target vocabulary.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param config_data: Data config.
    :return: The model configuration.
    """
    num_embed_source, num_embed_target = args.num_embed
    embed_dropout_source, embed_dropout_target = args.embed_dropout
    source_vocab_size, *source_factor_vocab_sizes = source_vocab_sizes

    check_encoder_decoder_args(args)

    config_conv = None
    if args.encoder == C.RNN_WITH_CONV_EMBED_NAME:
        config_conv = encoder.ConvolutionalEmbeddingConfig(num_embed=num_embed_source,
                                                           max_filter_width=args.conv_embed_max_filter_width,
                                                           num_filters=args.conv_embed_num_filters,
                                                           pool_stride=args.conv_embed_pool_stride,
                                                           num_highway_layers=args.conv_embed_num_highway_layers,
                                                           dropout=args.conv_embed_dropout)
    if args.encoder == C.TRANSFORMER_WITH_CONV_EMBED_TYPE:
        config_conv = encoder.ConvolutionalEmbeddingConfig(num_embed=num_embed_source,
                                                           output_dim=num_embed_source,
                                                           max_filter_width=args.conv_embed_max_filter_width,
                                                           num_filters=args.conv_embed_num_filters,
                                                           pool_stride=args.conv_embed_pool_stride,
                                                           num_highway_layers=args.conv_embed_num_highway_layers,
                                                           dropout=args.conv_embed_dropout)

    config_encoder, encoder_num_hidden = create_encoder_config(args, max_seq_len_source, max_seq_len_target,
                                                               config_conv)
    config_decoder = create_decoder_config(args, encoder_num_hidden, max_seq_len_source, max_seq_len_target)

    source_factor_configs = None
    if len(source_vocab_sizes) > 1:
        source_factor_configs = [encoder.FactorConfig(size, dim) for size, dim in zip(source_factor_vocab_sizes,
                                                                                      args.source_factors_num_embed)]

    config_embed_source = encoder.EmbeddingConfig(vocab_size=source_vocab_size,
                                                  num_embed=num_embed_source,
                                                  dropout=embed_dropout_source,
                                                  factor_configs=source_factor_configs)

    config_embed_target = encoder.EmbeddingConfig(vocab_size=target_vocab_size,
                                                  num_embed=num_embed_target,
                                                  dropout=embed_dropout_target)

    config_loss = loss.LossConfig(name=args.loss,
                                  vocab_size=target_vocab_size,
                                  normalization_type=args.loss_normalization_type,
                                  label_smoothing=args.label_smoothing)

    model_config = model.ModelConfig(config_data=config_data,
                                     vocab_source_size=source_vocab_size,
                                     vocab_target_size=target_vocab_size,
                                     config_embed_source=config_embed_source,
                                     config_embed_target=config_embed_target,
                                     config_encoder=config_encoder,
                                     config_decoder=config_decoder,
                                     config_loss=config_loss,
                                     weight_tying=args.weight_tying,
                                     weight_tying_type=args.weight_tying_type if args.weight_tying else None,
                                     weight_normalization=args.weight_normalization,
                                     lhuc=args.lhuc is not None)
    return model_config


def create_training_model(config: model.ModelConfig,
                          context: List[mx.Context],
                          output_dir: str,
                          train_iter: data_io.BaseParallelSampleIter,
                          args: argparse.Namespace) -> training.TrainingModel:
    """
    Create a training model and load the parameters from disk if needed.

    :param config: The configuration for the model.
    :param context: The context(s) to run on.
    :param output_dir: Output folder.
    :param train_iter: The training data iterator.
    :param args: Arguments as returned by argparse.
    :return: The training model.
    """
    training_model = training.TrainingModel(config=config,
                                            context=context,
                                            output_dir=output_dir,
                                            provide_data=train_iter.provide_data,
                                            provide_label=train_iter.provide_label,
                                            default_bucket_key=train_iter.default_bucket_key,
                                            bucketing=not args.no_bucketing,
                                            gradient_compression_params=gradient_compression_params(args),
                                            fixed_param_names=args.fixed_param_names)

    return training_model


def gradient_compression_params(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """
    :param args: Arguments as returned by argparse.
    :return: Gradient compression parameters or None.
    """
    if args.gradient_compression_type is None:
        return None
    else:
        return {'type': args.gradient_compression_type, 'threshold': args.gradient_compression_threshold}


def create_optimizer_config(args: argparse.Namespace, source_vocab_sizes: List[int],
                            extra_initializers: List[Tuple[str, mx.initializer.Initializer]] = None) -> OptimizerConfig:
    """
    Returns an OptimizerConfig.

    :param args: Arguments as returned by argparse.
    :param source_vocab_sizes: Source vocabulary sizes.
    :param extra_initializers: extra initializer to pass to `get_initializer`.
    :return: The optimizer type and its parameters as well as the kvstore.
    """
    optimizer_params = {'wd': args.weight_decay,
                        "learning_rate": args.initial_learning_rate}

    gradient_clipping_threshold = none_if_negative(args.gradient_clipping_threshold)
    if gradient_clipping_threshold is None:
        logger.info("Gradient clipping threshold set to negative value. Will not perform gradient clipping.")
        gradient_clipping_type = C.GRADIENT_CLIPPING_TYPE_NONE
    else:
        gradient_clipping_type = args.gradient_clipping_type

    # Note: for 'abs' we use the implementation inside of MXNet's optimizer and 'norm_*' we implement ourselves
    # inside the TrainingModel.
    if gradient_clipping_threshold is not None and gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_ABS:
        optimizer_params["clip_gradient"] = gradient_clipping_threshold
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

    weight_init = initializer.get_initializer(default_init_type=args.weight_init,
                                              default_init_scale=args.weight_init_scale,
                                              default_init_xavier_rand_type=args.weight_init_xavier_rand_type,
                                              default_init_xavier_factor_type=args.weight_init_xavier_factor_type,
                                              embed_init_type=args.embed_weight_init,
                                              embed_init_sigma=source_vocab_sizes[0] ** -0.5,
                                              rnn_init_type=args.rnn_h2h_init,
                                              extra_initializers=extra_initializers)

    lr_sched = lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type,
                                             args.checkpoint_frequency,
                                             none_if_negative(args.learning_rate_half_life),
                                             args.learning_rate_reduce_factor,
                                             args.learning_rate_reduce_num_not_improved,
                                             args.learning_rate_schedule,
                                             args.learning_rate_warmup)

    config = OptimizerConfig(name=args.optimizer,
                             params=optimizer_params,
                             kvstore=args.kvstore,
                             initializer=weight_init,
                             gradient_clipping_type=gradient_clipping_type,
                             gradient_clipping_threshold=gradient_clipping_threshold)
    config.set_lr_scheduler(lr_sched)
    logger.info("Optimizer: %s", config)
    logger.info("Gradient Compression: %s", gradient_compression_params(args))
    return config


def main():
    params = arguments.ConfigArgumentParser(description='Train Sockeye sequence-to-sequence models.')
    arguments.add_train_cli_args(params)
    args = params.parse_args()
    train(args)


def train(args: argparse.Namespace):
    if args.dry_run:
        # Modify arguments so that we write to a temporary directory and
        # perform 0 training iterations
        temp_dir = tempfile.TemporaryDirectory()  # Will be automatically removed
        args.output = temp_dir.name
        args.max_updates = 0

    utils.seedRNGs(args.seed)

    check_arg_compatibility(args)
    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    global logger
    logger = setup_main_logger(__name__,
                               file_logging=True,
                               console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    utils.log_basic_info(args)
    arguments.save_args(args, os.path.join(output_folder, C.ARGS_STATE_NAME))

    max_seq_len_source, max_seq_len_target = args.max_seq_len
    # The maximum length is the length before we add the BOS/EOS symbols
    max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
    max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
    logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)",
                max_seq_len_source, max_seq_len_target)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        train_iter, eval_iter, config_data, source_vocabs, target_vocab = create_data_iters_and_vocabs(
            args=args,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            shared_vocab=use_shared_vocab(args),
            resume_training=resume_training,
            output_folder=output_folder)
        max_seq_len_source = config_data.max_seq_len_source
        max_seq_len_target = config_data.max_seq_len_target

        # Dump the vocabularies if we're just starting up
        if not resume_training:
            vocab.save_source_vocabs(source_vocabs, output_folder)
            vocab.save_target_vocab(target_vocab, output_folder)

        source_vocab_sizes = [len(v) for v in source_vocabs]
        target_vocab_size = len(target_vocab)
        logger.info('Vocabulary sizes: source=[%s] target=%d',
                    '|'.join([str(size) for size in source_vocab_sizes]),
                    target_vocab_size)

        model_config = create_model_config(args=args,
                                           source_vocab_sizes=source_vocab_sizes, target_vocab_size=target_vocab_size,
                                           max_seq_len_source=max_seq_len_source, max_seq_len_target=max_seq_len_target,
                                           config_data=config_data)
        model_config.freeze()

        training_model = create_training_model(config=model_config,
                                               context=context,
                                               output_dir=output_folder,
                                               train_iter=train_iter,
                                               args=args)

        # Handle options that override training settings
        min_updates = args.min_updates
        max_updates = args.max_updates
        min_samples = args.min_samples
        max_samples = args.max_samples
        max_num_checkpoint_not_improved = args.max_num_checkpoint_not_improved
        min_epochs = args.min_num_epochs
        max_epochs = args.max_num_epochs
        if min_epochs is not None and max_epochs is not None:
            check_condition(min_epochs <= max_epochs,
                            "Minimum number of epochs must be smaller than maximum number of epochs")
        # Fixed training schedule always runs for a set number of updates
        if args.learning_rate_schedule:
            min_updates = None
            max_updates = sum(num_updates for (_, num_updates) in args.learning_rate_schedule)
            max_num_checkpoint_not_improved = -1
            min_samples = None
            max_samples = None
            min_epochs = None
            max_epochs = None

        trainer = training.EarlyStoppingTrainer(model=training_model,
                                                optimizer_config=create_optimizer_config(args, source_vocab_sizes),
                                                max_params_files_to_keep=args.keep_last_params,
                                                source_vocabs=source_vocabs,
                                                target_vocab=target_vocab)

        trainer.fit(train_iter=train_iter,
                    validation_iter=eval_iter,
                    early_stopping_metric=args.optimized_metric,
                    metrics=args.metrics,
                    checkpoint_frequency=args.checkpoint_frequency,
                    max_num_not_improved=max_num_checkpoint_not_improved,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    min_updates=min_updates,
                    max_updates=max_updates,
                    min_epochs=min_epochs,
                    max_epochs=max_epochs,
                    lr_decay_param_reset=args.learning_rate_decay_param_reset,
                    lr_decay_opt_states_reset=args.learning_rate_decay_optimizer_states_reset,
                    decoder=create_checkpoint_decoder(args, exit_stack, context),
                    mxmonitor_pattern=args.monitor_pattern,
                    mxmonitor_stat_func=args.monitor_stat_func,
                    allow_missing_parameters=args.allow_missing_params or model_config.lhuc,
                    existing_parameters=args.params)


if __name__ == "__main__":
    main()

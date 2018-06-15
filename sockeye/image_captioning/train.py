# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Training CLI for image captioning.
"""
import argparse
import json
import os
import pickle
from contextlib import ExitStack
from typing import cast, Dict, List, Tuple, Optional

import mxnet as mx
import numpy as np

from ..config import Config
from ..log import setup_main_logger
from ..train import check_resume, check_arg_compatibility, \
    determine_context, create_decoder_config, \
    create_optimizer_config, create_training_model
from ..utils import check_condition
# Sockeye captioner
from . import arguments as arguments_image
from . import checkpoint_decoder
from . import data_io as data_io_image
from . import encoder as encoder_image
from .. import constants as C
# Sockeye
from .. import arguments
from .. import data_io
from .. import encoder
from .. import loss
from .. import model
from .. import training
from .. import utils
from .. import vocab

# Temporary logger, the real one (logging to a file probably, will be created in the main function)
logger = setup_main_logger(__name__, file_logging=False, console=True)


def read_feature_shape(path):
    shape_file = os.path.join(path, "image_feature_sizes.pkl")
    with open(shape_file, "rb") as fout:
        shapes = pickle.load(fout)
    return shapes["image_shape"], shapes["features_shape"]


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

    return checkpoint_decoder.CheckpointDecoderImageModel(context=context,
                                                inputs=[args.validation_source] + args.validation_source_factors,
                                                references=args.validation_target,
                                                model=args.output,
                                                sample_size=sample_size,
                                                source_image_size=args.source_image_size,
                                                image_root=args.validation_source_root,
                                                max_output_length=args.max_output_length,
                                                use_feature_loader=args.image_preextracted_features)


def create_data_iters_and_vocab(args: argparse.Namespace,
                                max_seq_len_source: int,
                                max_seq_len_target: int,
                                resume_training: bool,
                                output_folder: str) -> Tuple['data_io.BaseParallelSampleIter',
                                                             'data_io.BaseParallelSampleIter',
                                                             'data_io.DataConfig', Dict]:
    """
    Create the data iterators and the vocabularies.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Source maximum sequence length.
    :param max_seq_len_target: Target maximum sequence length.
    :param resume_training: Whether to resume training.
    :param output_folder: Output folder.
    :return: The data iterators (train, validation, config_data) as well as the source and target vocabularies.
    """

    _, num_words_target = args.num_words
    _, word_min_count_target = args.word_min_count
    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    batch_by_words = args.batch_type == C.BATCH_TYPE_WORD

    either_raw_or_prepared_error_msg = "Either specify a raw training corpus with %s or a preprocessed corpus " \
                                       "with %s." % (C.TRAINING_ARG_TARGET,
                                                     C.TRAINING_ARG_PREPARED_DATA)
    # Note: ignore args.prepared_data for the moment
    utils.check_condition(args.prepared_data is None and args.target is not None,
                          either_raw_or_prepared_error_msg)

    if resume_training:
        # Load the existing vocab created when starting the training run.
        target_vocab = vocab.vocab_from_json(os.path.join(output_folder, C.VOCAB_TRG_NAME))

        # Recover the vocabulary path from the existing config file:
        data_info = cast(data_io.DataInfo, Config.load(os.path.join(output_folder, C.DATA_INFO)))
        target_vocab_path = data_info.target_vocab
    else:
        # Load vocab:
        target_vocab_path = args.target_vocab
        # Note: We do not care about the source vocab for images, that is why some inputs are mocked
        target_vocab = vocab.load_or_create_vocab(data=args.target,
                                                  vocab_path=target_vocab_path,
                                                  num_words=num_words_target,
                                                  word_min_count=word_min_count_target)

    train_iter, validation_iter, config_data, data_info = data_io_image.get_training_image_text_data_iters(
        source_root=args.source_root,
        source=os.path.abspath(args.source),
        target=os.path.abspath(args.target),
        validation_source_root=args.validation_source_root,
        validation_source=os.path.abspath(args.validation_source),
        validation_target=os.path.abspath(args.validation_target),
        vocab_target=target_vocab,
        vocab_target_path=target_vocab_path,
        batch_size=args.batch_size,
        batch_by_words=batch_by_words,
        batch_num_devices=batch_num_devices,
        source_image_size=args.source_image_size,
        fill_up=args.fill_up,
        max_seq_len_target=max_seq_len_target,
        bucketing=not args.no_bucketing,
        bucket_width=args.bucket_width,
        use_feature_loader=args.image_preextracted_features,
        preload_features=args.load_all_features_to_memory
    )

    data_info_fname = os.path.join(output_folder, C.DATA_INFO)
    logger.info("Writing data config to '%s'", data_info_fname)
    # Removing objects that cannot be saved:
    data_info.sources = None
    data_info.save(data_info_fname)

    return train_iter, validation_iter, config_data, target_vocab


def create_encoder_config(args: argparse.Namespace) -> Tuple[Config, int]:

    if args.encoder == C.IMAGE_PRETRAIN_TYPE:
        number_of_kernels = args.source_image_size[0]
        encoded_seq_len = np.prod(args.source_image_size[1:])
        config_encoder = encoder_image.ImageLoadedCnnEncoderConfig(model_path=args.image_encoder_model_path,
                                                          epoch=args.image_encoder_model_epoch,
                                                          layer_name=args.image_encoder_layer,
                                                          encoded_seq_len=encoded_seq_len,
                                                          num_embed=args.image_encoder_num_hidden,
                                                          no_global_descriptor=args.no_image_encoder_global_descriptor,
                                                          preextracted_features=args.image_preextracted_features,
                                                          number_of_kernels=number_of_kernels,
                                                          positional_embedding_type=args.image_positional_embedding_type)
        encoder_num_hidden = args.image_encoder_num_hidden
    else:
        raise ValueError("Image encoder must be provided. (current: {}, "
                         "expected: {})".format(args.encoder, C.ENCODERS))
    return config_encoder, encoder_num_hidden


def create_model_config(args: argparse.Namespace,
                        vocab_target_size: int,
                        max_seq_len_source: int,
                        max_seq_len_target: int,
                        config_data: data_io.DataConfig) -> model.ModelConfig:
    """
    Create a ModelConfig from the argument given in the command line.

    :param args: Arguments as returned by argparse.
    :param vocab_target_size: The size of the target vocabulary.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param config_data: Data config.
    :return: The model configuration.
    """
    num_embed_source, num_embed_target = args.num_embed
    _, embed_dropout_target = args.embed_dropout

    config_encoder, encoder_num_hidden = create_encoder_config(args)
    config_decoder = create_decoder_config(args, encoder_num_hidden, max_seq_len_source, max_seq_len_target)

    config_embed_source = encoder.PassThroughEmbeddingConfig()
    config_embed_target = encoder.EmbeddingConfig(vocab_size=vocab_target_size,
                                                  num_embed=num_embed_target,
                                                  dropout=embed_dropout_target)

    config_loss = loss.LossConfig(name=args.loss,
                                  vocab_size=vocab_target_size,
                                  normalization_type=args.loss_normalization_type,
                                  label_smoothing=args.label_smoothing)

    model_config = model.ModelConfig(config_data=config_data,
                                     vocab_source_size=0,
                                     vocab_target_size=vocab_target_size,
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


def get_preinit_encoders(encoders: List[encoder.Encoder]) -> List[Tuple[str, mx.init.Initializer]]:
    """
    Get initializers from encoders. Some encoders might be initialized from pretrained models.

    :param encoders: List of encoders
    :return: The list of initializers
    """
    init = []
    for enc in encoders:
        if hasattr(enc, "get_initializers"):
            init.extend(enc.get_initializers())
    return init


def main():
    params = arguments.ConfigArgumentParser(description='Train Sockeye images-to-text models.')
    arguments_image.add_image_train_cli_args(params)
    args = params.parse_args()
    # TODO: make training compatible with full net
    args.image_preextracted_features = True  # override this for now

    utils.seedRNGs(args.seed)

    check_arg_compatibility(args)
    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    global logger
    logger = setup_main_logger(__name__,
                               file_logging=True,
                               console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    utils.log_basic_info(args)
    with open(os.path.join(output_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)

    max_seq_len_source, max_seq_len_target = args.max_seq_len
    # The maximum length is the length before we add the BOS/EOS symbols
    max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
    max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
    logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)",
                max_seq_len_source, max_seq_len_target)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        # Read feature size
        if args.image_preextracted_features:
            _, args.source_image_size = read_feature_shape(args.source_root)

        train_iter, eval_iter, config_data, target_vocab = create_data_iters_and_vocab(
            args=args,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            resume_training=resume_training,
            output_folder=output_folder)
        max_seq_len_source = config_data.max_seq_len_source
        max_seq_len_target = config_data.max_seq_len_target

        # Dump the vocabularies if we're just starting up
        if not resume_training:
            vocab.vocab_to_json(target_vocab, os.path.join(output_folder, C.VOCAB_TRG_NAME))

        target_vocab_size = len(target_vocab)
        logger.info("Vocabulary sizes: target=%d", target_vocab_size)

        model_config = create_model_config(args=args,
                                           vocab_target_size=target_vocab_size,
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

        # Get initialization from encoders (useful for pretrained models)
        extra_initializers = get_preinit_encoders(training_model.encoder.encoders)
        if len(extra_initializers)==0:
            extra_initializers = None

        trainer = training.EarlyStoppingTrainer(model=training_model,
                                                optimizer_config=create_optimizer_config(args, [1.0], extra_initializers),
                                                max_params_files_to_keep=args.keep_last_params,
                                                source_vocabs=[None],
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
                    allow_missing_parameters=args.allow_missing_params,
                    existing_parameters=args.params)


if __name__ == "__main__":
    main()

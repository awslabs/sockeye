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

import copy
import time
import logging
import os
from typing import cast, Optional, Tuple, Union, List

import mxnet as mx
from sockeye import __version__
from sockeye.config import Config

from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import utils
from . import vocab

logger = logging.getLogger(__name__)


class ModelConfig(Config):
    """
    ModelConfig defines model parameters defined at training time which are relevant to model inference.
    Add new model parameters here. If you want backwards compatibility for models trained with code that did not
    contain these parameters, provide a reasonable default under default_values.

    :param config_data: Used training data.
    :param vocab_source_size: Source vocabulary size.
    :param vocab_target_size: Target vocabulary size.
    :param config_embed_source: Embedding config for source.
    :param config_embed_target: Embedding config for target.
    :param config_encoder: Encoder configuration.
    :param config_decoder: Decoder configuration.
    :param config_length_task: Optional length task configuration.
    :param weight_tying: Enables weight tying if True.
    :param weight_tying_type: Determines which weights get tied. Must be set if weight_tying is enabled.
    :param lhuc: LHUC (Vilar 2018) is applied at some part of the model.
    :param dtype: Data type of model parameters. Default: float32.
    """

    def __init__(self,
                 config_data: data_io.DataConfig,
                 vocab_source_size: int,
                 vocab_target_size: int,
                 config_embed_source: encoder.EmbeddingConfig,
                 config_embed_target: encoder.EmbeddingConfig,
                 config_encoder: encoder.EncoderConfig,
                 config_decoder: decoder.DecoderConfig,
                 config_length_task: layers.LengthRatioConfig = None,
                 weight_tying: bool = False,
                 weight_tying_type: Optional[str] = C.WEIGHT_TYING_TRG_SOFTMAX,
                 lhuc: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.config_data = config_data
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
        self.config_embed_source = config_embed_source
        self.config_embed_target = config_embed_target
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.config_length_task = config_length_task
        self.weight_tying = weight_tying
        self.weight_tying_type = weight_tying_type
        if weight_tying and weight_tying_type is None:
            raise RuntimeError("weight_tying_type must be specified when using weight_tying.")
        self.lhuc = lhuc
        self.dtype = dtype


class SockeyeModel(mx.gluon.Block):
    """
    SockeyeModel shares components needed for both training and inference.
    The main components of a Sockeye model are
    1) Source embedding
    2) Target embedding
    3) Encoder
    4) Decoder
    5) Output Layer

    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    :param prefix: Name prefix for all parameters of this model.
    """

    def __init__(self, config: ModelConfig, prefix: str = '', **kwargs) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self.config = copy.deepcopy(config)
        logger.info("%s", self.config)
        self.dtype = config.dtype

        with self.name_scope():
            # source & target embeddings
            self.source_embed_weight, self.target_embed_weight, self.output_weight = self._get_embedding_weights()

            self.embedding_source = encoder.Embedding(config.config_embed_source,
                                                      prefix=self.prefix,
                                                      is_source=True,
                                                      embed_weight=self.source_embed_weight)
            self.embedding_target = encoder.Embedding(config.config_embed_target,
                                                      prefix=self.prefix,
                                                      is_source=False,
                                                      embed_weight=self.target_embed_weight)

            # encoder & decoder first (to know the decoder depth)
            self.encoder = encoder.get_encoder(self.config.config_encoder, prefix=self.prefix)
            self.decoder = decoder.get_decoder(self.config.config_decoder, prefix=self.prefix)

            self.output_layer = layers.OutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                                   vocab_size=self.config.vocab_target_size,
                                                   weight=self.output_weight)

            self.length_ratio = None
            if self.config.config_length_task is not None:
                utils.check_condition(self.config.config_length_task.weight > 0.0,
                                      'Auxiliary length task requested, but its loss weight is zero')
                self.length_ratio = layers.LengthRatio(hidden_size=self.encoder.get_num_hidden(),
                                                       num_layers=self.config.config_length_task.num_layers,
                                                       prefix=self.prefix + C.LENRATIOS_OUTPUT_LAYER_PREFIX)

    def cast(self, dtype):
        self.dtype = dtype
        super().cast(dtype)

    def encode(self, inputs, valid_length=None):
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
        valid_length : NDArray or None, default None

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        source_embed, source_embed_length = self.embedding_source(inputs, valid_length)
        source_encoded, source_encoded_length = self.encoder(source_embed, source_embed_length)
        return source_encoded, source_encoded_length

    def decode_step(self, step_input, states, vocab_slice_ids = None):
        """One step decoding of the translation model.

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size,)
        states : list of NDArrays
        vocab_slice_ids : NDArray or None

        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        # TODO: do we need valid length!?
        valid_length = mx.nd.ones(shape=(step_input.shape[0],), ctx=step_input.context)
        # target_embed: (batch_size, num_hidden)
        target_embed, _ = self.embedding_target(step_input, valid_length=valid_length)

        # TODO: add step_additional_outputs
        step_additional_outputs = []
        # TODO: add support for states from the decoder
        decoder_out, new_states = self.decoder(target_embed, states)

        # step_output: (batch_size, target_vocab_size or vocab_slice_ids)
        step_output = self.output_layer(decoder_out, vocab_slice_ids)

        return step_output, new_states, step_additional_outputs

    def forward(self, source, source_length, target, target_length):  # pylint: disable=arguments-differ
        source_embed, source_embed_length = self.embedding_source(source, source_length)
        target_embed, target_embed_length = self.embedding_target(target, target_length)
        source_encoded, source_encoded_length = self.encoder(source_embed, source_embed_length)

        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_length, is_inference=False)
        target = self.decoder.decode_seq(target_embed, states=states)

        output = self.output_layer(target, None)

        if self.length_ratio is not None:
            # predicted_length_ratios: (batch_size,)
            predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
            return {C.LOGITS_NAME: output, C.LENRATIO_NAME: predicted_length_ratio}
        else:
            return {C.LOGITS_NAME: output}

    def predict_length_ratio(self, source_encoded, source_encoded_length):
        utils.check_condition(self.length_ratio is not None,
                              "Cannot predict length ratio, model does not seem to be trained with length task.")
        # predicted_length_ratios: (batch_size,)
        predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
        return predicted_length_ratio

    def save_config(self, folder: str):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved model config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = ModelConfig.load(fname)
        logger.info('Loaded model config from "%s"', fname)
        return cast(ModelConfig, config)  # type: ignore

    def save_parameters(self, fname: str):
        """
        Saves model parameters to file.
        :param fname: Path to save parameters to.
        """
        super().save_parameters(fname)
        logging.info('Saved params to "%s"', fname)

    def load_parameters(self,
                        filename: str,
                        ctx: Union[mx.Context, List[mx.Context]] = None,
                        allow_missing: bool = False,
                        ignore_extra: bool = False,
                        cast_dtype: bool = False,
                        dtype_source: str = 'current'):
        """Load parameters from file previously saved by `save_parameters`.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) to initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any.
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        References
        ----------
        `Saving and Loading Gluon Models \
        <https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html>`_
        """
        utils.check_condition(os.path.exists(filename), "No model parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % filename)
        super().load_parameters(filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra,
                                cast_dtype=cast_dtype, dtype_source=dtype_source)
        logger.info('Loaded params from "%s" to "%s"', filename, mx.cpu() if ctx is None else ctx)

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embedding_weights(self) -> Tuple[mx.gluon.Parameter, mx.gluon.Parameter, mx.gluon.Parameter]:
        """
        Returns embeddings for source, target, and output layer.
        When source and target embeddings are shared, they are created here and passed in to each side,
        instead of being created in the Embedding constructors.

        :return: Tuple of source, target, and output embedding parameters.
        """
        share_embed = self.config.weight_tying and \
                      C.WEIGHT_TYING_SRC in self.config.weight_tying_type and \
                      C.WEIGHT_TYING_TRG in self.config.weight_tying_type

        tie_weights = self.config.weight_tying and \
                      C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type

        source_embed_name = C.SOURCE_EMBEDDING_PREFIX + "weight" if not share_embed else C.SHARED_EMBEDDING_PREFIX + "weight"
        target_embed_name = C.TARGET_EMBEDDING_PREFIX + "weight" if not share_embed else C.SHARED_EMBEDDING_PREFIX + "weight"
        output_embed_name = "target_output_weight" if not tie_weights else target_embed_name

        source_embed_weight = self.params.get(source_embed_name,
                                              shape=(self.config.config_embed_source.vocab_size,
                                                     self.config.config_embed_source.num_embed),
                                              allow_deferred_init=True)

        if share_embed:
            target_embed_weight = source_embed_weight
        else:
            target_embed_weight = self.params.get(target_embed_name,
                                                  shape=(self.config.config_embed_target.vocab_size,
                                                         self.config.config_embed_target.num_embed),
                                                  allow_deferred_init=True)

        if tie_weights:
            output_weight = target_embed_weight
        else:
            output_weight = self.params.get(output_embed_name,
                                            shape=(self.config.config_embed_target.vocab_size, 0),
                                            allow_deferred_init=True)

        return source_embed_weight, target_embed_weight, output_weight

    @property
    def num_source_factors(self) -> int:
        """ Returns the number of source factors of this model (at least 1). """
        return self.config.config_data.num_source_factors

    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.training_max_seq_len_source

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.training_max_seq_len_target

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std


def load_model(model_folder: str,
               context: Union[List[mx.context.Context], mx.context.Context] = mx.cpu(),
               dtype: Optional[str] = None,
               checkpoint: Optional[int] = None,
               hybridize: bool = True) -> Tuple[SockeyeModel, List[vocab.Vocab], vocab.Vocab]:
    """
    Load a model from model_folder.

    :param model_folder: Model folder.
    :param context: MXNet context to bind modules to.
    :param checkpoint: Checkpoint to use. If none, uses best checkpoint.
    :param dtype: Optional data type to use. If None, will be inferred from stored model.
    :param hybridize: Whether to hybridize the loaded models. Default: true.
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    :return:
    """
    source_vocabs = vocab.load_source_vocabs(model_folder)
    target_vocab = vocab.load_target_vocab(model_folder)
    model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
    logger.info("Model version: %s", model_version)
    utils.check_version(model_version)
    model_config = SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))

    logger.info("Disabling dropout layers for performance reasons")
    model_config.disable_dropout()

    if checkpoint is None:
        params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
    else:
        params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

    model = SockeyeModel(model_config)
    model.initialize(ctx=context)
    model.cast(model_config.dtype)

    if dtype is None:
        logger.info("Model dtype: %s" % model_config.dtype)
        cast_dtype = False
        dtype_source = 'saved'
    else:
        logger.info("Model dtype: overriden to %s" % dtype)
        model.cast(dtype)
        cast_dtype = True
        dtype_source = 'current'

    model.load_parameters(filename=params_fname,
                          ctx=context,
                          allow_missing=False,
                          ignore_extra=False,
                          cast_dtype=cast_dtype,
                          dtype_source=dtype_source)
    for param in model.collect_params().values():
        param.grad_req = 'null'

    if hybridize:
        model.hybridize(static_alloc=True)

    utils.check_condition(model.num_source_factors == len(source_vocabs),
                          "Number of loaded source vocabularies (%d) does not match "
                          "number of source factors for model '%s' (%d)" % (len(source_vocabs), model_folder,
                                                                            model.num_source_factors))
    return model, source_vocabs, target_vocab


def load_models(context: Union[List[mx.context.Context], mx.context.Context],
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                dtype: Optional[str] = None,
                hybridize: bool = True) -> Tuple[List[SockeyeModel], List[vocab.Vocab], vocab.Vocab]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param dtype: Optional data type to use. If None, will be inferred from stored model.
    :param hybridize: Whether to hybridize the loaded models. Default: true.
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    """
    logger.info("Loading %d model(s) from %s ...", len(model_folders), model_folders)
    load_time_start = time.time()
    models = []  # type: List[SockeyeModel]
    source_vocabs = []  # type: List[List[vocab.Vocab]]
    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    else:
        utils.check_condition(len(checkpoints) == len(model_folders), "Must provide checkpoints for each model")

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model, src_vcbs, trg_vcb = load_model(model_folder,
                                              context=context,
                                              dtype=dtype,
                                              checkpoint=checkpoint,
                                              hybridize=hybridize)
        models.append(model)
        source_vocabs.append(src_vcbs)
        target_vocabs.append(trg_vcb)

    utils.check_condition(vocab.are_identical(*target_vocabs), "Target vocabulary ids do not match")
    first_model_vocabs = source_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[source_vocabs[i][fi] for i in range(len(source_vocabs))]),
                              "Source vocabulary ids do not match. Factor %d" % fi)

    load_time = time.time() - load_time_start
    logger.info("%d model(s) loaded in %.4fs", len(models), load_time)
    return models, source_vocabs[0], target_vocabs[0]

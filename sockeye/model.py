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
                 weight_normalization: bool = False,
                 lhuc: bool = False) -> None:
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
        self.weight_normalization = weight_normalization
        if weight_tying and weight_tying_type is None:
            raise RuntimeError("weight_tying_type must be specified when using weight_tying.")
        self.lhuc = lhuc


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
        self.dtype = 'float32'

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
            # TODO
            self.decoder = cast(decoder.TransformerDecoder, self.decoder)

            self.output_layer = layers.OutputLayer(vocab_size=self.config.vocab_target_size,
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
        states : list of NDArrays or None, default None
        valid_length : NDArray or None, default None

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        source_embed, source_embed_length = self.embedding_source(inputs, valid_length)
        source_encoded, source_encoded_length = self.encoder(source_embed, source_embed_length)
        return source_encoded, source_encoded_length

    def decode_step(self, step_input, states):
        """One step decoding of the translation model.

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size,)
        states : list of NDArrays

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
        # target_embed: (batch_size, num_factors, num_hidden)  # TODO(FH): why num_factors?
        target_embed, _ = self.embedding_target(step_input, valid_length=valid_length)

        # TODO: add step_additional_outputs
        step_additional_outputs = []
        # TODO: add support for states from the decoder
        step_output, new_states = self.decoder(target_embed, states)

        return step_output, new_states, step_additional_outputs

    def forward(self, source, source_length, target, target_length):  # pylint: disable=arguments-differ
        source_embed, source_embed_length = self.embedding_source(source, source_length)
        target_embed, target_embed_length = self.embedding_target(target, target_length)
        source_encoded, source_encoded_length = self.encoder(source_embed, source_embed_length)

        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_length, is_inference=False)
        target = self.decoder.decode_seq(target_embed, states=states)

        output = self.output_layer(target)

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

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.
        :param fname: Path to save parameters to.
        """
        self.save_parameters(fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self,
                              fname: str,
                              ctx: Union[mx.Context, List[mx.Context]] = None,
                              allow_missing: bool = False,
                              ignore_extra: bool = False):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        :param ctx: Context to load parameters to.
        :param allow_missing: Whether to not fail on missing parameters.
        :param ignore_extra: Whether to ignore extra parameters in the file.
        """
        utils.check_condition(os.path.exists(fname), "No model parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.load_parameters(fname, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        logger.info('Loaded params from "%s" to "%s"', fname, mx.cpu() if ctx is None else ctx)

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
        """
        Returns the number of source factors of this model (at least 1).
        """
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
        # TODO: this forced to training max length due to pos embeddings
        return self.training_max_seq_len_source

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        # TODO: this forced to training max length due to pos embeddings
        return self.training_max_seq_len_target

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std

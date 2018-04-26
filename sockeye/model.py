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
from typing import cast, Dict, Optional, Tuple

import mxnet as mx

from sockeye import __version__
from sockeye.config import Config
from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import loss
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
    :param config_loss: Loss configuration.
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
                 config_loss: loss.LossConfig,
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
        self.config_loss = config_loss
        self.weight_tying = weight_tying
        self.weight_tying_type = weight_tying_type
        self.weight_normalization = weight_normalization
        if weight_tying and weight_tying_type is None:
            raise RuntimeError("weight_tying_type must be specified when using weight_tying.")
        self.lhuc = lhuc


class SockeyeModel:
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

    def __init__(self, config: ModelConfig, prefix: str = '') -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()
        self.prefix = prefix
        logger.info("%s", self.config)

        # encoder & decoder first (to know the decoder depth)
        self.encoder = encoder.get_encoder(self.config.config_encoder, prefix=self.prefix)
        self.decoder = decoder.get_decoder(self.config.config_decoder, prefix=self.prefix)

        # source & target embeddings
        embed_weight_source, embed_weight_target, out_weight_target = self._get_embed_weights(self.prefix)
        self.embedding_source = encoder.Embedding(self.config.config_embed_source,
                                                  prefix=self.prefix + C.SOURCE_EMBEDDING_PREFIX,
                                                  embed_weight=embed_weight_source,
                                                  is_source=True)

        self.embedding_target = encoder.Embedding(self.config.config_embed_target,
                                                  prefix=self.prefix + C.TARGET_EMBEDDING_PREFIX,
                                                  embed_weight=embed_weight_target)

        # output layer
        self.output_layer = layers.OutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                               vocab_size=self.config.vocab_target_size,
                                               weight=out_weight_target,
                                               weight_normalization=self.config.weight_normalization,
                                               prefix=self.prefix + C.DEFAULT_OUTPUT_LAYER_PREFIX)

        self.params = None  # type: Optional[Dict]
        self.aux_params = None  # type: Optional[Dict]

    def save_config(self, folder: str):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = ModelConfig.load(fname)
        logger.info('ModelConfig loaded from "%s"', fname)
        return cast(ModelConfig, config)  # type: ignore

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        if self.aux_params is not None:
            utils.save_params(self.params.copy(), fname, self.aux_params.copy())
        else:
            utils.save_params(self.params.copy(), fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        utils.check_condition(os.path.exists(fname), "No model parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.params, self.aux_params = utils.load_params(fname)
        utils.check_condition(all(name.startswith(self.prefix) for name in self.params.keys()),
                              "Not all parameter names start with model prefix '%s'" % self.prefix)
        utils.check_condition(all(name.startswith(self.prefix) for name in self.aux_params.keys()),
                              "Not all auxiliary parameter names start with model prefix '%s'" % self.prefix)
        logger.info('Loaded params from "%s"', fname)

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embed_weights(self, prefix: str) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, mx.sym.Symbol]:
        """
        Returns embedding parameters for source and target.
        When source and target embeddings are shared, they are created here and passed in to each side,
        instead of being created in the Embedding constructors.

        :param prefix: Prefix.
        :return: Tuple of source and target parameter symbols.
        """
        w_embed_source = mx.sym.Variable(prefix + C.SOURCE_EMBEDDING_PREFIX + "weight",
                                         shape=(self.config.config_embed_source.vocab_size,
                                                self.config.config_embed_source.num_embed))
        w_embed_target = mx.sym.Variable(prefix + C.TARGET_EMBEDDING_PREFIX + "weight",
                                         shape=(self.config.config_embed_target.vocab_size,
                                                self.config.config_embed_target.num_embed))
        w_out_target = mx.sym.Variable(prefix + "target_output_weight",
                                       shape=(self.config.vocab_target_size, self.decoder.get_num_hidden()))

        if self.config.weight_tying:
            if C.WEIGHT_TYING_SRC in self.config.weight_tying_type \
                    and C.WEIGHT_TYING_TRG in self.config.weight_tying_type:
                logger.info("Tying the source and target embeddings.")
                w_embed_source = w_embed_target = mx.sym.Variable(prefix + C.SHARED_EMBEDDING_PREFIX + "weight",
                                                                  shape=(self.config.config_embed_source.vocab_size,
                                                                         self.config.config_embed_source.num_embed))

            if C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type:
                logger.info("Tying the target embeddings and output layer parameters.")
                utils.check_condition(self.config.config_embed_target.num_embed == self.decoder.get_num_hidden(),
                                      "Weight tying requires target embedding size and decoder hidden size " +
                                      "to be equal: %d vs. %d" % (self.config.config_embed_target.num_embed,
                                                                  self.decoder.get_num_hidden()))
                w_out_target = w_embed_target

        self._embed_weight_source_name = w_embed_source.name
        self._embed_weight_target_name = w_embed_target.name
        self._out_weight_target_name = w_out_target.name
        return w_embed_source, w_embed_target, w_out_target

    def get_source_embed_params(self) -> Optional[mx.nd.NDArray]:
        if self.params is None:
            return None
        return self.params.get(self._embed_weight_source_name)

    def get_target_embed_params(self) -> Optional[mx.nd.NDArray]:
        if self.params is None:
            return None
        return self.params.get(self._embed_weight_target_name)

    def get_output_embed_params(self) -> Optional[mx.nd.NDArray]:
        if self.params is None:
            return None
        return self.params.get(self._out_weight_target_name)

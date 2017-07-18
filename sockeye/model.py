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

from sockeye import __version__
from sockeye.config import Config
from . import attention
from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import lexicon
from . import loss
from . import utils

logger = logging.getLogger(__name__)


class ModelConfig(Config):
    """
    ModelConfig defines model parameters defined at training time which are relevant to model inference.
    Add new model parameters here. If you want backwards compatibility for models trained with code that did not
    contain these parameters, provide a reasonable default under default_values.

    :param config_data: Used training data.
    :param max_seq_len: Maximum sequence length to unroll during training.
    :param vocab_source_size: Source vocabulary size.
    :param vocab_target_size: Target vocabulary size.
    :param config_encoder: Encoder configuration.
    :param config_decoder: Decoder configuration.
    :param config_attention: Attention configuration.
    :param config_loss: Loss configuration.
    :param lexical_bias: Use lexical biases.
    :param learn_lexical_bias: Learn lexical biases during training.
    """
    yaml_tag = "!ModelConfig"

    def __init__(self,
                 config_data: data_io.DataConfig,
                 max_seq_len: int,
                 vocab_source_size: int,
                 vocab_target_size: int,
                 config_encoder: encoder.RecurrentEncoderConfig,
                 config_decoder: decoder.RecurrentDecoderConfig,
                 config_attention: attention.AttentionConfig,
                 config_loss: loss.LossConfig,
                 lexical_bias: bool = False,
                 learn_lexical_bias: bool = False):
        super().__init__()
        self.config_data = config_data
        self.max_seq_len = max_seq_len
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.config_attention = config_attention
        self.config_loss = config_loss
        self.lexical_bias = lexical_bias
        self.learn_lexical_bias = learn_lexical_bias


class SockeyeModel:
    """
    SockeyeModel shares components needed for both training and inference.
    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        self.config = copy.deepcopy(config)
        self.config.freeze()
        logger.info("%s", self.config)
        self.encoder = None
        self.attention = None
        self.decoder = None
        self.rnn_cells = []
        self.built = False
        self.params = None

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
        return config

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        assert self.built
        params = self.params.copy()
        # unpack rnn cell weights
        for cell in self.rnn_cells:
            params = cell.unpack_weights(params)
        utils.save_params(params, fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        assert self.built
        self.params, _ = utils.load_params(fname)
        # pack rnn cell weights
        for cell in self.rnn_cells:
            self.params = cell.pack_weights(self.params)
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

    def _build_model_components(self, max_seq_len: int, fused_encoder: bool):
        """
        Builds and sets model components given maximum sequence length.

        :param max_seq_len: Maximum sequence length supported by the model.
        :param fused_encoder: Use FusedRNNCells in encoder.
        """
        self.encoder = encoder.get_recurrent_encoder(self.config.config_encoder, fused_encoder)

        self.attention = attention.get_attention(self.config.config_attention, max_seq_len)

        self.lexicon = lexicon.Lexicon(self.config.vocab_source_size,
                                       self.config.vocab_target_size,
                                       self.config.learn_lexical_bias) if self.config.lexical_bias else None

        self.decoder = decoder.get_recurrent_decoder(self.config.config_decoder,
                                                     self.attention,
                                                     self.lexicon)

        self.rnn_cells = self.encoder.get_rnn_cells() + self.decoder.get_rnn_cells()

        self.built = True

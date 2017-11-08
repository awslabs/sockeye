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
from typing import Dict, List, Optional

import mxnet as mx

from sockeye import __version__
from sockeye.config import Config
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
    :param max_seq_len_source: Maximum source sequence length to unroll during training.
    :param max_seq_len_target: Maximum target sequence length to unroll during training.
    :param vocab_source_size: Source vocabulary size.
    :param vocab_target_size: Target vocabulary size.
    :param config_encoder: Encoder configuration.
    :param config_decoder: Decoder configuration.
    :param config_loss: Loss configuration.
    :param lexical_bias: Use lexical biases.
    :param learn_lexical_bias: Learn lexical biases during training.
    :param weight_tying: Enables weight tying if True.
    :param weight_tying_type: Determines which weights get tied. Must be set if weight_tying is enabled.
    """
    def __init__(self,
                 config_data: data_io.DataConfig,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 vocab_source_size: int,
                 vocab_target_size: int,
                 config_encoder: Config,
                 config_decoder: Config,
                 config_loss: loss.LossConfig,
                 lexical_bias: bool = False,
                 learn_lexical_bias: bool = False,
                 weight_tying: bool = False,
                 weight_tying_type: Optional[str] = C.WEIGHT_TYING_TRG_SOFTMAX) -> None:
        super().__init__()
        self.config_data = config_data
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.config_loss = config_loss
        self.lexical_bias = lexical_bias
        self.learn_lexical_bias = learn_lexical_bias
        self.weight_tying = weight_tying
        if weight_tying and weight_tying_type is None:
            raise RuntimeError("weight_tying_type must be specified when using weight_tying.")
        self.weight_tying_type = weight_tying_type


class SockeyeModel:
    """
    SockeyeModel shares components needed for both training and inference.
    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()
        logger.info("%s", self.config)
        self.encoder = None  # type: Optional[encoder.Encoder]
        self.decoder = None  # type: Optional[decoder.Decoder]
        self.built = False
        self.params = None  # type: Optional[Dict]

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
        return config  # type: ignore

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        assert self.built
        utils.save_params(self.params.copy(), fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        assert self.built
        utils.check_condition(os.path.exists(fname), "No model parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.params, _ = utils.load_params(fname)
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

    def _build_model_components(self):
        """
        Instantiates model components.
        """
        # we tie the source and target embeddings if both appear in the type
        if self.config.weight_tying and C.WEIGHT_TYING_SRC in self.config.weight_tying_type \
                and C.WEIGHT_TYING_TRG in self.config.weight_tying_type:
            logger.info("Tying the source and target embeddings.")
            embed_weight = encoder.Embedding.get_embed_weight(vocab_size=self.config.vocab_source_size,
                                                              embed_size=0,  # will get inferred
                                                              prefix=C.SHARED_EMBEDDING_PREFIX)
        else:
            embed_weight = None

        self.encoder = encoder.get_encoder(self.config.config_encoder, embed_weight)

        self.lexicon = lexicon.Lexicon(self.config.vocab_source_size,
                                       self.config.vocab_target_size,
                                       self.config.learn_lexical_bias) if self.config.lexical_bias else None

        self.decoder = decoder.get_decoder(self.config.config_decoder, self.lexicon, embed_weight)

        self.built = True

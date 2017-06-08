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

import json
import logging
import os

import sockeye.attention
import sockeye.coverage
import sockeye.data_io
import sockeye.decoder
import sockeye.encoder
import sockeye.lexicon
import sockeye.utils
from sockeye import constants as C

logger = logging.getLogger(__name__)

ModelConfig = sockeye.utils.namedtuple_with_defaults('ModelConfig',
                                                     [
                                                      "max_seq_len",
                                                      "vocab_source_size",
                                                      "vocab_target_size",
                                                      "num_embed_source",
                                                      "num_embed_target",
                                                      "attention_type",
                                                      "attention_num_hidden",
                                                      "attention_coverage_type",
                                                      "attention_coverage_num_hidden",
                                                      "attention_use_prev_word",
                                                      "dropout",
                                                      "rnn_cell_type",
                                                      "rnn_num_layers",
                                                      "rnn_num_hidden",
                                                      "rnn_residual_connections",
                                                      "weight_tying",
                                                      "context_gating",
                                                      "lexical_bias",
                                                      "learn_lexical_bias",
                                                      "data_info",
                                                      "loss",
                                                      "normalize_loss",
                                                      "smoothed_cross_entropy_alpha",
                                                  ],
                                                     default_values={
                                                      "attention_use_prev_word": False,
                                                      "context_gating": False,
                                                      "loss": C.CROSS_ENTROPY,
                                                      "normalize_loss": False
                                                  })
"""
ModelConfig defines model parameters defined at training time which are relevant to model inference.
Add new model parameters here. If you want backwards compatibility for models trained with code that did not
contain these parameters, provide a reasonable default under default_values.
"""


class SockeyeModel:
    """
    SockeyeModel shares components needed for both training and inference.
    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
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
        with open(fname, "w") as out:
            json.dump(self.config._asdict(), out, indent=2, sort_keys=True)
            logger.info('Saved config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        with open(fname, "r") as inp:
            config = ModelConfig(**json.load(inp))
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
        sockeye.utils.save_params(params, fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        assert self.built
        self.params, _ = sockeye.utils.load_params(fname)
        # pack rnn cell weights
        for cell in self.rnn_cells:
            self.params = cell.pack_weights(self.params)
        logger.info('Loaded params from "%s"', fname)

    def _build_model_components(self, max_seq_len: int, fused_encoder: bool, rnn_forget_bias: float = 0.0):
        """
        Builds and sets model components given maximum sequence length.
        
        :param max_seq_len: Maximum sequence length supported by the model.
        :param fused_encoder: Use FusedRNNCells in encoder.
        :param rnn_forget_bias: forget bias initialization for RNNs.
        """
        self.encoder = sockeye.encoder.get_encoder(self.config.num_embed_source,
                                                   self.config.vocab_source_size,
                                                   self.config.rnn_num_layers,
                                                   self.config.rnn_num_hidden,
                                                   self.config.rnn_cell_type,
                                                   self.config.rnn_residual_connections,
                                                   self.config.dropout,
                                                   rnn_forget_bias,
                                                   fused_encoder)

        self.attention = sockeye.attention.get_attention(self.config.attention_use_prev_word,
                                                         self.config.attention_type,
                                                         self.config.attention_num_hidden,
                                                         self.config.rnn_num_hidden,
                                                         max_seq_len,
                                                         self.config.attention_coverage_type,
                                                         self.config.attention_coverage_num_hidden)

        self.lexicon = sockeye.lexicon.Lexicon(self.config.vocab_source_size,
                                               self.config.vocab_target_size,
                                               self.config.learn_lexical_bias) if self.config.lexical_bias else None

        self.decoder = sockeye.decoder.get_decoder(self.config.num_embed_target,
                                                   self.config.vocab_target_size,
                                                   self.config.rnn_num_layers,
                                                   self.config.rnn_num_hidden,
                                                   self.attention,
                                                   self.config.rnn_cell_type,
                                                   self.config.rnn_residual_connections,
                                                   rnn_forget_bias,
                                                   self.config.dropout,
                                                   self.config.weight_tying,
                                                   self.lexicon,
                                                   self.config.context_gating)

        self.rnn_cells = self.encoder.get_rnn_cells() + self.decoder.get_rnn_cells()

        self.built = True

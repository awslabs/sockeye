# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import cast, Dict, Optional, Tuple, Union, List
from functools import lru_cache

import torch as pt
import mxnet as mx
from mxnet import gluon, np, npx
from sockeye import __version__
from sockeye.config import Config

from .model import ModelConfig, SockeyeModel
from . import constants as C
from . import data_io
from . import decoder_pt
from . import encoder_pt
from . import layers_pt
from . import quantization
from . import utils
from . import vocab
from dataclasses import dataclass
from .encoder import FactorConfig

logger = logging.getLogger(__name__)


class PyTorchSockeyeModel(pt.nn.Module):
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
    :param inference_only: Use the model only for inference, enabling optimizations.
    """

    def __init__(self,
                 config: ModelConfig,
                 inference_only: bool = False,
                 mc_dropout: bool = False,
                 forward_pass_cache_size: int = 0) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        logger.info("%s", self.config)
        self.dtype = config.dtype
        self.mc_dropout = mc_dropout
        self._output_layer_factor_format_string = 'output_layer_factor%i'
        self.forward_pass_cache_size = forward_pass_cache_size
        self.embed_and_encode = self._embed_and_encode
        if self.forward_pass_cache_size > 0:
            self.embed_and_encode = self._cache_wrapper(self._embed_and_encode)

        # source & target embeddings, potentially shared/tied
        source_embedding, target_embedding, output_weight = self._get_embeddings()

        self.embedding_source = encoder_pt.PyTorchEmbedding(config.config_embed_source, embedding=source_embedding)
        self.embedding_target = encoder_pt.PyTorchEmbedding(config.config_embed_target, embedding=target_embedding)

        # encoder & decoder first (to know the decoder depth)
        self.encoder = encoder_pt.pytorch_get_transformer_encoder(self.config.config_encoder, dtype=config.dtype)
        self.decoder = decoder_pt.pytorch_get_decoder(self.config.config_decoder, inference_only=inference_only,
                                                      dtype=config.dtype)

        self.output_layer = layers_pt.PyTorchOutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                                         vocab_size=self.config.vocab_target_size,
                                                         weight=output_weight, dtype=config.dtype)

        self.factor_output_layers = pt.nn.ModuleList()
        # Optional target factor output layers
        for i, factor_config in enumerate(self.target_factor_configs, 1):
            # Each target stream has its own, independent output layer
            # TODO also consider weight tying with target factor input embeddings
            output_layer = layers_pt.PyTorchOutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                                        vocab_size=factor_config.vocab_size,
                                                        weight=None,
                                                        dtype=config.dtype)
            self.factor_output_layers.append(output_layer)

        self.length_ratio = None  # type: Optional[layers_pt.PyTorchLengthRatio]
        if self.config.config_length_task is not None:
            utils.check_condition(self.config.config_length_task.weight > 0.0,
                                  'Auxiliary length task requested, but its loss weight is zero')
            self.length_ratio = layers_pt.PyTorchLengthRatio(hidden_size=self.encoder.get_num_hidden(),
                                                             num_layers=self.config.config_length_task.num_layers)

    def weights_from_mxnet_block(self, block_mx: SockeyeModel):
        self.embedding_source.weights_from_mxnet_block(block_mx.embedding_source)
        self.embedding_target.weights_from_mxnet_block(block_mx.embedding_target)
        self.encoder.weights_from_mxnet_block(block_mx.encoder)
        self.decoder.weights_from_mxnet_block(block_mx.decoder)
        self.output_layer.weights_from_mxnet_block(block_mx.output_layer)
        for i, factor_output_layer in enumerate(self.factor_output_layers):
            factor_output_layer.weights_from_mnxet_block(block_mx.factor_output_layers[0])
        if self.config.config_length_task is not None:
            self.length_ratio.weights_from_mxnet_block(block_mx.length_ratio)

    def cast(self, dtype):
        self.dtype = dtype
        super().cast(dtype)

    def state_structure(self):
        return self.decoder.state_structure()

    def encode(self, inputs: pt.Tensor, valid_length: Optional[pt.Tensor] = None) -> Tuple[pt.Tensor, pt.Tensor]:
        """Encode the input sequence.

        Parameters
        ----------
        inputs : tensor
        valid_length : tensor or None, default None

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        source_embed, source_embed_length = self.embedding_source(inputs, valid_length)
        source_encoded, source_encoded_length = self.encoder(source_embed, source_embed_length)
        return source_encoded, source_encoded_length

    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: Optional[pt.Tensor] = None,
                              constant_length_ratio: float = 0.0) -> Tuple[List[pt.Tensor], pt.Tensor]:
        """
        Encodes the input sequence and initializes decoder states (and predicted output lengths if available).
        Used for inference/decoding.

        Parameters
        ----------
        inputs : tensor
        valid_length : tensor or None, default None
        constant_length_ratio : float

        Returns
        -------
        states : list
            Initial states for the decoder.
        predicted_output_length : tensor
            Predicted output length of shape (batch_size,), 0 if not available.
        """
        if self.mc_dropout:
            raise NotImplementedError('mc dropout not implemented yet')
            # Turn on training mode so mxnet knows to add dropout
            _ = mx.autograd.set_training(True)

        # Encode input. Shape: (batch, length, num_hidden), (batch,)
        source_encoded, source_encoded_lengths = self.encode(inputs, valid_length=valid_length)

        predicted_output_length = self.predict_output_length(source_encoded,
                                                             source_encoded_lengths,
                                                             constant_length_ratio)
        # Decoder init states
        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_lengths)

        return states, predicted_output_length

    def _embed_and_encode(self,
                          source: pt.Tensor, source_length: pt.Tensor,
                          target: pt.Tensor, target_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor,
                                                                                pt.Tensor, List[pt.Tensor]]:
        """
        Encode the input sequence, embed the target sequence, and initialize the decoder.
        Used for training.

        :param source: Source input data.
        :param source_length: Length of source inputs.
        :param target: Target input data.
        :param target_length: Length of target inputs.
        :return: encoder outputs and lengths, target embeddings, and decoder initial states
        """
        source_embed, source_embed_length = self.embedding_source(source, source_length)
        target_embed, target_embed_length = self.embedding_target(target, target_length)
        source_encoded, source_encoded_length = self.encoder(source_embed, source_embed_length)
        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_length, target_embed)
        return source_encoded, source_encoded_length, target_embed, states

    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List[pt.Tensor],
                    vocab_slice_ids: Optional[pt.Tensor] = None) -> Tuple[pt.Tensor, List[pt.Tensor], List[pt.Tensor]]:
        """
        One step decoding of the translation model.

        Parameters
        ----------
        step_input : tensor
            Shape (batch_size, num_target_factors)
        states : list of tensors
        vocab_slice_ids : tensor or None

        Returns
        -------
        step_output : tensor
            Shape (batch_size, C_out)
        states : list
        target_factor_outputs : list
            Optional target factor predictions.
        """
        if self.mc_dropout:
            raise NotImplementedError('mc dropout not implented yet')
            # Turn on training mode so mxnet knows to add dropout
            _ = mx.autograd.set_training(True)

        valid_length = pt.ones(step_input.size()[0], device=step_input.device)
        target_embed, _ = self.embedding_target(step_input.unsqueeze(1), valid_length=valid_length)
        decoder_out, new_states = self.decoder(target_embed, states)
        decoder_out = decoder_out.squeeze(1)
        # step_output: (batch_size, target_vocab_size or vocab_slice_ids)
        step_output = self.output_layer(decoder_out, vocab_slice_ids)

        # Target factor outputs are currently stored in additional outputs.
        target_factor_outputs = []  # type: List[pt.Tensor]
        # TODO: consider a dictionary mapping as return value
        for factor_output_layer in self.factor_output_layers:
            target_factor_outputs.append(factor_output_layer(decoder_out, None))

        return step_output, new_states, target_factor_outputs

    def forward(self, source, source_length, target, target_length):  # pylint: disable=arguments-differ
        source_encoded, source_encoded_length, target_embed, states = self.embed_and_encode(source, source_length,
                                                                                            target, target_length)

        target = self.decoder.decode_seq(target_embed, states=states)

        forward_output = dict()

        forward_output[C.LOGITS_NAME] = self.output_layer(target, None)

        for i, factor_output_layer in enumerate(self.factor_output_layers, 1):
            forward_output[C.FACTOR_LOGITS_NAME % i] = factor_output_layer(target, None)

        if self.length_ratio is not None:
            # predicted_length_ratios: (batch_size,)
            forward_output[C.LENRATIO_NAME] = self.length_ratio(source_encoded, source_encoded_length)

        return forward_output

    def predict_output_length(self,
                              source_encoded: pt.Tensor,
                              source_encoded_length: pt.Tensor,
                              constant_length_ratio: float = 0.0) -> pt.Tensor:
        if self.length_ratio is not None:
            # predicted_length_ratios: (batch_size,)
            predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
            predicted_output_length = predicted_length_ratio * source_encoded_length
        elif constant_length_ratio > 0.0:
            # (batch,)
            predicted_output_length = source_encoded_length * constant_length_ratio
        else:
            # (batch,)
            predicted_output_length = pt.zeros_like(source_encoded_length)

        return predicted_output_length

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
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        """
        pt.save(self.state_dict(), fname)
        logging.info('Saved params/state_dict to "%s"', fname)

    def load_parameters(self,
                        filename: str,
                        ctx: Union[mx.Context, List[mx.Context]] = None,
                        allow_missing: bool = False,
                        ignore_extra: bool = False,
                        cast_dtype: bool = False,
                        dtype_source: str = 'current'):
        """Load parameters from file previously saved by `save_parameters`.
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

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
            Cast the data type of the ndarray loaded from the checkpoint to the dtype
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
        assert ctx is None, "not implemented yet"
        assert cast_dtype is False, "not implemented yet"
        assert dtype_source == 'current', "not implemented yet"
        utils.check_condition(os.path.exists(filename), "No model parameter file found under %s. "
                                                        "This is either not a model directory or the first training "
                                                        "checkpoint has not happened yet." % filename)
        state_dict = pt.load(filename)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if not allow_missing:
            utils.check_condition(not missing, f"missing keys: {missing}")
        if not ignore_extra:
            utils.check_condition(not unexpected, f"extra keys: {unexpected}")
        logger.info('Loaded params from "%s" to "%s"', filename, mx.cpu() if ctx is None else ctx)

    def set_parameters(self,  # TODO
                       new_params: Dict[str, gluon.Parameter],
                       allow_missing: bool = True,
                       ignore_extra: bool = False):
        """
        Update model params on all contexts of the model with new values from a dictionary.

        :param new_params: Dictionary containing the new parameters.
        :param allow_missing: Whether to skip setting parameters not represented in the dictionary.
        :param ignore_extra: Whether to ignore parameters from new_params that are not present in this model.
        """
        model_params = self.collect_params()
        if not allow_missing:
            for k in model_params.keys():
                assert k in new_params.keys(), "Parameter '%s' is missing in new_params dictionary. " \
                                               "Set allow_missing=True to ignore missing parameters." % k
        for k in new_params:
            assert new_params[k]._data is not None, "Parameter '%s' is not initialized in new_params dictionary." % k
            if not ignore_extra and k not in model_params:
                raise ValueError("Parameter '%s' in new_params dictionary is not preset in ParameterDict. "
                                 "Set ignore_extra=True to ignore." % k)
            if k in model_params:
                assert model_params[k]._data is not None, "Parameter '%s' must be initialized before it can be reset " \
                                                          "using set_parameters." % k
                assert model_params[k].shape == new_params[k].shape, \
                    "Parameter '%s' has shape '%s' in the model but shape '%s' in the new_params dictionary." % \
                    (k, model_params[k].shape, new_params[k].shape)
                model_params[k].set_data(new_params[k].data())

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embeddings(self) -> Tuple[pt.nn.Embedding, pt.nn.Embedding, Optional[pt.Tensor]]:
        """
        Returns embeddings for source, target, and output layer. Handles sharing and weight tying.
        """
        share_embed = C.WEIGHT_TYING_SRC in self.config.weight_tying_type and \
                      C.WEIGHT_TYING_TRG in self.config.weight_tying_type

        tie_weights = C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type

        source_grad_sparse = self.config.config_embed_source.allow_sparse_grad and not tie_weights
        source_embedding = pt.nn.Embedding(self.config.config_embed_source.vocab_size,
                                           self.config.config_embed_source.num_embed,
                                           sparse=source_grad_sparse)

        if share_embed:
            target_embedding = source_embedding
        else:
            target_grad_sparse = self.config.config_embed_target.allow_sparse_grad and not tie_weights
            target_embedding = pt.nn.Embedding(self.config.config_embed_target.vocab_size,
                                           self.config.config_embed_target.num_embed,
                                           sparse=target_grad_sparse)

        if tie_weights:
            output_weight = target_embedding.weight
        else:
            output_weight = None  # will be created when instantiating the OutputLayer

        return source_embedding, target_embedding, output_weight

    @property
    def num_source_factors(self) -> int:
        """ Returns the number of source factors of this model (at least 1). """
        return self.config.config_data.num_source_factors

    @property
    def num_target_factors(self) -> int:
        """ Returns the number of target factors of this model (at least 1). """
        return self.config.config_data.num_target_factors

    @property
    def target_factor_configs(self) -> List[FactorConfig]:
        """ Returns the factor configs for target factors. """
        factor_configs = []  # type: List[FactorConfig]
        if self.config.config_embed_target.factor_configs:
            factor_configs = self.config.config_embed_target.factor_configs
        return factor_configs

    @property
    def training_max_observed_len_source(self) -> int:
        """ The maximum sequence length on the source side observed during training. This includes the <eos> token. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_observed_len_target(self) -> int:
        """ The maximum sequence length on the target side observed during training. This includes the <bos> token. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_len_source(self) -> int:
        """ The maximum supported source length. This includes the <eos> token. """
        return self.config.config_data.max_seq_len_source

    @property
    def max_supported_len_target(self) -> int:
        """ The maximum supported target length. This includes the <bos> token. """
        return self.config.config_data.max_seq_len_target

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std

    @property
    def output_layer_vocab_size(self) -> int:
        return self.output_layer.vocab_size

    def _cache_wrapper(self, class_func):
        @lru_cache(maxsize=self.forward_pass_cache_size)
        def cache_func(*args):
            return class_func(*args)

        return cache_func

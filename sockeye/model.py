# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import cast, Dict, Optional, Tuple, Union, List

import mxnet as mx
from sockeye import __version__
from sockeye.config import Config

from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import quantization
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
    :param weight_tying_type: Determines which weights get tied.
    :param lhuc: LHUC (Vilar 2018) is applied at some part of the model.
    :param dtype: Data type of model parameters. Default: float32.
    :param intgemm_custom_lib: Path to intgemm custom operator library used for dtype is int8.  Default: libintgemm.so
                               in the same directory as this script.
    """

    def __init__(self,
                 config_data: data_io.DataConfig,
                 vocab_source_size: int,
                 vocab_target_size: int,
                 config_embed_source: encoder.EmbeddingConfig,
                 config_embed_target: encoder.EmbeddingConfig,
                 config_encoder: encoder.EncoderConfig,
                 config_decoder: decoder.DecoderConfig,
                 config_length_task: layers.LengthRatioConfig= None,
                 weight_tying_type: str = C.WEIGHT_TYING_SRC_TRG_SOFTMAX,
                 lhuc: bool = False,
                 dtype: str = C.DTYPE_FP32,
                 intgemm_custom_lib: str = os.path.join(os.path.dirname(__file__), "libintgemm.so")) -> None:
        super().__init__()
        self.config_data = config_data
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
        self.config_embed_source = config_embed_source
        self.config_embed_target = config_embed_target
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.config_length_task = config_length_task
        self.weight_tying_type = weight_tying_type
        self.lhuc = lhuc
        self.dtype = dtype
        self.intgemm_custom_lib = intgemm_custom_lib


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
    :param inference_only: Use the model only for inference, enabling optimizations.
    :param prefix: Name prefix for all parameters of this model.
    """

    def __init__(self,
                 config: ModelConfig,
                 inference_only: bool = False,
                 mc_dropout: bool = False,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self.config = copy.deepcopy(config)
        logger.info("%s", self.config)
        self.dtype = config.dtype
        self.mc_dropout = mc_dropout

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
            self.encoder = encoder.get_encoder(self.config.config_encoder, prefix=self.prefix, dtype=config.dtype)
            self.decoder = decoder.get_decoder(self.config.config_decoder, inference_only=inference_only,
                                               prefix=self.prefix, dtype=config.dtype)

            self.output_layer = layers.OutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                                   vocab_size=self.config.vocab_target_size,
                                                   weight=self.output_weight, dtype=config.dtype)

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

    def state_structure(self):
        return self.decoder.state_structure()

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

    def encode_and_initialize(self, inputs, valid_length=None, constant_length_ratio=0.0):
        """
        Encodes the input sequence and initializes decoder states (and predicted output lengths if available).
        Used for inference/decoding.

        Parameters
        ----------
        inputs : NDArray
        valid_length : NDArray or None, default None
        constant_length_ratio : float

        Returns
        -------
        states : list
            Initial states for the decoder.
        predicted_output_length : NDArray
            Predicted output length of shape (batch_size,), 0 if not available.
        """
        if self.mc_dropout:
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

    def decode_step(self, step_input, states, vocab_slice_ids=None):
        """
        One step decoding of the translation model.

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
        if self.mc_dropout:
            # Turn on training mode so mxnet knows to add dropout
            _ = mx.autograd.set_training(True)

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

        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_length)
        target = self.decoder.decode_seq(target_embed, states=states)

        output = self.output_layer(target, None)

        if self.length_ratio is not None:
            # predicted_length_ratios: (batch_size,)
            predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
            return {C.LOGITS_NAME: output, C.LENRATIO_NAME: predicted_length_ratio}
        else:
            return {C.LOGITS_NAME: output}

    def predict_output_length(self,
                              source_encoded: mx.nd.NDArray,
                              source_encoded_length: mx.nd.NDArray,
                              constant_length_ratio: float = 0.0):
        if self.length_ratio is not None:
            # predicted_length_ratios: (batch_size,)
            predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
            predicted_output_length = predicted_length_ratio * source_encoded_length
        elif constant_length_ratio > 0.0:
            # (batch,)
            predicted_output_length = source_encoded_length * constant_length_ratio
        else:
            # (batch,)
            predicted_output_length = mx.nd.zeros_like(source_encoded_length)

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
        """
        super().save_parameters(fname, deduplicate=True)
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

    def set_parameters(self,
                       new_params: Dict[str, mx.gluon.parameter.Parameter],
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

    def _get_embedding_weights(self) -> Tuple[mx.gluon.Parameter, mx.gluon.Parameter, mx.gluon.Parameter]:
        """
        Returns embeddings for source, target, and output layer.
        When source and target embeddings are shared, they are created here and passed in to each side,
        instead of being created in the Embedding constructors.

        :return: Tuple of source, target, and output embedding parameters.
        """
        share_embed = C.WEIGHT_TYING_SRC in self.config.weight_tying_type and \
                      C.WEIGHT_TYING_TRG in self.config.weight_tying_type

        tie_weights = C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type

        source_embed_name = C.SOURCE_EMBEDDING_PREFIX + "weight" if not share_embed else C.SHARED_EMBEDDING_PREFIX + "weight"
        target_embed_name = C.TARGET_EMBEDDING_PREFIX + "weight" if not share_embed else C.SHARED_EMBEDDING_PREFIX + "weight"
        output_embed_name = "target_output_weight" if not tie_weights else target_embed_name

        source_grad_stype = 'row_sparse' if self.config.config_embed_source.allow_sparse_grad and not tie_weights else 'default'
        source_embed_weight = self.params.get(source_embed_name,
                                              shape=(self.config.config_embed_source.vocab_size,
                                                     self.config.config_embed_source.num_embed),
                                              allow_deferred_init=True,
                                              grad_stype=source_grad_stype)

        if share_embed:
            target_embed_weight = source_embed_weight
        else:
            target_grad_stype = 'row_sparse' if self.config.config_embed_target.allow_sparse_grad and not tie_weights else 'default'
            target_embed_weight = self.params.get(target_embed_name,
                                                  shape=(self.config.config_embed_target.vocab_size,
                                                         self.config.config_embed_target.num_embed),
                                                  allow_deferred_init=True,
                                                  grad_stype=target_grad_stype)

        if tie_weights:
            output_weight = target_embed_weight
        else:
            output_weight = self.params.get(output_embed_name,
                                            shape=(self.config.config_embed_target.vocab_size,
                                                   self.config.config_decoder.model_size),
                                            allow_deferred_init=True)

        return source_embed_weight, target_embed_weight, output_weight

    @property
    def num_source_factors(self) -> int:
        """ Returns the number of source factors of this model (at least 1). """
        return self.config.config_data.num_source_factors

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


def load_model(model_folder: str,
               context: Union[List[mx.context.Context], mx.context.Context] = mx.cpu(),
               dtype: Optional[str] = None,
               checkpoint: Optional[int] = None,
               hybridize: bool = True,
               inference_only: bool = False,
               mc_dropout: bool = False,
               for_disk_saving: Optional[str] = None,
               allow_missing: bool = False,
               set_grad_req_null: bool = True) -> Tuple[SockeyeModel, List[vocab.Vocab], vocab.Vocab]:
    """
    Load a model from model_folder.

    :param model_folder: Model folder.
    :param context: MXNet context to bind modules to.
    :param checkpoint: Checkpoint to use. If none, uses best checkpoint.
    :param dtype: Optional data type to use. If None, will be inferred from stored model.
    :param hybridize: Whether to hybridize the loaded models. Default: true.
    :param inference_only: Use the model only for inference, enabling optimizations.
    :param mc_dropout: Turn on dropout during inference.
    :param for_disk_saving: For saving quantized models to disk.
           None: load as usual and the model will work.
           int8: The model loaded into RAM will not work, but is suitable for
               writing to disk in quantized format (including scaling factors).
           float32: The model loaded into RAM will not work, but is suitable
               for writing to disk as float32 with precomputed scaling factors.
    :param allow_missing: Allow missing parameters in the loaded model.
    :param set_grad_req_null: Set grad_req to null for model parameters.
    :return: List of models, source vocabularies, target vocabulary.
    """
    source_vocabs = vocab.load_source_vocabs(model_folder)
    target_vocab = vocab.load_target_vocab(model_folder)
    model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
    logger.info("Model version: %s", model_version)
    utils.check_version(model_version)
    model_config = SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))

    if inference_only and not mc_dropout:
        logger.info("Disabling dropout layers for performance reasons")
        model_config.disable_dropout()

    if mc_dropout:
        logger.info("Monte Carlo dropout enabled, inference output will be non-deterministic.")

    if checkpoint is None:
        params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
    else:
        params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

    if (dtype == C.DTYPE_INT8 or
        model_config.dtype == C.DTYPE_INT8 or
        for_disk_saving is not None) and "intgemm_fully_connected" not in dir(mx.nd.contrib):
        # We're going to use int8 but it's not compiled into mxnet.
        path = os.path.abspath(model_config.intgemm_custom_lib)
        try:
            mx.library.load(path)
        except mx.base.MXNetError:
            raise NotImplementedError("8-bit int inference requested but intgemm was not compiled into MXNet and a "
                                      "custom operator library was not found in `%s`.  Compile the custom "
                                      "operator then set the path using intgemm_custom_lib in the config file." % path)

    # Are we converting the model to 8-bit?
    quantizing = model_config.dtype != C.DTYPE_INT8 and (dtype == C.DTYPE_INT8 or for_disk_saving is not None)
    if quantizing:
        model_config.dtype = C.DTYPE_INT8 # Ensure the scaling factor parameters are created.

    model = SockeyeModel(model_config, inference_only=inference_only, mc_dropout=mc_dropout)
    model.initialize(ctx=context)
    if model_config.dtype != C.DTYPE_INT8:
        # If model_config.dtype is int8, then the above model construction
        # (which also used model_config) already set everything to the correct
        # mix of float32 and int8.  Cast would try to make everything int8.
        model.cast(model_config.dtype)

    if quantizing:
        logger.info("Model dtype: quantizing from float32 to int8")
        allow_missing = True  # The scaling factors are missing
        cast_dtype = True
        dtype_source = 'saved'
    elif dtype is None or dtype == model_config.dtype:
        logger.info("Model dtype: %s" % model_config.dtype)
        allow_missing = allow_missing
        cast_dtype = False
        dtype_source = 'saved'
    else:
        logger.info("Model dtype: overridden to %s" % dtype)
        model.cast(dtype)
        allow_missing = allow_missing
        cast_dtype = True
        dtype_source = 'current'

    model.load_parameters(filename=params_fname,
                          ctx=context,
                          allow_missing=allow_missing,
                          ignore_extra=True,  # Scaling factors may be present in float32 models.
                          cast_dtype=cast_dtype,
                          dtype_source=dtype_source)

    params = model.collect_params()
    if set_grad_req_null:
        for param in params.values():
            param.grad_req = 'null'

    if for_disk_saving is not None:
        # Saving scaling factors and possibly int8 values to disk.
        if not quantizing:
            raise RuntimeError("Model is already quantized and for_disk_saving is set.")
        quantization.convert_weights_disk_format(params, for_disk_saving)
        model.config.dtype = for_disk_saving
        # TODO: check for missing parameters somehow (we allowed scaling to be missing)
    if for_disk_saving is None and model_config.dtype == C.DTYPE_INT8:
        # Disk format to CPU-dependent format.
        quantization.convert_weights_cpu_dependent(params)

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
                dtype: Optional[str] = C.DTYPE_FP32,
                hybridize: bool = True,
                inference_only: bool = False,
                mc_dropout: bool = False,
                allow_missing: bool = False,
                set_grad_req_null: bool = True) -> Tuple[List[SockeyeModel], List[vocab.Vocab], vocab.Vocab]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param dtype: Optional data type to use. If None, will be inferred from stored model.
    :param hybridize: Whether to hybridize the loaded models. Default: true.
    :param inference_only: Use the model only for inference, enabling optimizations.
    :param mc_dropout: Turn on dropout during inference.
    :param allow_missing: Allow missing parameters in the loaded models.
    :param set_grad_req_null: Set grad_req to null for model parameters.
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
                                              hybridize=hybridize,
                                              inference_only=inference_only,
                                              mc_dropout=mc_dropout,
                                              allow_missing=allow_missing,
                                              set_grad_req_null=set_grad_req_null)
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

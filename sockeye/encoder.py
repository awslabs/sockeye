# Copyright 2017--2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Encoders for sequence-to-sequence models.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

import mxnet as mx

from . import config
from . import constants as C
from . import layers
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


ImageEncoderConfig = None


def get_encoder(config: 'EncoderConfig', dtype: str = C.DTYPE_FP32) -> 'Encoder':
    return get_transformer_encoder(config, dtype)


def get_transformer_encoder(config: transformer.TransformerConfig, dtype) -> 'Encoder':
    """
    Returns a Transformer encoder, consisting of an embedding layer with
    positional encodings and a TransformerEncoder instance.

    :param config: Configuration for transformer encoder.
    :return: Encoder instance.
    """
    return TransformerEncoder(config=config, dtype=dtype)


class Encoder(ABC, mx.gluon.HybridBlock):
    """
    Generic encoder interface.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        mx.gluon.HybridBlock.__init__(self, **kwargs)

    def forward(self, inputs, valid_length):  # pylint: disable=arguments-differ
        return mx.gluon.HybridBlock.forward(self, inputs, valid_length)

    def __call__(self, inputs, valid_length):  #pylint: disable=arguments-differ
        """
        Encodes inputs given valid lengths of individual examples.

        :param inputs: Input data.
        :param valid_length: Length of inputs without padding.
        :return: Encoded versions of input data (data, data_length).
        """
        return mx.gluon.HybridBlock.__call__(self, inputs, valid_length)

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this encoder.
        """
        raise NotImplementedError()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        :return: The size of the encoded sequence.
        """
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        return None

@dataclass
class FactorConfig(config.Config):
    vocab_size: int
    num_embed: int
    combine: str  # From C.FACTORS_COMBINE_CHOICES
    share_embedding: bool


@dataclass
class EmbeddingConfig(config.Config):
    vocab_size: int
    num_embed: int
    dropout: float
    num_factors: int = field(init=False)
    factor_configs: Optional[List[FactorConfig]] = None
    allow_sparse_grad: bool = False

    def __post_init__(self):
        self.num_factors = 1
        if self.factor_configs is not None:
            self.num_factors += len(self.factor_configs)


class Embedding(Encoder):
    """
    Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.

    :param config: Embedding config.
    :param dtype: Data type. Default: 'float32'.
    """

    def __init__(self,
                 config: EmbeddingConfig,
                 embed_weight: Optional[mx.gluon.Parameter] = None,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.config = config
        self._dtype = dtype
        self._factor_weight_format_string = 'factor%d_weight'

        if embed_weight is None:
            self.embed_weight = mx.gluon.Parameter('weight',
                                                   shape=(self.config.vocab_size, self.config.num_embed),
                                                   grad_stype='row_sparse',
                                                   dtype=dtype)
            self._use_sparse_grad = self.config.allow_sparse_grad
        else:
            self.embed_weight = embed_weight
            self._use_sparse_grad = embed_weight._grad_stype == 'row_sparse' and self.config.allow_sparse_grad

        self.factor_weights = []  # type: List[mx.gluon.Parameter]
        if self.config.factor_configs is not None:
            for i, fc in enumerate(self.config.factor_configs, 1):
                factor_weight_name = self._factor_weight_format_string % i
                factor_weight = embed_weight if fc.share_embedding else \
                    mx.gluon.Parameter(factor_weight_name, shape=(fc.vocab_size, fc.num_embed), dtype=dtype)
                # We set the attribute of the class to register the parameter with the block
                setattr(self, factor_weight_name, factor_weight)
                self.factor_weights.append(factor_weight)

    def forward(self, data, valid_length):  # pylint: disable=arguments-differ
        # We will catch the optional factor weights in kwargs
        average_factors_embeds = []  # type: List[Union[mx.sym.Symbol, mx.nd.ndarray]]
        concat_factors_embeds = []  # type: List[Union[mx.sym.Symbol, mx.nd.ndarray]]
        sum_factors_embeds = []  # type: List[Union[mx.sym.Symbol, mx.nd.ndarray]]
        if self.config.num_factors > 1 and self.config.factor_configs is not None:
            data, *data_factors = mx.nd.split(data=data,
                                              num_outputs=self.config.num_factors,
                                              axis=2,
                                              squeeze_axis=True)
            for i, (factor_data, factor_config) in enumerate(zip(data_factors,
                                                                 self.config.factor_configs)):
                factor_weight = self.factor_weights[i]
                factor_embedding = mx.nd.Embedding(data=factor_data,
                                                   input_dim=factor_config.vocab_size,
                                                   weight=factor_weight.data(),
                                                   output_dim=factor_config.num_embed)
                if factor_config.combine == C.FACTORS_COMBINE_CONCAT:
                    concat_factors_embeds.append(factor_embedding)
                elif factor_config.combine == C.FACTORS_COMBINE_SUM:
                    sum_factors_embeds.append(factor_embedding)
                elif factor_config.combine == C.FACTORS_COMBINE_AVERAGE:
                    average_factors_embeds.append(factor_embedding)
                else:
                    raise ValueError("Unknown combine value for factors: %s" % factor_config.combine)
        else:
            data = mx.nd.squeeze(data, axis=2)

        embed = mx.nd.Embedding(data,
                                weight=self.embed_weight.data(),
                                input_dim=self.config.vocab_size,
                                output_dim=self.config.num_embed,
                                dtype=self._dtype,
                                sparse_grad=self._use_sparse_grad)

        if self.config.num_factors > 1 and self.config.factor_configs is not None:
            if average_factors_embeds:
                embed = mx.nd.add_n(embed, *average_factors_embeds) / (len(average_factors_embeds) + 1)
            if sum_factors_embeds:
                embed = mx.nd.add_n(embed, *sum_factors_embeds)
            if concat_factors_embeds:
                embed = mx.nd.concat(embed, *concat_factors_embeds, dim=2)

        if self.config.dropout > 0:
            embed = mx.nd.Dropout(data=embed, p=self.config.dropout)

        return embed, mx.nd.identity(valid_length)  # identity: See https://github.com/apache/incubator-mxnet/issues/14228

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.num_embed


class EncoderSequence(Encoder, mx.gluon.nn.HybridSequential):
    """
    A sequence of encoders is itself an encoder.
    """

    def __init__(self, prefix: str = '') -> None:
        Encoder.__init__(self)
        mx.gluon.nn.HybridSequential.__init__(self)

    def add(self, *encoders):
        """Adds block on top of the stack."""
        for encoder in encoders:
            utils.check_condition(isinstance(encoder, Encoder), "%s is not of type Encoder" % encoder)
        mx.gluon.nn.HybridSequential.add(self, *encoders)

    def hybrid_forward(self, F, data, valid_length):  # pylint: disable=arguments-differ
        for block in self._children.values():
            data, valid_length = block(data, valid_length)
        return data, F.identity(valid_length)  # identity: See https://github.com/apache/incubator-mxnet/issues/14228

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return next(reversed(self._children.values())).get_num_hidden()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        Returns the size of the encoded sequence.
        """
        for encoder in self._children.values():
            seq_len = encoder.get_encoded_seq_len(seq_len)
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        max_seq_len = min((encoder.get_max_seq_len()
                           for encoder in self._children.values() if encoder.get_max_seq_len() is not None), default=None)
        return max_seq_len

    def append(self, cls, infer_hidden: bool = False, **kwargs) -> Encoder:
        """
        Extends sequence with new Encoder.

        :param cls: Encoder type.
        :param infer_hidden: If number of hidden should be inferred from previous encoder.
        :param kwargs: Named arbitrary parameters for Encoder.

        :return: Instance of Encoder.
        """
        params = dict(kwargs)
        if infer_hidden:
            params['num_hidden'] = self.get_num_hidden()

        encoder = cls(**params)
        self.add(encoder)
        return encoder


class TransformerEncoder(Encoder, mx.gluon.HybridBlock):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.config = config

        self.pos_embedding = layers.PositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                         num_embed=self.config.model_size,
                                                         max_seq_len=self.config.max_seq_len_source,
                                                         prefix=C.SOURCE_POSITIONAL_EMBEDDING_PREFIX,
                                                         scale_up_input=True,
                                                         scale_down_positions=False)

        self.layers = mx.gluon.nn.HybridSequential()
        for i in range(config.num_layers):
            self.layers.add(transformer.TransformerEncoderBlock(config, dtype=dtype))

        self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                 dropout=config.dropout_prepost,
                                                                 num_hidden=self.config.model_size)

    def forward(self, data, valid_length):
        # positional embedding
        data = self.pos_embedding(data, None)

        if self.config.dropout_prepost > 0.0:
            data = mx.nd.Dropout(data=data, p=self.config.dropout_prepost)

        # (batch_size * heads, seq_len)
        att_valid_length = layers.prepare_source_valid_lengths(valid_length, data,
                                                               num_heads=self.config.attention_heads)

        data = mx.nd.transpose(data, axes=(1, 0, 2))
        for block in self.layers:
            data = block(data, att_valid_length)
        data = self.final_process(data, None)
        data = mx.nd.transpose(data, axes=(1, 0, 2))
        return data, valid_length

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size


EncoderConfig = Union[transformer.TransformerConfig]

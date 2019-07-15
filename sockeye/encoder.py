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

"""
Encoders for sequence-to-sequence models.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import mxnet as mx

from . import config
from . import constants as C
from . import layers
from . import transformer
from . import utils

logger = logging.getLogger(__name__)


ImageEncoderConfig = None


def get_encoder(config: 'EncoderConfig', prefix: str = '') -> 'Encoder':
    return get_transformer_encoder(config, prefix)


def get_transformer_encoder(config: transformer.TransformerConfig, prefix: str) -> 'Encoder':
    """
    Returns a Transformer encoder, consisting of an embedding layer with
    positional encodings and a TransformerEncoder instance.

    :param config: Configuration for transformer encoder.
    :param prefix: Prefix for variable names.
    :return: Encoder instance.
    """
    return TransformerEncoder(config=config, prefix=prefix + C.TRANSFORMER_ENCODER_PREFIX)


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
        :param valid_length: bla.
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


class FactorConfig(config.Config):

    def __init__(self, vocab_size: int, num_embed: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed


class EmbeddingConfig(config.Config):

    def __init__(self,
                 vocab_size: int,
                 num_embed: int,
                 dropout: float,
                 factor_configs: Optional[List[FactorConfig]] = None,
                 source_factors_combine: str = C.SOURCE_FACTORS_COMBINE_CONCAT) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.dropout = dropout
        self.factor_configs = factor_configs
        self.num_factors = 1
        if self.factor_configs is not None:
            self.num_factors += len(self.factor_configs)
        self.source_factors_combine = source_factors_combine


class Embedding(Encoder):
    """
    Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.

    :param config: Embedding config.
    :param prefix: Name prefix for symbols of this encoder.
    :param is_source: Whether this is the source embedding instance. Default: False.
    """

    def __init__(self,
                 config: EmbeddingConfig,
                 prefix: str,
                 is_source: bool = False,
                 embed_weight: Optional[mx.gluon.Parameter] = None) -> None:
        super().__init__(prefix=prefix)
        self.config = config
        self.is_source = is_source

        with self.name_scope():
            if embed_weight is None:
                self.embed_weight = self.params.get('weight', shape=(self.config.vocab_size, self.config.num_embed))
            else:
                self.embed_weight = embed_weight  # adds to self._reg_params
                self.params.update({embed_weight.name: embed_weight})  # adds to self.params

            self.factor_embeds = None
            if self.config.factor_configs is not None:
                self.factor_embeds = mx.gluon.nn.HybridSequential()
                # Factor weights aren't shared so they're not passed in and we create them here.
                for i, fc in enumerate(self.config.factor_configs, 1):
                    self.factor_embeds.add(mx.gluon.nn.Embedding(fc.vocab_size, fc.num_embed,
                                                                 prefix="factor%d_" % i))

    def hybrid_forward(self, F, data, valid_length, embed_weight):  # pylint: disable=arguments-differ
        factor_embeds = []
        if self.is_source:
            if self.config.num_factors > 1 and self.config.factor_configs is not None:
                data, *data_factors = F.split(data, num_outputs=self.config.num_factors, axis=2, squeeze_axis=True)
                factor_embeds = [embed(data) for data, embed in zip(data_factors, self.factor_embeds)]
            else:
                data = F.squeeze(data, axis=2)

        embed = F.Embedding(data,
                            weight=embed_weight,
                            input_dim=self.config.vocab_size,
                            output_dim=self.config.num_embed)

        if factor_embeds:
            if self.config.source_factors_combine == C.SOURCE_FACTORS_COMBINE_CONCAT:
                embed = F.concat(embed, *factor_embeds, dim=2)
            else:
                embed = F.add_n(embed, *factor_embeds)

        if self.config.dropout > 0:
            embed = F.Dropout(data=embed, p=self.config.dropout)

        return embed, F.identity(valid_length)  # identity: See https://github.com/apache/incubator-mxnet/issues/14228

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
        mx.gluon.nn.HybridSequential.__init__(self, prefix=prefix)

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

        sig_params = inspect.signature(cls.__init__).parameters
        encoder = cls(**params)
        self.add(encoder)
        return encoder


class TransformerEncoder(Encoder, mx.gluon.HybridBlock):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    :param prefix: Name prefix for operations in this encoder.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 prefix: str = C.TRANSFORMER_ENCODER_PREFIX) -> None:
        super().__init__(prefix=prefix)
        self.config = config

        with self.name_scope():
            self.pos_embedding = layers.PositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                             num_embed=self.config.model_size,
                                                             max_seq_len=self.config.max_seq_len_source,
                                                             prefix=C.SOURCE_POSITIONAL_EMBEDDING_PREFIX,
                                                             scale_up_input=True,
                                                             scale_down_positions=False)
            self.valid_length_mask = transformer.TransformerValidLengthMask(num_heads=self.config.attention_heads,
                                                                            fold_heads=True,
                                                                            name="bias")

            self.layers = mx.gluon.nn.HybridSequential()
            for i in range(config.num_layers):
                self.layers.add(transformer.TransformerEncoderBlock(config, prefix="%d_" % i))

            self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                     dropout=config.dropout_prepost,
                                                                     prefix="final_process_",
                                                                     num_hidden=self.config.model_size)

    def hybrid_forward(self, F, data, valid_length):
        # positional embedding
        data = self.pos_embedding(data, None)

        if self.config.dropout_prepost > 0.0:
            data = F.Dropout(data=data, p=self.config.dropout_prepost)

        # (batch_size * heads, 1, seq_len)
        bias = F.expand_dims(self.valid_length_mask(data, valid_length), axis=1)

        for block in self.layers:
            data = block(data, bias)

        data = self.final_process(data, None)
        return data, valid_length

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size


EncoderConfig = Union[transformer.TransformerConfig]

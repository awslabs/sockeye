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

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

import torch as pt

import sockeye.constants as C
from . import config
from . import layers_pt
from . import transformer_pt


def pytorch_get_transformer_encoder(config: transformer_pt.TransformerConfig, inference_only: bool = False):
    return PyTorchTransformerEncoder(config=config, inference_only=inference_only)


get_encoder = pytorch_get_transformer_encoder
EncoderConfig = Union[transformer_pt.TransformerConfig]


class PyTorchEncoder(pt.nn.Module):
    """
    Generic encoder interface.
    """

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


class PyTorchEmbedding(PyTorchEncoder):
    """
    Thin wrapper around PyTorch's Embedding op.

    :param config: Embedding config.
    :param embedding: pre-existing embedding Module.
    """

    def __init__(self, config: EmbeddingConfig, embedding: Optional[pt.nn.Embedding] = None) -> None:
        super().__init__()
        self.config = config

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = pt.nn.Embedding(self.config.vocab_size, self.config.num_embed,
                                             sparse=self.config.allow_sparse_grad)

        self.factor_embeds = pt.nn.ModuleList()
        if self.config.factor_configs is not None:
            for i, fc in enumerate(self.config.factor_configs, 1):
                if fc.share_embedding:
                    factor_embed = self.embedding
                else:
                    factor_embed = pt.nn.Embedding(fc.vocab_size, fc.num_embed,
                                                   sparse=self.config.allow_sparse_grad)
                self.factor_embeds.append(factor_embed)

        self.dropout = pt.nn.Dropout(p=self.config.dropout) if self.config.dropout > 0.0 else None

    def forward(self, data: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:  # pylint: disable=arguments-differ
        # We will catch the optional factor weights in kwargs
        average_factors_embeds = []  # type: List[pt.Tensor]
        concat_factors_embeds = []  # type: List[pt.Tensor]
        sum_factors_embeds = []  # type: List[pt.Tensor]

        primary_data = data[:, :, 0]
        embedded = self.embedding(primary_data)

        if self.config.num_factors > 1 and self.config.factor_configs is not None:
            for i, (factor_embedding, factor_config) in enumerate(zip(self.factor_embeds,
                                                                      self.config.factor_configs), 1):
                factor_data = data[:, :, i]
                factor_embedded = factor_embedding(factor_data)
                if factor_config.combine == C.FACTORS_COMBINE_CONCAT:
                    concat_factors_embeds.append(factor_embedded)
                elif factor_config.combine == C.FACTORS_COMBINE_SUM:
                    sum_factors_embeds.append(factor_embedded)
                elif factor_config.combine == C.FACTORS_COMBINE_AVERAGE:
                    average_factors_embeds.append(factor_embedded)
                else:
                    raise ValueError("Unknown combine value for factors: %s" % factor_config.combine)

        if self.config.num_factors > 1 and self.config.factor_configs is not None:
            if average_factors_embeds:
                embedded = pt.mean(pt.stack([embedded] + average_factors_embeds, dim=0), dim=0)
            if sum_factors_embeds:
                embedded = pt.sum(pt.stack([embedded] + sum_factors_embeds, dim=0), dim=0)
            if concat_factors_embeds:
                embedded = pt.cat([embedded] + concat_factors_embeds, dim=2)

        if self.dropout is not None:
            embedded = self.dropout(embedded)

        return embedded

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.num_embed

    def weights_from_mxnet_block(self, block_mx: 'Embedding'):  # type: ignore
        self.embedding.weight.data[:] = pt.as_tensor(block_mx.weight.data().asnumpy())
        if self.config.factor_configs is not None:
            for embedding, mx_weight, fc in zip(self.factor_embeds, block_mx.factor_weights, self.config.factor_configs):
                embedding.weight.data[:] = pt.as_tensor(mx_weight.data().asnumpy())


class PyTorchTransformerEncoder(PyTorchEncoder):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    """

    def __init__(self, config: transformer_pt.TransformerConfig, inference_only: bool = False) -> None:
        pt.nn.Module.__init__(self)
        self.config = config

        self.dropout = pt.nn.Dropout(p=config.dropout_prepost) if config.dropout_prepost > 0.0 else None

        self.pos_embedding = layers_pt.PyTorchPositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                                   num_embed=self.config.model_size,
                                                                   max_seq_len=self.config.max_seq_len_source,
                                                                   scale_up_input=True,
                                                                   scale_down_positions=False)

        self.layers = pt.nn.ModuleList(  # using ModuleList because we have additional inputs
            transformer_pt.PyTorchTransformerEncoderBlock(config, inference_only=inference_only)
            for _ in range(config.num_layers))

        self.final_process = transformer_pt.PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                           dropout=config.dropout_prepost,
                                                                           num_hidden=self.config.model_size)

    def forward(self, data: pt.Tensor, valid_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        # positional embedding
        data = self.pos_embedding(data)

        if self.dropout is not None:
            data = self.dropout(data)

        _, max_len, __ = data.size()
        # length_mask for source attention masking. Shape: (batch_size * heads, 1, max_len)
        att_mask = layers_pt.prepare_source_length_mask(valid_length, self.config.attention_heads, max_length=max_len)
        att_mask = att_mask.repeat(1, max_len, 1)

        data = data.transpose(1, 0)  # batch to time major
        for layer in self.layers:
            data = layer(data, att_mask=att_mask)

        data = self.final_process(data)
        data = data.transpose(1, 0)  # time to batch major
        return data, valid_length

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size

    def weights_from_mxnet_block(self, block_mx: 'TransformerEncoder'):  # type: ignore
        self.pos_embedding.weights_from_mxnet_block(block_mx.pos_embedding)
        for i, l in enumerate(self.layers):
            l.weights_from_mxnet_block(block_mx.layers[i])
        self.final_process.weights_from_mxnet_block(block_mx.final_process)


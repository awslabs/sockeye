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
from typing import Tuple, Optional

import torch as pt

import sockeye.constants as C
from sockeye.encoder import TransformerEncoder
from . import layers_pt
from . import transformer
from . import transformer_pt


def pytorch_get_transformer_encoder(config: transformer.TransformerConfig, dtype):
    return PyTorchTransformerEncoder(config=config, dtype=dtype)


class PyTorchEncoder():
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


class PyTorchTransformerEncoder(PyTorchEncoder, pt.nn.Module):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    """

    def __init__(self,
                 config: transformer.TransformerConfig,
                 dtype: str = C.DTYPE_FP32) -> None:
        pt.nn.Module.__init__(self)
        self.config = config

        self.dropout = pt.nn.Dropout(p=config.dropout_prepost) if config.dropout_prepost > 0.0 else None

        self.pos_embedding = layers_pt.PyTorchPositionalEmbeddings(weight_type=self.config.positional_embedding_type,
                                                                   num_embed=self.config.model_size,
                                                                   max_seq_len=self.config.max_seq_len_source,
                                                                   scale_up_input=True,
                                                                   scale_down_positions=False)

        self.layers = pt.nn.ModuleList(  # using ModuleList because we have additional
            transformer_pt.PyTorchTransformerEncoderBlock(config, dtype) for _ in range(config.num_layers))

        self.final_process = transformer_pt.PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                           dropout=config.dropout_prepost,
                                                                           num_hidden=self.config.model_size)

    def forward(self, data: pt.Tensor, valid_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        # positional embedding
        data = self.pos_embedding(data, None)

        if self.dropout is not None:
            data = self.dropout(data)

        # (batch_size * heads, seq_len)
        att_valid_length = layers_pt.pytorch_prepare_source_valid_lengths(valid_length, self.config.attention_heads)

        data = data.transpose(1, 0)  # batch to time major
        for layer in self.layers:
            data = layer(data, att_valid_length)

        data = self.final_process(data, None)
        data = data.transpose(1, 0)  # time to batch major
        return data, valid_length

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size

    def weights_from_mxnet_block(self, block_mx: TransformerEncoder):
        self.pos_embedding.weights_from_mxnet_block(block_mx.pos_embedding)
        for i, l in enumerate(self.layers):
            l.weights_from_mxnet_block(block_mx.layers[i])
        self.final_process.weights_from_mxnet_block(block_mx.final_process)


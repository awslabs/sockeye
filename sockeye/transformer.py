# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


from dataclasses import dataclass
from typing import Optional, Tuple

import torch as pt

import sockeye.layers
from sockeye import constants as C
from . import config


@dataclass
class TransformerConfig(config.Config):
    model_size: int
    attention_heads: int
    feed_forward_num_hidden: int
    act_type: str
    num_layers: int
    dropout_attention: float
    dropout_act: float
    dropout_prepost: float
    positional_embedding_type: str
    preprocess_sequence: str
    postprocess_sequence: str
    max_seq_len_source: int
    max_seq_len_target: int
    decoder_type: str = C.TRANSFORMER_TYPE
    use_lhuc: bool = False
    depth_key_value: int = 0
    use_glu: bool = False


class TransformerEncoderBlock(pt.nn.Module):
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 inference_only: bool = False,
                 dtype: Optional[pt.dtype] = None,
                 clamp_to_dtype: bool = False) -> None:
        super().__init__()

        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          num_hidden=config.model_size,
                                                          dtype=dtype,
                                                          clamp_to_dtype=clamp_to_dtype)
        self.self_attention = sockeye.layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                                    heads=config.attention_heads,
                                                                    depth_out=config.model_size,
                                                                    dropout=config.dropout_attention,
                                                                    dtype=dtype,
                                                                    clamp_to_dtype=clamp_to_dtype)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           num_hidden=config.model_size,
                                                           dtype=dtype,
                                                           clamp_to_dtype=clamp_to_dtype)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              num_hidden=config.model_size,
                                              dtype=dtype,
                                              clamp_to_dtype=clamp_to_dtype)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         use_glu=config.use_glu,
                                         inference_only=inference_only,
                                         dtype=dtype,
                                         clamp_to_dtype=clamp_to_dtype)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               num_hidden=config.model_size,
                                               dtype=dtype,
                                               clamp_to_dtype=clamp_to_dtype)
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers.LHUC(config.model_size, dtype=dtype)

    def forward(self, data: pt.Tensor, att_mask: pt.Tensor = None) -> pt.Tensor:
        """
        :param data: Input tensor of shape (length, batch_size, hidden)
        :param att_mask: Optional data length mask of shape (batch_size * self.heads, 1, length)
                         to mask self-attention scores. True for padding positions.
        """
        # self-attention
        data_self_att, _ = self.self_attention(inputs=self.pre_self_attention(data),
                                               previous_states=None,
                                               mask=att_mask,
                                               bias=None)
        data = self.post_self_attention(data_self_att, data)

        # feed-forward
        data_ff = self.ff(self.pre_ff(data))

        data = self.post_ff(data_ff, data)

        if self.lhuc is not None:
            data = self.lhuc(data)

        return data


class TransformerDecoderBlock(pt.nn.Module):
    """
    A transformer decoder block consists of an autoregressive attention block, encoder attention,
    and a feed-forward layer with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 inference_only: bool,
                 dtype: Optional[pt.dtype] = None,
                 clamp_to_dtype: bool = False) -> None:
        super().__init__()
        self.decoder_type = config.decoder_type

        self.autoregr_layer = None
        if self.decoder_type == C.TRANSFORMER_TYPE:
            self.autoregr_layer = sockeye.layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                                        heads=config.attention_heads,
                                                                        depth_out=config.model_size,
                                                                        dropout=config.dropout_attention,
                                                                        dtype=dtype,
                                                                        clamp_to_dtype=clamp_to_dtype)
        elif self.decoder_type == C.SSRU_TRANSFORMER:
            self.autoregr_layer = sockeye.layers.SSRU(model_size=config.model_size,  # type: ignore
                                                      inference_only=inference_only,  # type: ignore
                                                      dtype=dtype,
                                                      clamp_to_dtype=clamp_to_dtype)
        else:
            raise ValueError("Invalid decoder type.")

        self.pre_autoregr_layer = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          num_hidden=config.model_size,
                                                          dtype=dtype,
                                                          clamp_to_dtype=clamp_to_dtype)

        self.post_autoregr_layer = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           num_hidden=config.model_size,
                                                           dtype=dtype,
                                                           clamp_to_dtype=clamp_to_dtype)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         dropout=config.dropout_prepost,
                                                         num_hidden=config.model_size,
                                                         dtype=dtype,
                                                         clamp_to_dtype=clamp_to_dtype)
        self.enc_attention = sockeye.layers.MultiHeadAttention(depth_att=config.model_size,
                                                               heads=config.attention_heads,
                                                               depth_out=config.model_size,
                                                               dropout=config.dropout_attention,
                                                               depth_key_value=config.depth_key_value,
                                                               dtype=dtype,
                                                               clamp_to_dtype=clamp_to_dtype)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          num_hidden=config.model_size,
                                                          dtype=dtype,
                                                          clamp_to_dtype=clamp_to_dtype)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              num_hidden=config.model_size,
                                              dtype=dtype,
                                              clamp_to_dtype=clamp_to_dtype)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         use_glu=config.use_glu,
                                         inference_only=inference_only,
                                         dtype=dtype,
                                         clamp_to_dtype=clamp_to_dtype)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               num_hidden=config.model_size,
                                               dtype=dtype,
                                               clamp_to_dtype=clamp_to_dtype)

        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers.LHUC(config.model_size, dtype=dtype)

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return self.autoregr_layer.num_state_tensors

    @property
    def needs_mask(self):
        """ Whether the block makes use of a mask tensor or not """
        return self.autoregr_layer.needs_mask

    def get_states_shape(self, batch_size: int) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of an output state (assuming all of them have the same shape)
        """
        return self.autoregr_layer.get_state_shape(batch_size)

    def forward(self,
                target: pt.Tensor,
                target_mask: Optional[pt.Tensor],
                source: pt.Tensor,
                source_mask: Optional[pt.Tensor],
                autoregr_states: Optional[pt.Tensor],
                enc_att_kv: Optional[pt.Tensor] = None) -> Tuple[pt.Tensor, pt.Tensor]:
        target_autoregr, *new_autoregr_states = self.autoregr_layer(inputs=self.pre_autoregr_layer(target),
                                                                    previous_states=autoregr_states,
                                                                    mask=target_mask)

        target = self.post_autoregr_layer(target_autoregr, target)

        # encoder attention
        target_enc_att = self.enc_attention(queries=self.pre_enc_attention(target),
                                            key_values=source,
                                            mask=source_mask,
                                            projected_memory_kv=enc_att_kv)

        target = self.post_enc_attention(target_enc_att, target)

        # feed-forward
        target_ff = self.ff(self.pre_ff(target))
        target = self.post_ff(target_ff, target)

        if self.lhuc:
            target = self.lhuc(target)

        return target, new_autoregr_states


class TransformerProcessBlock(pt.nn.Module):
    """
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self,
                 sequence: str,
                 dropout: float,
                 num_hidden: int = 0,
                 dtype: Optional[pt.dtype] = None,
                 clamp_to_dtype: bool = False) -> None:
        super().__init__()
        self.sequence = sequence
        self.clamp_to_dtype = clamp_to_dtype
        self.layer_norm = None
        if 'n' in sequence:
            # do not use Apex' FusedLayerNorm because of
            # https://github.com/huggingface/transformers/issues/9377
            self.layer_norm = pt.nn.LayerNorm(num_hidden, eps=1e-06, dtype=dtype)
        self.dropout = dropout
        if dropout > 0.0:
            self.drop = pt.nn.Dropout(p=dropout)

    def forward(self, data: pt.Tensor, prev: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        Apply processing sequence to data with optional previous input.

        :param data: Input data. Shape: (batch, length, num_hidden).
        :param prev: Previous data. Shape: (batch, length, num_hidden).
        :return: Processed data. Shape: (batch, length, num_hidden).
        """
        if not self.sequence:
            return data

        if prev is None:
            assert 'r' not in self.sequence, "Residual connection not allowed if no previous value given."

        for step in self.sequence:

            if step == "r":
                data = data + prev

            elif step == "n":
                data = self.layer_norm(data)

            elif step == "d":
                if self.dropout > 0.0:
                    data = self.drop(data)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        if self.clamp_to_dtype:
            data = sockeye.layers.clamp_to_dtype_min_max(data)

        return data


class TransformerFeedForward(pt.nn.Module):

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 use_glu: bool = False,
                 inference_only: bool = False,
                 dtype: Optional[pt.dtype] = None,
                 clamp_to_dtype: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_glu = use_glu
        self.clamp_to_dtype = clamp_to_dtype
        self.ff1 = pt.nn.Linear(in_features=num_model, out_features=num_hidden, dtype=dtype)
        self.act = sockeye.layers.get_activation(act_type, inplace=inference_only)
        if self.use_glu:
            self.linear = pt.nn.Linear(in_features=num_model, out_features=num_hidden, dtype=dtype)
        if self.dropout > 0.0:
            self.drop = pt.nn.Dropout(p=self.dropout, inplace=inference_only)
        self.ff2 = pt.nn.Linear(in_features=num_hidden, out_features=num_model, dtype=dtype)

    def forward(self, x):
        h = self.ff1(x)
        h = self.act(h)
        if self.use_glu:
            h = h * self.linear(x)
        if self.dropout > 0.0:
            h = self.drop(h)
        y = self.ff2(h)
        if self.clamp_to_dtype:
            y = sockeye.layers.clamp_to_dtype_min_max(y)
        return y


class AutoRegressiveMask(pt.nn.Module):

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        """ Input tensor with length on dimension 1 """
        mask = pt.full((x.shape[1], x.shape[1]), fill_value=1, device=x.device, dtype=pt.bool)
        mask = pt.triu(mask, diagonal=1)
        return mask.detach()  # Shape: (len, len)

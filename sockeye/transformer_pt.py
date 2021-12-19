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


from dataclasses import dataclass
from typing import Optional, Tuple

import torch as pt

import sockeye.layers_pt
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


class PyTorchTransformerEncoderBlock(pt.nn.Module):
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self, config: TransformerConfig, inference_only: bool = False) -> None:
        super().__init__()

        self.pre_self_attention = PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                 dropout=config.dropout_prepost,
                                                                 num_hidden=config.model_size)
        self.self_attention = sockeye.layers_pt.PyTorchMultiHeadSelfAttention(depth_att=config.model_size,
                                                                              heads=config.attention_heads,
                                                                              depth_out=config.model_size,
                                                                              dropout=config.dropout_attention)
        self.post_self_attention = PyTorchTransformerProcessBlock(sequence=config.postprocess_sequence,
                                                                  dropout=config.dropout_prepost,
                                                                  num_hidden=config.model_size)

        self.pre_ff = PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                     dropout=config.dropout_prepost,
                                                     num_hidden=config.model_size)
        self.ff = PyTorchTransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                num_model=config.model_size,
                                                act_type=config.act_type,
                                                dropout=config.dropout_act,
                                                use_glu=config.use_glu,
                                                inference_only=inference_only)
        self.post_ff = PyTorchTransformerProcessBlock(sequence=config.postprocess_sequence,
                                                      dropout=config.dropout_prepost,
                                                      num_hidden=config.model_size)
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers_pt.PyTorchLHUC(config.model_size)

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

    def weights_from_mxnet_block(self, block_mx: 'TransformerEncoderBlock'):  # type: ignore
        self.pre_self_attention.weights_from_mxnet_block(block_mx.pre_self_attention)
        self.self_attention.weights_from_mxnet_block(block_mx.self_attention)
        self.post_self_attention.weights_from_mxnet_block(block_mx.post_self_attention)
        self.pre_ff.weights_from_mxnet_block(block_mx.pre_ff)
        self.ff.weights_from_mxnet_block(block_mx.ff)
        self.post_ff.weights_from_mxnet_block(block_mx.post_ff)
        if self.lhuc is not None:
            self.lhuc.weights_from_mxnet_block(block_mx.lhuc)


class PyTorchTransformerDecoderBlock(pt.nn.Module):
    """
    A transformer decoder block consists of an autoregressive attention block, encoder attention,
    and a feed-forward layer with pre/post process blocks in between.
    """

    def __init__(self, config: TransformerConfig, inference_only: bool) -> None:
        super().__init__()
        self.decoder_type = config.decoder_type

        self.autoregr_layer = None
        if self.decoder_type == C.TRANSFORMER_TYPE:
            self.autoregr_layer = sockeye.layers_pt.PyTorchMultiHeadSelfAttention(depth_att=config.model_size,
                                                                                  heads=config.attention_heads,
                                                                                  depth_out=config.model_size,
                                                                                  dropout=config.dropout_attention)
        elif self.decoder_type == C.SSRU_TRANSFORMER:
            self.autoregr_layer = sockeye.layers_pt.PyTorchSSRU(model_size=config.model_size,  # type: ignore
                                                                inference_only=inference_only)  # type: ignore
        else:
            raise ValueError("Invalid decoder type.")

        self.pre_autoregr_layer = PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                 dropout=config.dropout_prepost,
                                                                 num_hidden=config.model_size)

        self.post_autoregr_layer = PyTorchTransformerProcessBlock(sequence=config.postprocess_sequence,
                                                                  dropout=config.dropout_prepost,
                                                                  num_hidden=config.model_size)

        self.pre_enc_attention = PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                dropout=config.dropout_prepost,
                                                                num_hidden=config.model_size)
        self.enc_attention = sockeye.layers_pt.PyTorchMultiHeadAttention(depth_att=config.model_size,
                                                                         heads=config.attention_heads,
                                                                         depth_out=config.model_size,
                                                                         dropout=config.dropout_attention,
                                                                         depth_key_value=config.depth_key_value)
        self.post_enc_attention = PyTorchTransformerProcessBlock(sequence=config.postprocess_sequence,
                                                                 dropout=config.dropout_prepost,
                                                                 num_hidden=config.model_size)

        self.pre_ff = PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                     dropout=config.dropout_prepost,
                                                     num_hidden=config.model_size)
        self.ff = PyTorchTransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                                num_model=config.model_size,
                                                act_type=config.act_type,
                                                dropout=config.dropout_act,
                                                use_glu=config.use_glu,
                                                inference_only=inference_only)
        self.post_ff = PyTorchTransformerProcessBlock(sequence=config.postprocess_sequence,
                                                      dropout=config.dropout_prepost,
                                                      num_hidden=config.model_size)

        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers_pt.PyTorchLHUC(config.model_size)

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

    def weights_from_mxnet_block(self, block_mx: 'TransformerDecoderBlock'):  # type: ignore
        self.pre_autoregr_layer.weights_from_mxnet_block(block_mx.pre_autoregr_layer)
        self.autoregr_layer.weights_from_mxnet_block(block_mx.autoregr_layer)
        self.post_autoregr_layer.weights_from_mxnet_block(block_mx.post_autoregr_layer)
        self.pre_enc_attention.weights_from_mxnet_block(block_mx.pre_enc_attention)
        self.enc_attention.weights_from_mxnet_block(block_mx.enc_attention)
        self.post_enc_attention.weights_from_mxnet_block(block_mx.post_enc_attention)
        self.pre_ff.weights_from_mxnet_block(block_mx.pre_ff)
        self.ff.weights_from_mxnet_block(block_mx.ff)
        self.post_ff.weights_from_mxnet_block(block_mx.post_ff)
        if self.lhuc is not None:
            self.lhuc.weights_from_mxnet_block(block_mx.lhuc)


class PyTorchTransformerProcessBlock(pt.nn.Module):
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
                 num_hidden: int = 0) -> None:
        super().__init__()
        self.sequence = sequence
        self.layer_norm = None
        if 'n' in sequence:
            # do not use Apex' FusedLayerNorm because of
            # https://github.com/huggingface/transformers/issues/9377
            self.layer_norm = pt.nn.LayerNorm(num_hidden, eps=1e-06)
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

        return data

    def weights_from_mxnet_block(self, block_mx: 'TransformerProcessBlock'):  # type: ignore
        if 'n' in self.sequence:
            assert 'n' in block_mx.sequence
            self.layer_norm.bias.data[:] = pt.as_tensor(block_mx.layer_norm.beta.data().asnumpy())
            self.layer_norm.weight.data[:] = pt.as_tensor(block_mx.layer_norm.gamma.data().asnumpy())


class PyTorchTransformerFeedForward(pt.nn.Module):

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 use_glu: bool = False,
                 inference_only: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_glu = use_glu
        self.ff1 = pt.nn.Linear(in_features=num_model, out_features=num_hidden)
        self.act = sockeye.layers_pt.pytorch_get_activation(act_type, inplace=inference_only)
        if self.use_glu:
            self.linear = pt.nn.Linear(in_features=num_model, out_features=num_hidden)
        if self.dropout > 0.0:
            self.drop = pt.nn.Dropout(p=self.dropout, inplace=inference_only)
        self.ff2 = pt.nn.Linear(in_features=num_hidden, out_features=num_model)

    def forward(self, x):
        h = self.ff1(x)
        h = self.act(h)
        if self.use_glu:
            h = h * self.linear(x)
        if self.dropout > 0.0:
            h = self.drop(h)
        y = self.ff2(h)
        return y

    def weights_from_mxnet_block(self, block_mx: 'TransformerFeedForward'):  # type: ignore
        self.ff1.weight.data[:] = pt.as_tensor(block_mx.ff1.weight.data().asnumpy())
        self.ff2.weight.data[:] = pt.as_tensor(block_mx.ff2.weight.data().asnumpy())
        self.ff1.bias.data[:] = pt.as_tensor(block_mx.ff1.bias.data().asnumpy())
        self.ff2.bias.data[:] = pt.as_tensor(block_mx.ff2.bias.data().asnumpy())
        if self.use_glu:
            self.linear.weight.data[:] = pt.as_tensor(block_mx.linear.weight.data().asnumpy())
            self.linear.bias.data[:] = pt.as_tensor(block_mx.linear.bias.data().asnumpy())


class AutoRegressiveMask(pt.nn.Module):

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        """ Input tensor with length on dimension 1 """
        mask = pt.full((x.shape[1], x.shape[1]), fill_value=1, device=x.device, dtype=pt.bool)
        mask = pt.triu(mask, diagonal=1)
        return mask.detach()  # Shape: (len, len)

# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


import torch as pt

import sockeye.layers_pt
from sockeye import constants as C
from sockeye.transformer import TransformerConfig, TransformerEncoderBlock, TransformerProcessBlock


class PyTorchTransformerEncoderBlock(pt.nn.Module):
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 dtype: str) -> None:
        super().__init__()

        self.pre_self_attention = PyTorchTransformerProcessBlock(sequence=config.preprocess_sequence,
                                                                 dropout=config.dropout_prepost,
                                                                 num_hidden=config.model_size)
        self.self_attention = sockeye.layers_pt.PyTorchMultiHeadSelfAttention(depth_att=config.model_size,
                                                                              heads=config.attention_heads,
                                                                              depth_out=config.model_size,
                                                                              dropout=config.dropout_attention,
                                                                              dtype=dtype)
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
                                                dtype=dtype,
                                                use_glu=config.use_glu)
        self.post_ff = PyTorchTransformerProcessBlock(sequence=config.postprocess_sequence,
                                                      dropout=config.dropout_prepost,
                                                      num_hidden=config.model_size)
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers_pt.PyTorchLHUC(config.model_size)

    def forward(self, data: pt.Tensor, lengths: pt.Tensor) -> pt.Tensor:
        # self-attention
        data_self_att, _ = self.self_attention(self.pre_self_attention(data, None), None, lengths, None)
        data = self.post_self_attention(data_self_att, data)

        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))

        data = self.post_ff(data_ff, data)

        if self.lhuc is not None:
            data = self.lhuc(data)

        return data

    def weights_from_mxnet_block(self, block_mx: TransformerEncoderBlock):
        self.pre_self_attention.weights_from_mxnet_block(block_mx.pre_self_attention)
        self.self_attention.weights_from_mxnet_block(block_mx.self_attention)
        self.post_self_attention.weights_from_mxnet_block(block_mx.post_self_attention)
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
            self.layer_norm = pt.nn.LayerNorm(num_hidden, eps=1e-06)
        self.dropout = None  # type: Optional[pt.nn.Module]
        if dropout > 0.0:
            self.dropout = pt.nn.Dropout(p=dropout)

    def forward(self, data: pt.Tensor, prev: pt.Tensor) -> pt.Tensor:
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

            elif step == "d" and self.dropout is not None:
                data = self.dropout(data)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        return data

    def weights_from_mxnet_block(self, block_mx: TransformerProcessBlock):
        if 'n' in self.sequence:
            assert 'n' in block_mx.sequence
            self.layer_norm.bias[:] = pt.as_tensor(block_mx.layer_norm.beta.data().asnumpy())
            self.layer_norm.weight[:] = pt.as_tensor(block_mx.layer_norm.gamma.data().asnumpy())


class PyTorchTransformerFeedForward(pt.nn.Module):

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 dtype: str,
                 use_glu: bool = False) -> None:
        super().__init__()
        assert dtype == C.DTYPE_FP32
        self.dropout = dropout
        self.use_glu = use_glu
        self.ff1 = pt.nn.Linear(in_features=num_model, out_features=num_hidden)
        self.act = sockeye.layers_pt.pytorch_get_activation(act_type)
        if self.use_glu:
            self.linear = pt.nn.Linear(in_features=num_model, out_features=num_hidden)
        if self.dropout > 0.0:
            self.drop = pt.nn.Dropout(p=self.dropout, inplace=True)
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

    def weights_from_mxnet_block(self, block_mx: 'TransformerFeedForward'):
        self.ff1.weight[:] = pt.as_tensor(block_mx.ff1.weight.data().asnumpy())
        self.ff2.weight[:] = pt.as_tensor(block_mx.ff2.weight.data().asnumpy())
        self.ff1.bias[:] = pt.as_tensor(block_mx.ff1.bias.data().asnumpy())
        self.ff2.bias[:] = pt.as_tensor(block_mx.ff2.bias.data().asnumpy())
        if self.use_glu:
            self.linear.weight[:] = pt.as_tensor(block_mx.linear.weight.data().asnumpy())
            self.linear.bias[:] = pt.as_tensor(block_mx.linear.bias.data().asnumpy())


class PyTorchAutoRegressiveBias(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self._dtype = 'float32'

    def forward(self, x):
        bias = pt.ones(x.shape[1], x.shape[1], device=x.device) * -C.LARGE_VALUES[self._dtype]
        bias = pt.triu(bias, diagonal=1).unsqueeze(0).detach()
        return bias
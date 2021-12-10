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

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch as pt
import torch.nn.functional as F

from sockeye import constants as C, utils
from . import config

logger = logging.getLogger(__name__)


def pytorch_get_activation(act_type: str, inplace: bool = False) -> pt.nn.Module:
    if act_type == C.SWISH1:
        return pt.nn.SiLU(inplace=inplace)
    if act_type == C.GELU:
        return pt.nn.GELU()
    return pt.nn.ReLU(inplace=inplace)


class PyTorchLHUC(pt.nn.Module):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    """

    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.weight = pt.nn.Parameter(pt.Tensor(num_hidden,))

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight = 2 * pt.sigmoid(self.weight)
        return weight * data

    def weights_from_mxnet_block(self, block_mx: 'LHUC'):  # type: ignore
        self.weight.data[:] = pt.as_tensor(block_mx.weight.data().asnumpy())


class PyTorchWeightNormalization(pt.nn.Module):
    """
    Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
    For a given tensor the normalization is done per hidden dimension.

    :param num_hidden: Size of the first dimension.
    :param ndim: The total number of dimensions of the weight tensor.
    """

    def __init__(self, num_hidden: int, ndim: int = 2) -> None:
        super().__init__()
        _shape = tuple([num_hidden] + [1] * (ndim - 1))
        self.scale = pt.ones(*_shape)
        self._axis_arg = tuple(range(1, ndim))

    def forward(self, weight: pt.Tensor) -> pt.Tensor:
        return F.normalize(weight, p=2, dim=self._axis_arg, eps=0) * self.scale  # type: ignore


class PyTorchOutputLayer(pt.nn.Module):
    """
    Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.

    :param hidden_size: Input hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight: Optional shared weight Parameter.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 weight: Optional[pt.nn.Parameter] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = hidden_size
        self.out_features = vocab_size

        if weight is None:
            self.weight = pt.nn.Parameter(pt.Tensor(vocab_size, hidden_size))
        else:
            self.weight = weight
        self.bias = pt.nn.Parameter(pt.Tensor(vocab_size))

        self.previous_slice_ids = None  # type: Optional[pt.Tensor]
        self.reduced_weight_bias = None  # type: Optional[Tuple[pt.Tensor, pt.Tensor]]

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={} dtype={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.weight.dtype)

    def _is_new_slice(self, x: pt.Tensor) -> bool:
        if self.previous_slice_ids is None or \
                x.size() != self.previous_slice_ids.size() or \
                pt.any(x != self.previous_slice_ids):
            return True
        return False

    def _take_slice(self, vocab_slice_ids: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        weight = self.weight[vocab_slice_ids]  # Shape: (len(vocab_slice_ids), hidden)
        bias = self.bias[vocab_slice_ids]
        return weight, bias

    def forward(self, data: pt.Tensor, vocab_slice_ids: Optional[pt.Tensor] = None) -> pt.Tensor:
        if vocab_slice_ids is not None:
            # Imperative, reduced matrix multiplication for vocabulary selection.
            # vocab_slice_ids is constant across decoder step calls, so we cache the result of _take_slice
            # across decoder steps. If a new vocab_slice_ids tensor is observed, we re-run _take_slice.
            # This significantly reduces latency for CPU decoding.
            if self._is_new_slice(vocab_slice_ids):
                self.previous_slice_ids = vocab_slice_ids
                weight, bias = self.reduced_weight_bias = self._take_slice(vocab_slice_ids)
            else:
                weight, bias = self.reduced_weight_bias
        else:
            weight, bias = self.weight, self.bias

        return F.linear(data, weight, bias)

    def weights_from_mxnet_block(self, block_mx: 'OutputLayer'):  # type: ignore
        self.weight.data[:] = pt.as_tensor(block_mx.weight.data().asnumpy())
        self.bias.data[:] = pt.as_tensor(block_mx.bias.data().asnumpy())


@dataclass
class LengthRatioConfig(config.Config):
    num_layers: int  # Number of layers
    weight: float  # Weight of this loss


class PyTorchLengthRatio(pt.nn.Module):
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int) -> None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        modules = []  # type: List[pt.nn.Module]
        for _ in range(num_layers - 1):
            modules.append(pt.nn.Linear(in_features=hidden_size, out_features=hidden_size))
            modules.append(pt.nn.Tanh())
        modules.append(pt.nn.Linear(in_features=hidden_size, out_features=1))
        modules.append(pt.nn.Softplus())  # SoftReLU activation to ensure positiveness of the predicted length ratio
        self.layers = pt.nn.Sequential(*modules)

    def forward(self, source_encoded: pt.Tensor, source_encoded_length: pt.Tensor) -> pt.Tensor:
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        # True when outside length. Shape: (n, source_encoded_length, 1)
        mask = pt.arange(source_encoded.size()[1], device=source_encoded_length.device)[None, :, None] >= source_encoded_length[:, None, None]
        source_masked = source_encoded.masked_fill(mask, 0.)

        # data: (n, hidden_size)
        data = source_masked.sum(dim=1, keepdim=False) / source_encoded_length.unsqueeze(1)
        data = self.layers(data).squeeze(1)  # (n, 1)
        return data

    def weights_from_mxnet_block(self, block_mx: 'LengthRatio'):  # type: ignore
        for l_pt, l_mx in zip(self.layers, block_mx.layers):
            if isinstance(l_pt, pt.nn.Linear):
                l_pt.weight.data[:] = pt.as_tensor(l_mx.weight.data().asnumpy())
                l_pt.bias.data[:] = pt.as_tensor(l_mx.bias.data().asnumpy())


# TODO: port NVIDIAs implementation to PT C++ custom op
@pt.jit.script
def pytorch_interleaved_matmul_encdec_qk(q: pt.Tensor,
                                         kv: pt.Tensor,
                                         heads: int) -> pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_qk with PyTorch.

    :param q: (qlen, batch, hidden)
    :param kv: (kvlen, batch, hidden * 2) -- interleaved
    :param heads: number of attention heads
    :return: (batch * heads, qlen, klen)
    """
    qlen, batch, hidden = q.size()
    head_dim = hidden // heads

    # batch * heads, qlen, head_dim)
    q = q.contiguous().view(qlen, batch * heads, head_dim).transpose(0, 1)
    q = q * head_dim ** -0.5

    tmp = kv.reshape(-1, batch, heads, 2, head_dim)
    k = tmp[:, :, :, 0, :]  # pick keys
    k = k.permute(1, 2, 3, 0)  # (batch, heads, head_dim, kvlen)
    k = k.reshape(batch * heads, head_dim, -1)  # (batch * heads, head_dim, kvlen)

    return pt.bmm(q, k)  # (batch * heads, qlen, klen)


# TODO: port NVIDIAs implementation to PT C++ custom op
@pt.jit.script
def pytorch_interleaved_matmul_encdec_valatt(kv: pt.Tensor,
                                             att: pt.Tensor,
                                             heads: int) -> pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_valatt with PyTorch.
    There is probably something to be gained by using views more
    efficiently but this is placeholder code anyway.

    :param kv: (kvlen, batch, hidden * 2)
    :param att: (batch * heads, qlen, kvlen)
    :param heads: number of attention heads
    :return: (qlen, batch, hidden)
    """
    kvlen, batch, hidden2 = kv.size()
    hidden = hidden2 // 2
    head_dim = hidden // heads

    tmp = kv.reshape(kvlen, batch, heads, 2, -1)
    v = tmp[:, :, :, 1, :]  # pick values
    v = v.permute(1, 2, 0, 3)  # bsz, heads, kvlen, head_dim
    v = v.reshape(-1, kvlen, head_dim)  # bsz * heads, kvlen, head_dim

    output = pt.bmm(att, v)  # bsz * heads, qlen, head_dim
    output = output.transpose(0, 1).contiguous().view(-1, batch, hidden)
    return output


class PyTorchDotAttentionCell(pt.nn.Module):

    def __init__(self, dropout: float = 0.0, heads: int = 1) -> None:
        super().__init__()
        self.dropout = pt.nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.heads = heads

    def forward(self,
                queries: pt.Tensor,
                key_values: pt.Tensor,
                mask: Optional[pt.Tensor] = None):
        """
        :param queries: Query tensor of shape (query_length, batch_size, hidden)
        :param key_values: Interleaved Key & value tensor of shape (key/value_length, batch_size, hidden * 2)
        :param mask: Optional boolean tensor for attention masking of shape (batch * heads, <qlen>, <kvlen>).
                     If this is cross-attention, <qlen> dimension can be 1 for broadcasting,
                     i.e. (batch * heads, 1, kvlen). For self-attention on the decoder side an autoregressive mask
                     should be provided of shape (1, len, len) or (len, len).
                     Value of this mask is True for positions that should be masked out (padding positions),
                     False for valid positions.
        """
        # (batch * heads, qlen, klen)
        logits = pytorch_interleaved_matmul_encdec_qk(queries, key_values, heads=self.heads)

        if mask is not None:
            logits = logits.masked_fill(mask, -C.LARGE_VALUES[logits.dtype])

        probs = F.softmax(logits, dim=-1)

        probs = self.dropout(probs) if self.dropout is not None else probs

        # key_values: (lk, n, dv * 2)
        # probs: (n*h, lq, lk)
        # result: (n, lq, dv)
        return pytorch_interleaved_matmul_encdec_valatt(key_values, probs, heads=self.heads)


def prepare_source_length_mask(lengths: pt.Tensor, heads: int, max_length: int) -> pt.Tensor:
    lengths = lengths.repeat_interleave(heads, dim=0)  # (batch_size * heads, seq_len)
    # (batch_size * heads, 1, max_len)
    return ~(pt.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]).view(-1, 1, max_length)


class PyTorchMultiHeadAttentionBase(pt.nn.Module):
    """
    Base class for Multi-head attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """
    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__()
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads

        self.dot_att = PyTorchDotAttentionCell(dropout=dropout, heads=heads)
        self.ff_out = pt.nn.Linear(in_features=depth_att, out_features=depth_out, bias=False)

    def _attend(self,
                queries: pt.Tensor,
                key_values: pt.Tensor,
                mask: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (queries_length, batch_size, depth).
        :param key_values: Keys/Values. Shape: (keys_values_length, batch_size, depth * 2).
        :param mask: Optional boolean attention mask. See DotAttentionCell for shape requirements.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """

        # (query_max_length, batch, depth)
        contexts = self.dot_att(queries=queries, key_values=key_values, mask=mask)

        # (query_max_length, batch, output_depth)
        contexts = self.ff_out(contexts)

        return contexts


class AutoregressiveLayer(pt.nn.Module):
    @property
    @abstractmethod
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: pt.Tensor, previous_states: pt.Tensor, *args) -> Tuple:
        """
        :param inputs: layer input
        :param previous_states: Previous states array or list of arrays
        :param args: layer-specific arguments and/or arguments to be ignored
        :return: layer output and new states
        """
        raise NotImplementedError


class PyTorchMultiHeadSelfAttention(PyTorchMultiHeadAttentionBase, AutoregressiveLayer):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    """

    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__(depth_att, heads, depth_out, dropout)

        self.depth_att = depth_att
        self.ff_in = pt.nn.Linear(in_features=depth_att, out_features=depth_att * 3, bias=False)
        self._drop_p = dropout
        # indicates whether self.ff_in.weight of shape (depth_att * 3, depth_key_value) is in interleaved format or not.
        # Interleaved format is used for inference, non-interleaved format is used for fused MHA in training.
        self.kv_interleaved = False

    def separate_kv(self):
        """ write kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention) """
        assert self.kv_interleaved
        with pt.no_grad():
            kv = self.ff_in.weight.data[self.depth:, :]
            k, v = kv.view(self.heads, 2 * self.depth_per_head, self.depth).split(
                self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self.depth)
            v = v.reshape(self.depth, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self):
        """ write kv input projection parameters in interleaved format (compatible with interleaved matmul) """
        assert not self.kv_interleaved
        with pt.no_grad():
            _, k, v = self.ff_in.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self.depth)
        self.kv_interleaved = True

    def train(self, mode: bool = True):
        """
        Overrides super().train() to ensure key-value parameters are stored in non-interleaved format during training
        and interleaved format during inference (mod.eval()).
        """
        if mode and self.kv_interleaved:
            # training operates with non-interleaved format
            self.separate_kv()
        elif not mode and not self.kv_interleaved:
            # eval/inference operates in interleaved format
            self.interleave_kv()
        return super().train(mode)

    def _load_from_state_dict(self, *args):
        self.kv_interleaved = True  # see SockeyeModel.save_parameters(): models store kv weight in interleaved format
        super()._load_from_state_dict(*args)

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return True

    def get_state_shape(self, batch_size: int) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        # shape: (length, batch, key_depth + value_depth)
        return 0, batch_size, self.depth_out * 2

    def forward(self, inputs: pt.Tensor, previous_states: Optional[pt.Tensor] = None, mask: Optional[pt.Tensor] = None, **args) -> Tuple[pt.Tensor, pt.Tensor]:  # type: ignore
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a tensor of shape (max_length, batch, output_depth).

        :param inputs: Input Data. Shape: (length, batch, input_depth).
        :param previous_states: Optional list with two tensors - previous input's keys and values.
                                Shape: 2 * (batch, max_length+1, depth_att).
        :param mask: Optional attention mask. See DotAttentionCell for shape information.
        :return: tensor of shape (max_length, batch, output_depth).
        """
        if self.training:  # use fused multi-head attention op during training
            assert not self.kv_interleaved
            contexts, _ = F.multi_head_attention_forward(query=inputs, key=inputs, value=inputs,
                                                         embed_dim_to_check=self.depth, num_heads=self.heads,
                                                         in_proj_weight=self.ff_in.weight,
                                                         in_proj_bias=None,
                                                         bias_k=None, bias_v=None, add_zero_attn=False,
                                                         dropout_p=self._drop_p,
                                                         out_proj_weight=self.ff_out.weight,
                                                         out_proj_bias=self.ff_out.bias,
                                                         training=self.training,
                                                         key_padding_mask=None,
                                                         need_weights=False,
                                                         attn_mask=mask,
                                                         use_separate_proj_weight=False,
                                                         q_proj_weight=None,
                                                         k_proj_weight=None,
                                                         v_proj_weight=None)
            return contexts, contexts  # dummy return
        else:  # during inference multi-head attention with interleaved key-value parameters is used
            proj = self.ff_in(inputs)
            queries, states = proj.split((self.depth_att, 2 * self.depth_att), dim=2)

            if previous_states is not None:
                states = pt.cat((previous_states, states), dim=0)

            return self._attend(queries=queries, key_values=states, mask=mask), states

    def weights_from_mxnet_block(self, block_mx: 'MultiHeadSelfAttention'):  # type: ignore
        was_train = self.training
        if was_train:
            self.eval()
        self.ff_in.weight.data[:] = pt.as_tensor(block_mx.ff_in.weight.data().asnumpy())
        self.ff_out.weight.data[:] = pt.as_tensor(block_mx.ff_out.weight.data().asnumpy())
        if was_train:
            self.train()


class PyTorchMultiHeadAttention(PyTorchMultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param depth_key_value: Dimension of input key and value vectors.
    :param dropout: Dropout probability on attention scores
    """

    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 depth_key_value: int = 512) -> None:
        super().__init__(depth_att, heads, depth_out, dropout)
        self.ff_q = pt.nn.Linear(in_features=depth_out, out_features=depth_att, bias=False)
        self.ff_kv = pt.nn.Linear(in_features=depth_key_value, out_features=depth_att * 2, bias=False)
        self._drop_p = dropout
        self._depth_key_value = depth_key_value
        # indicates whether self.ff_kv.weight of shape (depth_att * 2, depth_key_value) is in interleaved format or not.
        # Interleaved format is used for inference, non-interleaved format is used for fused MHA in training.
        self.kv_interleaved = False

    def separate_kv(self):
        """ Writes kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention). """
        assert self.kv_interleaved
        with pt.no_grad():
            k, v = self.ff_kv.weight.data.view(self.heads, 2 * self.depth_per_head, self._depth_key_value).split(
                self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self._depth_key_value)
            v = v.reshape(self.depth, self._depth_key_value)
        self.ff_kv.weight.data[:] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self):
        """ Writes kv input projection parameters in interleaved format (compatible with interleaved matmul). """
        assert not self.kv_interleaved
        with pt.no_grad():
            k, v = self.ff_kv.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_kv.weight.data[:] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self._depth_key_value)
        self.kv_interleaved = True

    def train(self, mode: bool = True):
        """
        Overrides super().train() to ensure key-value parameters are stored in non-interleaved format during training
        and interleaved format during inference (mod.eval()).
        """
        if mode and self.kv_interleaved:
            # training operates with non-interleaved format
            self.separate_kv()
        elif not mode and not self.kv_interleaved:
            # eval/inference operates in interleaved format
            self.interleave_kv()
        return super().train(mode)

    def _load_from_state_dict(self, *args):
        self.kv_interleaved = True  # see SockeyeModel.save_parameters(): models store kv weight in interleaved format
        super()._load_from_state_dict(*args)

    def forward(self,
                queries: pt.Tensor,
                key_values: pt.Tensor,
                mask: Optional[pt.Tensor] = None,
                projected_memory_kv: Optional[pt.Tensor] = None) -> pt.Tensor:  # mypy: ignore
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns an tensor of shape (max_length, batch, output_depth).

        :param queries: Query tensor. Shape: (queries_length, batch, input_depth).
        :param key_values: Memory data to attend to. Shape: (key_values_length, batch, input_depth).
        :param mask: Optional attention mask. See DotAttentionCell for shape information.
        :param projected_memory_kv: Optional previously projected memory keys and values.
        :return: tensor of shape (query_seq_len, batch, output_depth).
        """
        if self.training:  # use fused multi-head attention op during training
            assert not self.kv_interleaved
            assert projected_memory_kv is None, "caching not supported in training"
            contexts, _ = F.multi_head_attention_forward(query=queries, key=key_values, value=key_values,
                                                         embed_dim_to_check=self.depth, num_heads=self.heads,
                                                         in_proj_weight=None,
                                                         in_proj_bias=None,
                                                         bias_k=None, bias_v=None, add_zero_attn=False,
                                                         dropout_p=self._drop_p,
                                                         out_proj_weight=self.ff_out.weight,
                                                         out_proj_bias=self.ff_out.bias,
                                                         training=self.training,
                                                         key_padding_mask=None,
                                                         need_weights=False,
                                                         attn_mask=mask,
                                                         use_separate_proj_weight=True,
                                                         q_proj_weight=self.ff_q.weight,
                                                         k_proj_weight=self.ff_kv.weight[:self.depth, :],
                                                         v_proj_weight=self.ff_kv.weight[self.depth:, :])
            return contexts
        else:  # during inference multi-head attention with interleaved key-value parameters is used
            queries = self.ff_q(queries)
            key_values = projected_memory_kv if projected_memory_kv is not None else self.ff_kv(key_values)
            return self._attend(queries=queries, key_values=key_values, mask=mask)

    def weights_from_mxnet_block(self, block_mx: 'MultiHeadAttention'):  # type: ignore
        was_train = self.training
        if was_train:
            self.eval()
        self.ff_q.weight.data[:] = pt.as_tensor(block_mx.ff_q.weight.data().asnumpy())
        self.ff_kv.weight.data[:] = pt.as_tensor(block_mx.ff_kv.weight.data().asnumpy())
        self.ff_out.weight.data[:] = pt.as_tensor(block_mx.ff_out.weight.data().asnumpy())
        if was_train:
            self.train()


def interleave_kv(module: pt.nn.Module):
    """ Writes kv input projection parameters in interleaved format (compatible with interleaved matmul). """
    if isinstance(module, PyTorchMultiHeadAttention) or isinstance(module, PyTorchMultiHeadSelfAttention):
        if not module.kv_interleaved:
            module.interleave_kv()


def separate_kv(module: pt.nn.Module):
    """ Writes kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention). """
    if isinstance(module, PyTorchMultiHeadAttention) or isinstance(module, PyTorchMultiHeadSelfAttention):
        if module.kv_interleaved:
            module.separate_kv()


@pt.jit.script
def pytorch_get_positional_embeddings(length: int, depth: int) -> pt.Tensor:
    utils.check_condition(depth % 2 == 0, "Positional embeddings require an even embedding size it "
                                          "is however %d." % depth)
    # (1, depth)
    channels = pt.arange(depth // 2).unsqueeze(0)

    # (length, 1)
    positions = pt.arange(0, length).unsqueeze(1)
    scaled_positions = positions / pt.pow(10000, (2 * channels) / depth)
    # sinusoids:
    sin = pt.sin(scaled_positions)
    # cosines:
    cos = pt.cos(scaled_positions)
    # interleave: (length, num_embed)
    encodings = pt.hstack([sin, cos])
    return encodings


class PyTorchPositionalEmbeddings(pt.nn.Module):
    """
    Takes an encoded sequence and adds sinusoidal or learned positional embeddings as in Vaswani et al, 2017 to it.

    :param weight_type: type of embeddings, fixed or learned.
    :param num_embed: Embedding size.
    :param max_seq_len: Maximum sequence length.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    """

    def __init__(self,
                 weight_type: str,
                 num_embed: int,
                 max_seq_len: int,
                 scale_up_input: bool,
                 scale_down_positions: bool) -> None:
        utils.check_condition(num_embed % 2 == 0, "Positional embeddings require an even embedding size it "
                                                  "is however %d." % num_embed)
        super().__init__()
        self.weight_type = weight_type
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions

        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            weight = pytorch_get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
            if self.scale_down_positions:
                weight *= self.num_embed ** -0.5
            self.weight = pt.nn.Parameter(weight, requires_grad=False)
        elif self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight = pt.nn.Parameter(pt.Tensor(self.max_seq_len, self.num_embed))
        else:
            raise ValueError("weight_type '%s' is not supported!" % self.weight_type)

    def forward(self, data: pt.Tensor, steps: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        Applies positional embeddings to input data.

        :param data: Input data. Shape: (batch, length or 1, num_embed)
        :param steps: Optional steps input. If given, shape is (batch_size or 1, seq_len,)

        :return: Data with positional embeddings added
        """
        # (length, num_embed)
        if steps is None:
            # (batch, length, num_embed)
            pos_embedding = self.weight.unsqueeze(0)[:, :data.size()[1]]
        else:
            # (batch_size or 1, seq_len, num_embed)
            # NOTE: temporary fix until we decide how to handle output steps > max_supported_seq_len_target
            steps = pt.clip(steps, max=self.max_seq_len - 1)
            pos_embedding = F.embedding(steps, self.weight)

        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            pos_embedding = pos_embedding.detach()

        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        return data + pos_embedding

    def weights_from_mxnet_block(self, block_mx: 'PositionalEmbeddings'):  # type: ignore
        if self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight.data[:] = pt.as_tensor(block_mx.weight.data().asnumpy())


class PyTorchSSRU(AutoregressiveLayer):
    """
    Simpler Simple Recurrent Unit

    Kim et al, "From Research to Production and Back: Ludicrously Fast Neural Machine Translation" WNGT 2019

    Variant of an LSTM cell aimed at reducing computational dependency across time steps.
    Formally described as:

    (1) f[t] = sigmoid(W1[t] * x[t] + b[t])
    (2) c[t] = f[t] . c[t-1] + (1 - f[t]) . W2[t] * x[t]
    (3) h = ReLU(c[t])

    where:
        . represents elementwise multiplication;
        x[t] is the input at time step t;
        f[t] is the output of the forget gate at time step t;
        c[t] is the cell state at time step t;
        h is the output of the unit.

    :param model_size: number of hidden units
    :param inference_only: flag used to indicate execution at inference time
    """
    def __init__(self, model_size: int, inference_only: bool) -> None:
        super().__init__()
        self.model_size = model_size
        self.inference_only = inference_only

        self.cell_state_transform = self._inference_cell_state_transform \
                                    if inference_only else self._training_cell_state_transform

        self.forget_gate = pt.nn.Linear(in_features=model_size, out_features=model_size, bias=True)
        self.forget_gate_act = pt.nn.Sigmoid()

        self.linear = pt.nn.Linear(in_features=model_size, out_features=model_size, bias=False)

        self.relu = pt.nn.ReLU(inplace=False)  # inplace=False because we need to non-activated data as well

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return False

    def get_state_shape(self, batch_size: int) -> Tuple:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        return 1, batch_size, self.model_size

    @staticmethod
    @pt.jit.script_if_tracing
    def _training_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) -> Tuple[pt.Tensor,
                                                                                                    pt.Tensor]:
        """Update SSRU cell at training time"""
        steps = weighted_inputs.size()[0]
        cell_state = previous_cell_state.squeeze(0)
        states = []
        for t in range(steps):
            cell_state = forget_rates[t, :, :] * cell_state + weighted_inputs[t, :, :]
            states.append(cell_state)

        states = pt.stack(states, dim=0)  # type: ignore
        return states, cell_state.unsqueeze(0)  # type: ignore

    @staticmethod
    def _inference_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) -> Tuple[pt.Tensor,
                                                                                                     pt.Tensor]:
        """Update SSRU cell at inference time"""
        new_step_state = forget_rates * previous_cell_state + weighted_inputs  # (1, batch, input_depth)
        return new_step_state, new_step_state

    def forward(self, inputs: pt.Tensor, previous_states: pt.Tensor, **args) -> Tuple[pt.Tensor, pt.Tensor]:  # type: ignore
        """
        :param inputs: input data. Shape: (max_length, batch, input_depth).
        :param previous_states: previous cell states. Shape: (max_length, batch, input_depth)
        :return: cell output and new cell states.  Both with shape (max_length, batch, input_depth).
        """
        forget_rates = self.forget_gate_act(self.forget_gate(inputs))
        weighted_inputs = (1 - forget_rates) * self.linear(inputs)

        cell_state, last_step_state = self.cell_state_transform(previous_states, weighted_inputs, forget_rates)

        return self.relu(cell_state), last_step_state

    def weights_from_mxnet_block(self, block_mx: 'SSRU'):  # type: ignore
        self.forget_gate.weight.data[:] = pt.as_tensor(block_mx.forget_gate.weight.data().asnumpy())
        self.forget_gate.bias.data[:] = pt.as_tensor(block_mx.forget_gate.bias.data().asnumpy())
        self.linear.weight.data[:] = pt.as_tensor(block_mx.linear.weight.data().asnumpy())



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

from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch as pt

from sockeye import constants as C, utils
from sockeye.layers import AutoregressiveLayer, SSRU, PositionalEmbeddings


def pytorch_get_activation(act_type: str) -> pt.nn.Module:
    if act_type == C.SWISH1:
        return pt.nn.SiLU(inplace=True)
    if act_type == C.GELU:
        return pt.nn.GELU()
    return pt.nn.ReLU(inplace=True)


class PyTorchLHUC(pt.nn.Module):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    """

    def __init__(self,
                 num_hidden: int,
                 weight_init: Callable = partial(pt.nn.init.uniform_, a=0.1)) -> None:
        super().__init__()
        self._weight_init = weight_init
        self.weight = pt.nn.Parameter(pt.Tensor(num_hidden,))
        self.reset_parameters()

    def reset_parameters(self):
        self._weight_init(self.weight)

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
        weight = 2 * pt.sigmoid(self.weight)
        return weight * data

    def weights_from_mxnet_block(self, block_mx: 'LHUC'):
        self.weight[:] = pt.as_tensor(block_mx.weight.data().asnumpy())


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
        return pt.nn.functional.normalize(weight, p=2, dim=self._axis_arg, eps=0) * self.scale

# TODO: Port LengthRatio


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
    q *= head_dim ** -0.5

    kvlen, batch, hidden2 = kv.size()
    tmp = kv.reshape(kvlen, batch, heads, 2, head_dim)
    k = tmp[:, :, :, 0, :]  # pick keys
    k = k.permute(1, 2, 3, 0)  # (batch, heads, head_dim, kvlen)
    k = k.reshape(batch * heads, head_dim, kvlen)  # (batch * heads, head_dim, kvlen)

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
    batch_heads, qlen, kvlen_ = att.size()
    assert kvlen == kvlen_
    assert batch_heads // heads == batch
    hidden = hidden2 // 2
    head_dim = hidden // heads

    tmp = kv.reshape(kvlen, batch, heads, 2, -1)
    v = tmp[:, :, :, 1, :]  # pick values
    v = v.permute(1, 2, 0, 3)  # bsz, heads, kvlen, head_dim
    v = v.reshape(-1, kvlen, head_dim)  # bsz * heads, kvlen, head_dim

    output = pt.bmm(att, v)  # bsz * heads, qlen, head_dim
    output = output.transpose(0, 1).contiguous().view(qlen, batch, hidden)
    return output


class PyTorchDotAttentionCell(pt.nn.Module):

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = pt.nn.Dropout(p=dropout) if dropout > 0.0 else None
        self._dtype = C.DTYPE_FP32

    def forward(self, queries: pt.Tensor, key_values: pt.Tensor, heads: int,
                lengths: Optional[pt.Tensor] = None, bias: Optional[pt.Tensor] = None):
        # (batch * heads, qlen, klen)
        logits = pytorch_interleaved_matmul_encdec_qk(queries, key_values, heads=heads)

        if bias is not None:
            logits = logits + bias

        if lengths is not None:
            # lengths shape: (n*h,) (different than for mxnet where we cant use broadcasting on the qlen dim.
            # this is a temporary implementation that is likely slow. Once fully ported, we should prepare the mask below
            # once for the encoder/decoder (like the bias). Similarly, the bias code path above should probably use masked_fill eventually.
            klen = logits.size()[2]
            mask = pt.arange(klen)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(1)  # (n*h, 1, klen)
            logits = logits.masked_fill(~mask, -C.LARGE_VALUES[self._dtype])

        probs = pt.nn.functional.softmax(logits, dim=-1)

        probs = self.dropout(probs) if self.dropout is not None else probs

        # key_values: (lk, n, dv * 2)
        # probs: (n*h, lq, lk)
        # result: (n, lq, dv)
        return pytorch_interleaved_matmul_encdec_valatt(key_values, probs, heads=heads)


def pytorch_prepare_source_valid_lengths(valid_length: pt.Tensor, num_heads: int) -> pt.Tensor:
    """
    TODO: update documentation and change it to create a mask once porting to PT is complete
    Returns valid length tensor repeated by number of heads.
    """
    # (batch * heads, seq_len)
    return valid_length.repeat_interleave(num_heads, dim=0)


class PyTorchMultiHeadAttentionBase(pt.nn.Module):
    """
    Base class for Multi-head attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """
    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads

        self.dot_att = PyTorchDotAttentionCell(dropout=dropout)
        self.ff_out = pt.nn.Linear(in_features=depth_att, out_features=depth_out, bias=False)

    def _attend(self,
                queries: pt.Tensor,
                key_values: pt.Tensor,
                lengths: Optional[pt.Tensor] = None,
                bias: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (query_max_length, batch_size, depth).
        :param key_values: Keys. Shape: (memory_max_length, batch_size, depth * 2).
        :param lengths: Optional lengths of keys. Shape: (batch_size*heads,).
        :param bias: Optional 3d bias.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """

        # (query_max_length, batch, depth)
        contexts = self.dot_att(queries, key_values, self.heads, lengths, bias)

        # (query_max_length, batch, output_depth)
        contexts = self.ff_out(contexts)

        return contexts


class PyTorchMultiHeadSelfAttention(PyTorchMultiHeadAttentionBase, AutoregressiveLayer):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """

    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype)
        assert dtype == C.DTYPE_FP32, "only supports float32 for now"

        self.depth_att = depth_att
        self.ff_in = pt.nn.Linear(in_features=depth_att, out_features=depth_att * 3, bias=False)

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

    def forward(self,
                inputs: pt.Tensor,
                previous_states: Optional[pt.Tensor] = None,
                input_lengths: Optional[pt.Tensor] = None,
                bias: Optional[pt.Tensor] = None,
                *args) -> Tuple[pt.Tensor, pt.Tensor]:  # mypy: ignore
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a ndarray of shape (batch, max_length, output_depth).

        :param inputs: Input Data. Shape: (max_length, batch, input_depth).
        :param input_lengths: Optional lengths of inputs to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param previous_states: Optional list with two ndarrays - previous input's keys and values.
                                Shape: 2 * (batch, max_length+1, depth_att).
        :return: ndarray of shape (batch, max_length, output_depth).
        """
        proj = self.ff_in(inputs)
        queries, states = proj.split((self.depth_att, 2 * self.depth_att), dim=2)

        if previous_states is not None:
            states = pt.cat((previous_states, states), dim=0)

        return self._attend(queries, states, lengths=input_lengths, bias=bias), states

    def weights_from_mxnet_block(self, block_mx: 'MultiHeadSelfAttention'):
        self.ff_in.weight[:] = pt.as_tensor(block_mx.ff_in.weight.data().asnumpy())
        self.ff_out.weight[:] = pt.as_tensor(block_mx.ff_out.weight.data().asnumpy())


class PyTorchMultiHeadAttention(PyTorchMultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param depth_key_value: Dimension of input key and value vectors.
    :param dropout: Dropout probability on attention scores
    :param dtype: Data type for weights
    """

    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32,
                 depth_key_value: int = 512) -> None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype)

        self.ff_q = pt.nn.Linear(in_features=depth_out, out_features=depth_att, bias=False)
        # TODO: pytorch does not allow underspecified dimensions, so we use depth_out for in_features
        # TODO: here. This should be fine for standard transformer models.
        self.ff_kv = pt.nn.Linear(in_features=depth_out, out_features=depth_att * 2, bias=False)

    def forward(self, queries: pt.Tensor,
                memory: pt.Tensor,
                memory_lengths: Optional[pt.Tensor] = None,
                bias: Optional[pt.Tensor] = None,
                projected_memory_kv: Optional[pt.Tensor] = None) -> pt.Tensor:  # mypy: ignore
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns an ndarray of shape (max_length, batch, output_depth).

        :param queries: Query tensor. Shape: (query_max_length, batch, input_depth).
        :param memory: Memory data to attend to. Shape: (memory_max_length, batch, input_depth).
        :param memory_lengths: Optional lengths of memory to mask attention scores. Shape: (batch, 1).
        :param bias: Optional 3d bias tensor to mask attention scores.
        :param projected_memory_kv: Optional previously projected memory keys and values.
        :return: ndarray of shape (query_seq_len, batch, output_depth).
        """

        queries = self.ff_q(queries)
        kv = projected_memory_kv if projected_memory_kv is not None else self.ff_kv(memory)
        return self._attend(queries, kv, bias=bias, lengths=memory_lengths)

    def weights_from_mxnet_block(self, block_mx: 'MultiHeadAttention'):
        self.ff_q.weight[:] = pt.as_tensor(block_mx.ff_q.weight.data().asnumpy())
        self.ff_kv.weight[:] = pt.as_tensor(block_mx.ff_kv.weight.data().asnumpy())
        self.ff_out.weight[:] = pt.as_tensor(block_mx.ff_out.weight.data().asnumpy())


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
    :param weight_init: Optional initializer for learned embeddings.
    """

    def __init__(self,
                 weight_type: str,
                 num_embed: int,
                 max_seq_len: int,
                 scale_up_input: bool,
                 scale_down_positions: bool,
                 weight_init: Optional[Callable] = None) -> None:
        utils.check_condition(num_embed % 2 == 0, "Positional embeddings require an even embedding size it "
                                                  "is however %d." % num_embed)
        super().__init__()
        self.weight_type = weight_type
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions

        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            self.weight = pytorch_get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
            if self.scale_down_positions:
                self.weight *= self.num_embed ** -0.5
        elif self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight = pt.nn.Parameter(pt.Tensor(self.max_seq_len, self.num_embed))
        else:
            raise ValueError("weight_type '%s' is not supported!" % self.weight_type)
        # TODO consider weight initialization

    def forward(self, data, steps):  # pylint: disable=arguments-differ
        """
        Applies positional embeddings to input data.

        :param data: Input data. Shape: (batch, length or 1, num_embed)
        :param steps: Optional steps input. If given, shape is (batch_size or 1, seq_len,)

        :return: Data with positional embeddings added
        """
        # (length, num_embed)
        if steps is None:
            # (batch, length, num_embed)
            # TODO: using size here, thats dynamic
            pos_embedding = self.weight.unsqueeze(0)[:, :data.size()[1]]
        else:
            # (batch_size or 1, seq_len, num_embed)
            pos_embedding = pt.nn.functional.embedding(steps, self.weight)

        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            pos_embedding = pos_embedding.detach()

        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)

        return data + pos_embedding

    def weights_from_mxnet_block(self, block_mx: PositionalEmbeddings):
        if self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight[:] = pt.as_tensor(block_mx.weight.data().asnumpy())


class PyTorchSSRU(pt.nn.Module, AutoregressiveLayer):
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
    :param dtype: data type
    """
    def __init__(self,
                 model_size: int,
                 inference_only: bool,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        assert dtype == C.DTYPE_FP32, "other dtpyes not yet supported"
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
    def _training_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) -> Tuple[pt.Tensor,
                                                                                                    pt.Tensor]:
        """Update SSRU cell at training time"""
        steps = weighted_inputs.size()[0]
        cell_state = previous_cell_state.squeeze(0)
        states = []
        for t in range(steps):
            cell_state = forget_rates[t, :, :] * cell_state + weighted_inputs[t, :, :]
            states.append(cell_state)

        states = pt.stack(states, dim=0)
        return states, cell_state.unsqueeze(0)

    @staticmethod
    def _inference_cell_state_transform(previous_cell_state, weighted_inputs, forget_rates) -> Tuple[pt.Tensor,
                                                                                                     pt.Tensor]:
        """Update SSRU cell at inference time"""
        new_step_state = forget_rates * previous_cell_state + weighted_inputs  # (1, batch, input_depth)
        return new_step_state, new_step_state

    def forward(self, inputs: pt.Tensor, previous_states: pt.Tensor, *args) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        :param inputs: input data. Shape: (max_length, batch, input_depth).
        :param previous_states: previous cell states. Shape: (max_length, batch, input_depth)
        :return: cell output and new cell states.  Both with shape (max_length, batch, input_depth).
        """
        forget_rates = self.forget_gate_act(self.forget_gate(inputs))
        weighted_inputs = (1 - forget_rates) * self.linear(inputs)

        cell_state, last_step_state = self.cell_state_transform(previous_states, weighted_inputs, forget_rates)

        return self.relu(cell_state), last_step_state

    def weights_from_mxnet_block(self, block_mx: SSRU):
        self.forget_gate.weight[:] = pt.as_tensor(block_mx.forget_gate.weight.data().asnumpy())
        self.forget_gate.bias[:] = pt.as_tensor(block_mx.forget_gate.bias.data().asnumpy())
        self.linear.weight[:] = pt.as_tensor(block_mx.linear.weight.data().asnumpy())



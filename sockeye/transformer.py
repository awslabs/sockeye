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

from dataclasses import dataclass
from typing import Optional, Tuple

from mxnet import gluon, npx, np

from . import config
from . import constants as C
from . import layers
from . import quantization


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


class TransformerEncoderBlock(gluon.HybridBlock):
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 dtype: str) -> None:
        super().__init__()

        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          num_hidden=config.model_size)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            dtype=dtype)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           num_hidden=config.model_size)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              num_hidden=config.model_size)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         dtype=dtype,
                                         use_glu=config.use_glu)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               num_hidden=config.model_size)
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size)

    def forward(self, data: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        # self-attention
        data_self_att, _ = self.self_attention(self.pre_self_attention(data, None), None, lengths, None)
        data = self.post_self_attention(data_self_att, data)

        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)

        if self.lhuc is not None:
            data = self.lhuc(data)

        return data


class TransformerDecoderBlock(gluon.HybridBlock):
    """
    A transformer decoder block consists of an autoregressive attention block, encoder attention,
    and a feed-forward layer with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 inference_only: bool,
                 dtype: str) -> None:
        super().__init__()
        self.decoder_type = config.decoder_type

        if self.decoder_type == C.TRANSFORMER_TYPE:
            self.autoregr_layer = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                                heads=config.attention_heads,
                                                                depth_out=config.model_size,
                                                                dropout=config.dropout_attention,
                                                                dtype=dtype)
        elif self.decoder_type == C.SSRU_TRANSFORMER:
            self.autoregr_layer = layers.SSRU(model_size=config.model_size,
                                              inference_only=inference_only,
                                              dtype=dtype)
        else:
            raise ValueError("Invalid decoder type.")

        self.pre_autoregr_layer = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          num_hidden=config.model_size)

        self.post_autoregr_layer = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           num_hidden=config.model_size)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         dropout=config.dropout_prepost,
                                                         num_hidden=config.model_size)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       depth_key_value=config.depth_key_value,
                                                       dtype=dtype)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          num_hidden=config.model_size)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              num_hidden=config.model_size)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         dtype=dtype,
                                         use_glu=config.use_glu)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               num_hidden=config.model_size)

        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size)

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
                target: np.ndarray,
                target_bias: np.ndarray,
                source: np.ndarray,
                source_att_lengths: np.ndarray,
                autoregr_states: np.ndarray,
                enc_att_kv: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        target_autoregr, *new_autoregr_states = self.autoregr_layer(self.pre_autoregr_layer(target, None),
                                                                    autoregr_states,
                                                                    None,
                                                                    target_bias)

        target = self.post_autoregr_layer(target_autoregr, target)

        # encoder attention
        target_enc_att = self.enc_attention(self.pre_enc_attention(target, None),
                                            source,
                                            source_att_lengths,
                                            None,
                                            enc_att_kv)

        target = self.post_enc_attention(target_enc_att, target)

        # feed-forward
        target_ff = self.ff(self.pre_ff(target, None))
        target = self.post_ff(target_ff, target)

        if self.lhuc:
            target = self.lhuc(target)

        return target, new_autoregr_states


class TransformerProcessBlock(gluon.HybridBlock):
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
        self.dropout = dropout
        self.layer_norm = None
        if 'n' in sequence:
            self.layer_norm = gluon.nn.LayerNorm(axis=-1, in_channels=num_hidden, epsilon=1e-06)

    def forward(self, data: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
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
                    data = npx.dropout(data, p=self.dropout)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        return data


class TransformerFeedForward(gluon.HybridBlock):
    """
    Position-wise feed-forward block with activation.
    """

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 dtype: str,
                 use_glu: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_glu = use_glu
        self.ff1 = quantization.QuantizableDense(in_units=num_model, units=num_hidden, flatten=False, dtype=dtype)
        self.act = layers.get_activation(act_type)
        if use_glu:
            self.linear = quantization.QuantizableDense(in_units=num_model, units=num_hidden, flatten=False,
                                                        dtype=dtype)
        self.ff2 = quantization.QuantizableDense(in_units=num_hidden, units=num_model, flatten=False, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.ff1(x)
        h = self.act(h)
        if self.use_glu:
            h = h * self.linear(x)
        if self.dropout > 0.0:
            h = npx.dropout(h, p=self.dropout)
        y = self.ff2(h)
        return y


class AutoRegressiveBias(gluon.HybridBlock):
    def __init__(self) -> None:
        super().__init__()
        self._dtype = 'float32'

    def cast(self, dtype):
        self._dtype = dtype
        super().cast(dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Shape: (length, 1)
        length_array = npx.arange_like(x, axis=1)
        # matrix with lower triangle and main diagonal set to 0, upper triangle set to 1
        # Shape: (length, length)
        bias = npx.broadcast_greater(np.expand_dims(length_array, axis=0),
                                     np.expand_dims(length_array, axis=1))
        bias = bias * -C.LARGE_VALUES[self._dtype]
        bias = np.expand_dims(bias, axis=0)
        return npx.stop_gradient(bias)



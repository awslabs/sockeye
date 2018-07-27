# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Convolutional layers.
"""
from sockeye.config import Config
from . import utils
from . import constants as C
from . import layers
from typing import List, Tuple, Dict, Optional

import mxnet as mx
import math


class ConvolutionConfig(Config):
    """
    Configuration for a stack of convolutions with Gated Linear Units between layers, similar to Gehring et al. 2017.

    :param kernel_width: Kernel size for 1D convolution.
    :param num_hidden: Size of hidden representation after convolution.
    :param act_type: The type of activation to use.
    :param dropout: The dropout rate.
    :param dilate: Dilation rate.
    :param stride: Kernel window stride.
    :param weight_normalization: If True weight normalization is applied.
    """

    def __init__(self,
                 kernel_width: int,
                 num_hidden: int,
                 act_type: str = C.GLU,
                 dropout: float = 0.0,
                 dilate: int = 1,
                 stride: int = 1,
                 weight_normalization: bool = False) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_hidden = num_hidden
        utils.check_condition(act_type in C.CNN_ACTIVATION_TYPES, "Unknown activation %s." % act_type)
        self.act_type = act_type
        self.dropout = dropout
        self.weight_normalization = weight_normalization
        self.dilate = dilate
        self.stride = stride

    def effective_kernel_size(self):
        return self.kernel_width + (self.dilate - 1) * (self.kernel_width - 1)


class ConvolutionBlock:
    """
    A Convolution-GLU block consists of the 2 following sublayers:
    1. Dropout (optional)
    1. A Convolution (padded either both to the left and to the right or just to the left).
    2. An activation: Either a Gated Linear Unit or any other activation supported by MXNet.

    :param config: Configuration for Convolution block.
    :param pad_type: 'left' or 'centered'. 'left' only pads to the left (for decoding
           the target sequence). 'centered' pads on both sides (for encoding the source sequence).
    :param prefix: Name prefix for symbols of this block.
    """

    def __init__(self,
                 config: ConvolutionConfig,
                 pad_type: str,
                 prefix: str) -> None:
        self.prefix = prefix
        self.pad_type = pad_type
        self.config = config
        self.conv_weight = mx.sym.Variable("%sconv_weight" % prefix,
                                           shape=(
                                               self._pre_activation_num_hidden(),
                                               self.config.num_hidden,
                                               self.config.kernel_width)
                                           )
        if self.config.weight_normalization:
            self.weight_norm = layers.WeightNormalization(self.conv_weight,
                                                          self._pre_activation_num_hidden(),
                                                          ndim=3,
                                                          prefix="%sconv_" % prefix)
            self.conv_weight = self.weight_norm()
        else:
            self.weight_norm = None
        self.conv_bias = mx.sym.Variable("%sconv_bias" % prefix)

    def _pre_activation_num_hidden(self):
        if self.config.act_type == C.GLU:
            return 2 * self.config.num_hidden
        else:
            return self.config.num_hidden

    def __call__(self,
                 data: mx.sym.Symbol,
                 data_length: mx.sym.Symbol,
                 seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Run the convolutional block.

        :param data: Input data. Shape: (batch_size, seq_len, num_hidden).
        :param data_length: Vector with sequence lengths. Shape: (batch_size,).
        :param seq_len: Maximum sequence length.
        :return: Shape: (batch_size, seq_len, num_hidden).
        """
        # TODO: rethink when we really need masking...
        # Apply masking (so that we properly have zero padding for variable sequence length batches)
        # (seq_len, batch_size, num_hidden)
        data = mx.sym.SequenceMask(data=data, axis=1, sequence_length=data_length, use_sequence_length=True, value=0)

        # (batch_size,  num_hidden, seq_len)
        data = mx.sym.transpose(data, axes=(0, 2, 1))

        if self.pad_type == C.CNN_PAD_LEFT:
            # TODO (tdomhan): Implement striding with left-padding
            assert self.config.stride == 1, "Striding currently not supported with left padding."
            # we pad enough on both sides and later slice the extra padding from the right
            padding = self.config.effective_kernel_size() - 1
            # TODO: potentially remove zero-padding
        elif self.pad_type == C.CNN_PAD_CENTERED:
            # we pad enough so that the output sizeis equal to the input size and we don't need to slice
            utils.check_condition(self.config.effective_kernel_size() % 2 == 1,
                                  "Only odd kernel widths supported, but got %d" % self.config.effective_kernel_size())
            padding = int((self.config.effective_kernel_size() - 1) / 2)

            seq_len_padded = seq_len + padding * 2

            stride = self.config.stride
            if stride > 1:
                if seq_len_padded % stride != 0:
                    pad_after = stride - (seq_len_padded % stride)
                    # pad to the right so that stride int divides the time axis
                    # temporary 4d due to pad op constraint
                    data = mx.sym.expand_dims(data, axis=3)
                    data = mx.sym.pad(data=data,
                                      mode="constant",
                                      constant_value=0,
                                      pad_width=(0, 0,
                                                 0, 0,
                                                 0, pad_after,
                                                 0, 0))
                    data = mx.sym.reshape(data, shape=(0, 0, -1))
                    data_length = data_length + pad_after
                    seq_len = seq_len + pad_after

                # formula is: floor((x+2*p-k)/s)+1
                # with 2p = k - 1 we get: floor((x-1)/s)+1

                data_length = mx.sym.BlockGrad(mx.sym.floor((data_length - 1) / stride) + 1)
                seq_len = int(math.floor((seq_len - 1) / stride)) + 1
        else:
            raise ValueError("Unknown pad type %s" % self.pad_type)

        num_hidden = self._pre_activation_num_hidden()

        data_conv = mx.sym.Convolution(data=data,
                                       weight=self.conv_weight,
                                       bias=self.conv_bias,
                                       pad=(padding,),
                                       kernel=(self.config.kernel_width,),
                                       stride=self.config.stride,
                                       num_filter=num_hidden,
                                       dilate=(self.config.dilate,),
                                       layout="NCW")

        # (batch_size, 2 * num_hidden, seq_len)
        if self.pad_type == C.CNN_PAD_LEFT:
            data_conv = mx.sym.slice_axis(data=data_conv, axis=2, begin=0, end=seq_len)

        return self._post_convolution(data_conv), data_length, seq_len

    def step(self, data):
        """
        Run convolution over a single position. The data must be exactly as wide as the convolution filters.

        :param data: Shape: (batch_size, kernel_width, num_hidden).
        :return: Single result of a convolution. Shape: (batch_size, 1, num_hidden).
        """
        assert self.config.stride == 1, "Striding not supported on the target side"

        num_hidden = self._pre_activation_num_hidden()
        if self.config.dilate != 1:
            # (batch_size, num_hidden, kernel_width)
            data = mx.sym.swapaxes(data, dim1=1, dim2=2)

            # (batch_size, num_hidden, 1)
            data_conv = mx.sym.Convolution(data=data,
                                           weight=self.conv_weight,
                                           bias=self.conv_bias,
                                           kernel=(self.config.kernel_width,),
                                           num_filter=num_hidden,
                                           dilate=(self.config.dilate,),
                                           layout="NCW")

            return self._post_convolution(data_conv)
        else:
            # As we only run convolution over a single window that is exactly the size of the convolutional filter
            # we can use FullyConnected instead of Convolution for efficiency reasons. Additionally we do not need to
            # perform any masking.

            # (batch_size, num_hidden, kernel_width)
            data = mx.sym.swapaxes(data, dim1=1, dim2=2)
            # (batch_size, num_hidden * kernel_width)
            data = mx.sym.reshape(data, shape=(0, -3))
            # (preact_num_hidden, num_hidden * kernel_width)
            weight = mx.sym.reshape(self.conv_weight, shape=(0, -3))
            data_conv = mx.sym.FullyConnected(data=data,
                                              weight=weight,
                                              bias=self.conv_bias,
                                              num_hidden=num_hidden)
            # (batch_size, num_hidden, 1)
            data_conv = mx.sym.expand_dims(data_conv, axis=2)
            return self._post_convolution(data_conv)

    def _post_convolution(self, data_conv):
        # data_conv: (batch_size, pre_activation_num_hidden, seq_len)

        if self.config.act_type == C.GLU:
            # GLU
            # two times: (batch_size, num_hidden, seq_len)
            # pylint: disable=unbalanced-tuple-unpacking
            gate_a, gate_b = mx.sym.split(data_conv, num_outputs=2, axis=1)
            # (batch_size, num_hidden, seq_len)
            block_output = mx.sym.broadcast_mul(gate_a,
                                                mx.sym.Activation(data=gate_b, act_type="sigmoid"))
            # TODO: use the activation function from layers.py
        elif self.config.act_type == "none":
            block_output = data_conv
        else:
            # (batch_size, num_hidden, seq_len)
            block_output = mx.sym.Activation(data_conv, act_type=self.config.act_type)

        # (batch_size, seq_len, num_hidden)
        block_output = mx.sym.swapaxes(block_output, dim1=1, dim2=2)

        if self.config.dropout > 0.0:
            block_output = mx.sym.Dropout(block_output, p=self.config.dropout)

        return block_output

# TODO: encoder side left-padding
class ConvolutionalEncoderLayer(layers.EncoderLayer):
    def __init__(self, cnn_config: ConvolutionConfig, prefix: str):
        self.prefix = prefix
        self.cnn_config = cnn_config
        self.cnn_block = ConvolutionBlock(self.cnn_config, pad_type=C.CNN_PAD_CENTERED, prefix=prefix)

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        return self.cnn_block(source_encoded, source_encoded_lengths, source_encoded_max_length)

    def get_encoded_seq_len(self, seq_len: int):
        stride = self.cnn_config.stride
        if stride == 1:
            return seq_len
        else:
            padding = int((self.cnn_config.effective_kernel_size() - 1) / 2)
            seq_len_padded = seq_len + 2 * padding
            if seq_len_padded % stride != 0:
                pad_after = stride - (seq_len_padded % stride)
                seq_len = seq_len + pad_after
            return int(math.floor((seq_len - 1) / stride)) + 1

    def get_num_hidden(self) -> int:
        return self.cnn_config.num_hidden


class ConvolutionalDecoderLayer(layers.DecoderLayer):

    def __init__(self, input_num_hidden: int, cnn_config: ConvolutionConfig, prefix: str):
        self.input_num_hidden = input_num_hidden
        self.prefix = prefix
        self.cnn_config = cnn_config
        self.cnn_block = ConvolutionBlock(self.cnn_config, pad_type=C.CNN_PAD_LEFT, prefix=prefix)

    def decode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol, target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        # TODO: for the decoder we don't actually need the masking operation ...
        return self.cnn_block(target_encoded, target_encoded_lengths, target_encoded_max_length)[0]

    def decode_step(self, step: int, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int, target: mx.sym.Symbol, states: List[mx.sym.Symbol], att_dict) -> Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        # (batch_size, kernel_width - 1, num_hidden)
        prev_target = states[0]

        # target: (batch_size, num_hidden) -> (batch_size, 1, num_hidden)
        target = mx.sym.expand_dims(target, axis=1)

        # (batch_size, kernel_width, num_hidden)
        target = mx.sym.concat(prev_target, target, dim=1)

        # (batch_size, kernel_width, num_hidden) -> (batch_size, 1, num_hidden)
        out = self.cnn_block.step(target)

        # arg: (batch_size, kernel_width - 1, num_hidden).
        new_prev_target = mx.sym.slice_axis(data=target, axis=1, begin=1, end=self.cnn_config.effective_kernel_size())

        out = mx.sym.reshape(out, shape=(0, -1))

        return out, [new_prev_target]

    def num_states(self, step) -> int:
        return 1

    def state_variables(self, step: int):
        return [mx.sym.Variable(name="%s_conv_state" % self.prefix)]

    def state_shapes(self,
                     batch_size: int,
                     target_max_length: int,
                     source_encoded_max_length: int,
                     source_encoded_num_hidden: int):
        input_num_hidden = self.input_num_hidden
        kernel_width = self.cnn_config.effective_kernel_size()
        return [mx.io.DataDesc("%s_conv_state" % self.prefix,
                               shape=(batch_size, kernel_width - 1, input_num_hidden),
                               layout="NTW")]

    def init_states(self,
                    batch_size: int,
                    source_encoded: mx.sym.Symbol,
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int):
        input_num_hidden = self.input_num_hidden
        kernel_width = self.cnn_config.effective_kernel_size()
        return [mx.sym.zeros(shape=(batch_size, kernel_width - 1, input_num_hidden),
                             name="%s_conv_state" % self.prefix)]

    def get_num_hidden(self) -> int:
        return self.cnn_config.num_hidden


class ConvolutionalLayerConfig(layers.LayerConfig):

    def __init__(self,
                 num_hidden: int,
                 kernel_width: int = 3,
                 act_type: str = C.GLU,
                 dropout: float = 0.0,
                 dilate: int = 1,
                 stride: int = 1,
                 prefix: str=""):
        super().__init__()
        self.num_hidden = num_hidden
        self.kernel_width = kernel_width
        self.act_type = act_type
        self.dropout=dropout
        self.dilate = dilate
        self.stride = stride
        self.prefix = prefix

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> layers.EncoderLayer:
        cnn_config = ConvolutionConfig(kernel_width=self.kernel_width,
                                       num_hidden=self.num_hidden,
                                       dropout=self.dropout,
                                       dilate=self.dilate,
                                       stride=self.stride,
                                       act_type=self.act_type)
        return ConvolutionalEncoderLayer(cnn_config, prefix=prefix + "cnn_")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> layers.DecoderLayer:
        assert self.stride == 1, "Stride only supported on the encoder side."
        cnn_config = ConvolutionConfig(kernel_width=self.kernel_width,
                                       num_hidden=self.num_hidden,
                                       dropout=self.dropout,
                                       dilate=self.dilate,
                                       act_type=self.act_type)
        return ConvolutionalDecoderLayer(input_num_hidden=input_num_hidden, cnn_config=cnn_config,
                                         prefix=prefix + "cnn_")


class PoolingEncoderLayer(layers.EncoderLayer):
    """
    Pooling operating with a given stride and kernel. Sequences are extended to the right to cover sequences that
    are not a multiple of the stride.
    """

    def __init__(self, num_hidden, stride: int = 3, kernel: int = 3, pool_type: str = "avg"):
        self.pool_type = pool_type
        self.stride = stride
        self.kernel = kernel
        self.num_hidden = num_hidden

    def encode_sequence(self,
                        source_encoded: mx.sym.Symbol,
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        att_dict: Dict[str, mx.sym.Symbol]) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        # source_encoded: (batch_size, seq_len, num_hidden) -> (batch_size, num_hidden, seq_len)
        source_encoded = mx.sym.transpose(source_encoded, axes=(0, 2, 1))
        # (batch_size, num_hidden, seq_len, 1)
        source_encoded = mx.sym.expand_dims(source_encoded, axis=3)

        if self.kernel > source_encoded_max_length:
            pad_after = self.kernel - source_encoded_max_length
            source_encoded = mx.sym.pad(data=source_encoded,
                                        mode="constant",
                                        constant_value=0,
                                        pad_width=(0, 0,
                                                   0, 0,
                                                   0, pad_after,
                                                   0, 0))

        # (batch_size, num_hidden, seq_len/stride, 1)
        source_encoded = mx.sym.Pooling(data=source_encoded,
                                        pool_type=self.pool_type,
                                        pooling_convention='full',
                                        kernel=(self.kernel, 1),
                                        stride=(self.stride, 1))

        # (batch_size, num_hidden, seq_len/stride)
        source_encoded = mx.sym.reshape(source_encoded, shape=(0, 0, -1))
        source_encoded = mx.sym.transpose(source_encoded, axes=(0, 2, 1))

        source_encoded_lengths = mx.sym.BlockGrad(mx.sym.ceil((source_encoded_lengths - self.kernel) / self.stride) + 1)
        source_encoded_max_length = self.get_encoded_seq_len(source_encoded_max_length)
        return source_encoded, source_encoded_lengths, source_encoded_max_length

    def get_num_hidden(self):
        return self.num_hidden

    def get_encoded_seq_len(self, seq_len: int):
        # if the sequence is not as large as the kernel it is padded to the kernel size:
        seq_len = max(seq_len, self.kernel)
        return int(math.ceil((seq_len - self.kernel) / self.stride)) + 1


class PoolingLayerConfig(layers.LayerConfig):

    def __init__(self, stride: int = 3, kernel: Optional[int] = None, pool_type: str = "avg", prefix: str=""):
        super().__init__()
        self.stride = stride
        self.kernel = kernel if kernel is not None else stride
        self.pool_type = pool_type

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> layers.EncoderLayer:
        return PoolingEncoderLayer(num_hidden=input_num_hidden, stride=self.stride, kernel=self.kernel,
                                   pool_type=self.pool_type)

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> layers.DecoderLayer:
        raise NotImplementedError("Pooling only available on the encoder side.")


class QRNNBlock:
    """
    Implements Quasi-recurrent neural networks as described by Bradbury, James, et al. "Quasi-recurrent neural
    networks." arXiv preprint arXiv:1611.01576 (2016).

    QRNNs do not have any recurrency in calculating the gates but rather use convolutions. We implement the f-pooling
    variant so that the hidden states are calculated as
    h_t = f_t * h_{t-1} + (1 - f_t) * z_t,
    where f is the forget gate and z the input.
    """

    def __init__(self,
                 num_hidden: int,
                 input_num_hidden: int,
                 kernel_width: int,
                 act_type: str = "tanh",
                 prefix: str = ""):
        self.num_hidden = num_hidden
        self.kernel_width = kernel_width
        self.act_type = act_type
        num_out = 2
        self.conv_weight = mx.sym.Variable("%sconv_weight" % prefix, shape=(num_out * num_hidden,
                                                                            input_num_hidden,
                                                                            kernel_width))
        self.conv_bias = mx.sym.Variable("%sconv_bias" % prefix)

    def __call__(self,
                 data: mx.sym.Symbol,
                 data_length: mx.sym.Symbol,
                 seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        # (batch_size, seq_len, num_hidden) -> (batch_size, num_hidden, seq_len)
        data = mx.sym.transpose(data, axes=(0, 2, 1))

        padding = self.kernel_width - 1
        num_out = 2
        # (batch_size, 2 * num_hidden, left_pad + seq_len)
        data_conv = mx.sym.Convolution(data=data,
                                       weight=self.conv_weight,
                                       bias=self.conv_bias,
                                       pad=(padding,),
                                       kernel=(self.kernel_width,),
                                       num_filter=num_out * self.num_hidden,
                                       layout="NCW")

        # (batch_size, 2 * num_hidden, seq_len)
        data_conv = mx.sym.slice_axis(data=data_conv, axis=2, begin=0, end=seq_len)

        # (batch_size, seq_len, 2 * num_hidden)
        data_conv = mx.sym.transpose(data_conv, axes=(0, 2, 1))

        # 2 * (batch_size, seq_len, num_hidden)
        # pylint: disable=unbalanced-tuple-unpacking
        out, f_gates = mx.sym.split(data_conv, num_outputs=2, axis=2)
        out = mx.sym.Activation(data=out, act_type=self.act_type)
        f_gates = mx.sym.Activation(data=f_gates, act_type="sigmoid")

        gated_out = mx.sym.broadcast_mul(1 - f_gates, out)

        # accumulate hidden state
        hidden = mx.sym.zeros(shape=(0, self.num_hidden))
        hiddens = []

        for f_gate, out in zip(mx.sym.split(f_gates, num_outputs=seq_len, axis=1, squeeze_axis=True),
                               mx.sym.split(gated_out, num_outputs=seq_len, axis=1, squeeze_axis=True)):
            hidden = f_gate * hidden + out
            hiddens.append(mx.sym.expand_dims(hidden, axis=1))
        # (batch_size, seq_len, num_hidden)
        hiddens = mx.sym.concat(*hiddens, dim=1)
        return hiddens, data_length, seq_len

    def step(self, data: mx.sym.Symbol, prev_h: mx.sym.Symbol):
        """
        Run the qrnn cell over a single position. The data must be exactly as wide as the convolution filters.

        :param data: Shape: (batch_size, kernel_width, num_hidden).
        :param prev_h: The previous hidden state, Shape: (batch_size, num_hidden).
        :return: Single result of a convolution. Shape: (batch_size, 1, num_hidden).
        """
        # (batch_size, num_hidden, kernel_width)
        data = mx.sym.swapaxes(data, dim1=1, dim2=2)
        # (batch_size, num_hidden * kernel_width)
        data = mx.sym.reshape(data, shape=(0, -3))
        # (preact_num_hidden, num_hidden * kernel_width)
        weight = mx.sym.reshape(self.conv_weight, shape=(0, -3))
        num_out = 2

        # (batch_size, 2 * num_hidden)
        data_conv = mx.sym.FullyConnected(data=data,
                                          weight=weight,
                                          bias=self.conv_bias,
                                          num_hidden=num_out * self.num_hidden)
        # TODO: refactor post FC code into a function to be shared by decode_step and decode_sequence
        # pylint: disable=unbalanced-tuple-unpacking
        out, f_gates = mx.sym.split(data_conv, num_outputs=2, axis=1)

        out = mx.sym.Activation(data=out, act_type=self.act_type)
        f_gate = mx.sym.Activation(data=f_gates, act_type="sigmoid")

        curr_h = f_gate * prev_h + (1.0 - f_gate) * out
        return curr_h


class QRNNDecoderLayer(layers.DecoderLayer):
    """
    QRNN implemented with masked (left-padded) convolutions.
    """

    def __init__(self, num_hidden: int, input_num_hidden: int, kernel_width: int, act_type: str = "tanh",
                 prefix: str = ""):
        self.num_hidden = num_hidden
        self.input_num_hidden = input_num_hidden
        self.prefix = prefix
        self.qrnn = QRNNBlock(num_hidden=num_hidden, input_num_hidden=input_num_hidden,
                              kernel_width=kernel_width, act_type=act_type, prefix=prefix)

    def decode_sequence(self,
                        source_encoded: List[mx.sym.Symbol],
                        source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int,
                        target_encoded: mx.sym.Symbol,
                        target_encoded_lengths: mx.sym.Symbol,
                        target_encoded_max_length: int,
                        target_autoregressive_bias: mx.sym.Symbol) -> mx.sym.Symbol:
        return self.qrnn(target_encoded, target_encoded_lengths, target_encoded_max_length)[0]

    def decode_step(self, step: int, source_encoded: List[mx.sym.Symbol], source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int, target: mx.sym.Symbol, states: List[mx.sym.Symbol],
                    att_dict: dict) -> Tuple[mx.sym.Symbol, List[mx.sym.Symbol]]:
        # (batch_size, kernel_width - 1, num_hidden)
        prev_h = states[0]
        prev_target = states[1]

        # target: (batch_size, num_hidden) -> (batch_size, 1, num_hidden)
        target = mx.sym.expand_dims(target, axis=1)

        # (batch_size, kernel_width, num_hidden)
        target = mx.sym.concat(prev_target, target, dim=1)

        # (batch_size, kernel_width, num_hidden) -> (batch_size, num_hidden)
        out = self.qrnn.step(target, prev_h)

        # arg: (batch_size, kernel_width - 1, num_hidden).
        new_prev_target = mx.sym.slice_axis(data=target, axis=1, begin=1, end=self.qrnn.kernel_width)

        return out, [out, new_prev_target]

    def get_num_hidden(self) -> int:
        return self.num_hidden

    def reset(self):
        pass

    def num_states(self, step: int) -> int:
        return 2

    def state_variables(self, step: int) -> List[mx.sym.Symbol]:
        return [mx.sym.Variable(name="%s_qrnn_prev_h" % self.prefix),
                mx.sym.Variable(name="%s_qrnn_in_state" % self.prefix)]

    def init_states(self,
                    batch_size: int,
                    source_encoded: List[mx.sym.Symbol],
                    source_encoded_lengths: mx.sym.Symbol,
                    source_encoded_max_length: int) -> List[mx.sym.Symbol]:
        input_num_hidden = self.input_num_hidden
        kernel_width = self.qrnn.kernel_width
        return [mx.sym.zeros(shape=(batch_size, self.qrnn.num_hidden),
                             name="%s_qrnn_prev_h" % self.prefix),
                mx.sym.zeros(shape=(batch_size, kernel_width - 1, input_num_hidden),
                             name="%s_qrnn_in_state" % self.prefix)]

    def state_shapes(self,
                     batch_size: int,
                     target_max_length: int,
                     source_encoded_max_length: int,
                     source_encoded_num_hidden: int) -> List[mx.io.DataDesc]:
        input_num_hidden = self.input_num_hidden
        kernel_width = self.qrnn.kernel_width
        return [mx.io.DataDesc("%s_qrnn_prev_h" % self.prefix,
                               shape=(batch_size, self.qrnn.num_hidden),
                               layout="NTW"),
                mx.io.DataDesc("%s_qrnn_in_state" % self.prefix,
                               shape=(batch_size, kernel_width - 1, input_num_hidden),
                               layout="NTW")]


class QRNNEncoderLayer(layers.EncoderLayer):
    """
    QRNN encoder with f-pooling.
    """

    def __init__(self, num_hidden: int, input_num_hidden: int,
                 kernel_width: int, act_type: str = "tanh", prefix: str = ""):
        self.num_hidden = num_hidden
        self.qrnn = QRNNBlock(num_hidden=num_hidden, input_num_hidden=input_num_hidden,
                              kernel_width=kernel_width, act_type=act_type, prefix=prefix)

    def encode_sequence(self, source_encoded: mx.sym.Symbol, source_encoded_lengths: mx.sym.Symbol,
                        source_encoded_max_length: int, att_dict: Dict[str, mx.sym.Symbol]) -> Tuple[
        mx.sym.Symbol, mx.sym.Symbol, int]:
        return self.qrnn(source_encoded, source_encoded_lengths, source_encoded_max_length)

    def get_num_hidden(self) -> int:
        return self.num_hidden


class QRNNLayerConfig(layers.LayerConfig):

    def __init__(self, num_hidden: int, kernel_width: int = 3, act_type: str = "tanh"):
        super().__init__()
        self.num_hidden = num_hidden
        self.kernel_width = kernel_width
        self.act_type = act_type

    def create_encoder_layer(self, input_num_hidden: int, prefix: str) -> layers.EncoderLayer:
        return QRNNEncoderLayer(num_hidden=self.num_hidden, input_num_hidden=input_num_hidden,
                                kernel_width=self.kernel_width, act_type=self.act_type, prefix=prefix + "qrnn_")

    def create_decoder_layer(self, input_num_hidden: int, prefix: str) -> layers.DecoderLayer:
        return QRNNDecoderLayer(num_hidden=self.num_hidden, input_num_hidden=input_num_hidden,
                                kernel_width=self.kernel_width, act_type=self.act_type, prefix=prefix + "qrnn_")


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
Convolutional layers.
"""
from sockeye.config import Config
from . import utils
from . import constants as C
from . import layers

import mxnet as mx


class ConvolutionConfig(Config):
    """
    Configuration for a stack of convolutions with Gated Linear Units between layers, similar to Gehring et al. 2017.

    :param kernel_width: Kernel size for 1D convolution.
    :param num_hidden: Size of hidden representation after convolution.
    :param act_type: The type of activation to use.
    """

    def __init__(self,
                 kernel_width: int,
                 num_hidden: int,
                 act_type: str = C.GLU,
                 weight_normalization: bool = False) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_hidden = num_hidden
        utils.check_condition(act_type in C.CNN_ACTIVATION_TYPES, "Unknown activation %s." % act_type)
        self.act_type = act_type
        self.weight_normalization = weight_normalization


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
                 seq_len: int) -> mx.sym.Symbol:
        """
        Run the convolutional block.

        :param data: Input data. Shape: (batch_size, seq_len, num_hidden).
        :param data_length: Vector with sequence lengths. Shape: (batch_size,).
        :param seq_len: Maximum sequence length.
        :return: Shape: (batch_size, seq_len, num_hidden).
        """
        if self.pad_type == C.CNN_PAD_LEFT:
            # we pad enough on both sides and later slice the extra padding from the right
            padding = (self.config.kernel_width - 1,)
        elif self.pad_type == C.CNN_PAD_CENTERED:
            # we pad enough so that the output size is equal to the input size and we don't need to slice
            utils.check_condition(self.config.kernel_width % 2 == 1,
                                  "Only odd kernel widths supported, but got %d" % self.config.kernel_width)
            padding = (int((self.config.kernel_width - 1) / 2),)
        else:
            raise ValueError("Unknown pad type %s" % self.pad_type)

        num_hidden = self._pre_activation_num_hidden()

        # Apply masking (so that we properly have zero padding for variable sequence length batches)
        data = mx.sym.SequenceMask(data=data, axis=1, sequence_length=data_length, use_sequence_length=True, value=0)

        # (batch_size, num_hidden, seq_len)
        data = mx.sym.transpose(data, axes=(0, 2, 1))
        data_conv = mx.sym.Convolution(data=data,
                                       weight=self.conv_weight,
                                       bias=self.conv_bias,
                                       pad=padding,
                                       kernel=(self.config.kernel_width,),
                                       num_filter=num_hidden,
                                       layout="NCW")

        # (batch_size, 2 * num_hidden, seq_len)
        if self.pad_type == C.CNN_PAD_LEFT:
            data_conv = mx.sym.slice_axis(data=data_conv, axis=2, begin=0, end=seq_len)

        return self._post_convolution(data_conv)

    def step(self, data):
        """
        Run convolution over a single position. The data must be exactly as wide as the convolution filters.

        :param data: Shape: (batch_size, kernel_width, num_hidden).
        :return: Single result of a convolution. Shape: (batch_size, 1, num_hidden).
        """

        # As we only run convolution over a single window that is exactly the size of the convolutional filter
        # we can use FullyConnected instead of Convolution for efficiency reasons. Additionally we do not need to
        # perform any masking.

        num_hidden = self._pre_activation_num_hidden()

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
        # TODO: add layer norm (can we do this without reshaping?!)

        if self.config.act_type == C.GLU:
            # GLU
            # two times: (batch_size, num_hidden, seq_len)
            # pylint: disable=unbalanced-tuple-unpacking
            gate_a, gate_b = mx.sym.split(data_conv, num_outputs=2, axis=1)
            # (batch_size, num_hidden, seq_len)
            block_output = mx.sym.broadcast_mul(gate_a,
                                                mx.sym.Activation(data=gate_b, act_type="sigmoid"))
        else:
            # (batch_size, num_hidden, seq_len)
            block_output = mx.sym.Activation(data_conv, act_type=self.config.act_type)

        # (batch_size, seq_len, num_hidden)
        block_output = mx.sym.swapaxes(block_output, dim1=1, dim2=2)
        return block_output


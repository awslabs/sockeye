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

from sockeye.config import Config
from . import utils

import mxnet as mx


class ConvolutionGluConfig(Config):
    """
    Configuration for a stack of convolutions with Gated Linear Units between layers, similar to Gehring et al. 2017.
    """
    def __init__(self,
                 kernel_width: int,
                 num_hidden: int):
        super().__init__()
        self.kernel_width = kernel_width
        self.num_hidden = num_hidden


class ConvolutionGluBlock:
    """
    A convolution-GLU block consists of the 2 following sublayers:
    1. Convolution
    2. GLU

    :param pad_type: 'left' or 'centered'.
    """
    def __init__(self,
                 config: ConvolutionGluConfig,
                 pad_type: str,
                 prefix: str) -> None:
        self.prefix = prefix
        self.pad_type = pad_type
        self.config = config
        self.conv_weight = mx.sym.Variable("%sconv_weight" % prefix)
        self.conv_bias = mx.sym.Variable("%sconv_bias" % prefix)

    def __call__(self, data: mx.sym.Symbol,
                 data_length: mx.sym.Symbol,
                 seq_len: int) -> mx.sym.Symbol:
        """

        :param data: (batch_size, seq_len, num_hidden)
        :param data_length: (batch_size,)
        :param seq_len: int
        :return: (batch_size, seq_len, num_hidden)
        """
        #TODO: pad + slice differently in decoder vs encoder: pad left vs pad centered
        #TODO: masking
        #TODO: dropout?

        # (batch_size, num_hidden, seq_len)
        data = mx.sym.swapaxes(data, dim1=1, dim2=2)
        if self.pad_type == 'left':
            # we pad enough on both sides and later slice the extra padding from the right
            padding = (self.config.kernel_width - 1,)
        else:
            # we pad enough so that the output size is equal to the input size and we don't need to slice
            utils.check_condition(self.config.kernel_width % 2 == 1,
                                  "Only odd kernel width's supported, but got %d" % self.config.kernel_width)
            padding = ((self.config.kernel_width - 1)/2,)
        data_conv = mx.sym.Convolution(data=data,
                                       weight=self.conv_weight,
                                       bias=self.conv_bias,
                                       pad=padding,
                                       kernel=(self.config.kernel_width,),
                                       num_filter=2 * self.config.num_hidden)

        # (batch_size, 2 * num_hidden, seq_len)
        if self.pad_type == 'left':
            data_conv = mx.sym.slice_axis(data=data_conv, axis=2, begin=0, end=seq_len)

        # GLU
        # two times: (batch_size, num_hidden, seq_len)
        gate_a, gate_b = mx.sym.split(data_conv, num_outputs=2, axis=1)
        # (batch_size, num_hidden, seq_len)
        block_output = mx.sym.broadcast_mul(gate_a,
                                            mx.sym.Activation(data=gate_b, act_type="sigmoid"))
        # (batch_size, seq_len, num_hidden)
        block_output = mx.sym.swapaxes(block_output, dim1=1, dim2=2)
        return block_output



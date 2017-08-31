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

import mxnet as mx
from sockeye.config import Config
from typing import Tuple


class StackedConvolutionConfig(Config):
    """
    Configuration for a stack of convolutions with Gated Linear Units between layers, similar to Gehring et al. 2017.
    """
    def __init__(self,
                 kernel_width: int,
                 num_hidden: int,
                 num_layers: int):
        super().__init__()
        self.kernel_width = kernel_width
        self.num_hidden = num_hidden
        self.num_layers = num_layers


class ConvolutionGluBlock:
    """
    A convolution-GLU block consists of the 2 following sublayers:
    1. Convolution
    2. GLU
    """
    def __init__(self,
                 config: StackedConvolutionConfig,
                 prefix: str = C.CONVOLUTIONGLUBLOCK_PREFIX) -> None:
        
        self.config = config
        self.convolution_weight = mx.sym.Variable("%sconvolution_weight" % prefix)
        self.convolution_bias = mx.sym.Variable("%sconvolution_bias" % prefix)
        
        # @TODO: initialize in init or in call?
        
    def __call__(self, data: mx.sym.Symbol,) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        source_conv = my.sym.Convolution(data=data,
                                         weight=self.convolution_weight,
                                         bias=self.convolution_bias,
                                         pad=(self.config.convolution_config.kernel_width - 1),
                                         kernel=(self.config.convolution_config.kernel_widht,),
                                         num_filter=2 * self.config.convolution_config.num_hidden,)

        source_conv = mx.sym.slice_axis(data=source_conv, axis=2, begin=0, end=seq_length)
        source_gate_a, source_gate_b = my.sym.split(source_conv, num_outputs=2, axis=1)
        block_output = mx.sym.broadcast_mul(source_gate_a,
                                             mx.sym.Activaion(data=source_gate_b, act_type="sigmoid"))
        return block_output

-    
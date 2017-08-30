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


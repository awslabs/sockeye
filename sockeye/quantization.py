# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import math

import mxnet as mx
from mxnet.gluon.nn.activations import Activation

from . import constants as C

logger = logging.getLogger(__name__)


# Modified from the source to mxnet.gluon.nn.basic_layers.Dense which is:
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
class QuantizableDense(mx.gluon.HybridBlock):
    r"""Optionally Quantized fully-connected NN layer.

    `QuantDense` implements the operation:
    `output = activation(dot(input, weight) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `weight` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use `flatten` to convert it
    to rank 2 manually if necessary.

    Parameters
    ----------
    units : int
        Dimensionality of the output space.
    activation : str
        Activation function to use. See help on `Activation` layer.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    flatten: bool, default True
        Whether the input tensor should be flattened.
        If true, all but the first axis of input data are collapsed together.
        If false, all but the last axis of input data are kept the same, and the transformation
        applies on the last axis.
    dtype : str or np.dtype, default C.DTYPE_FP32
        Data type of output embeddings.
    weight_initializer : str or `Initializer`
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    in_units : int, optional
        Size of the input data. If not specified, initialization will be
        deferred to the first time `forward` is called and `in_units`
        will be inferred from the shape of input data.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.


    Inputs:
        - **data**: if `flatten` is True, `data` should be a tensor with shape
          `(batch_size, x1, x2, ..., xn)`, where x1 * x2 * ... * xn is equal to
          `in_units`. If `flatten` is False, `data` should have shape
          `(x1, x2, ..., xn, in_units)`.

    Outputs:
        - **out**: if `flatten` is True, `out` will be a tensor with shape
          `(batch_size, units)`. If `flatten` is False, `out` will have shape
          `(x1, x2, ..., xn, units)`.
    """
    def __init__(self, units, dtype: str, activation=None, use_bias=True, flatten=True,
                 weight_initializer=None, bias_initializer='zeros',
                 in_units=0, **kwargs):
        super(QuantizableDense, self).__init__(**kwargs)
        self._flatten = flatten
        self._dtype = dtype
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            if dtype == C.DTYPE_INT8:
                self.scaling = self.params.get('scaling', shape=(1,),
                                               #Initialize to an obviously wrong value so we can detect later
                                               init=mx.initializer.Constant(-1.0), dtype=C.DTYPE_FP32,
                                               allow_deferred_init=True)
                weight_initializer = 'zeros' # Most initializers don't work for int8, but this is for inference anyway.

            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)

            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer, dtype = C.DTYPE_FP32,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def cast(self, dtype):
        if self._dtype != C.DTYPE_INT8:
            self._dtype = dtype
            super(QuantizableDense, self).cast(dtype)
        else:
            #No casting an already quantized matrix.
            logger.warning("Ignoring casting on int8 matrix")

    def hybrid_forward(self, F, x, weight, scaling=None, bias=None):
        if self._dtype == C.DTYPE_INT8:
            if bias is not None:
                act = F.contrib.intgemm_fully_connected(x, weight, scaling, bias, no_bias=False, num_hidden=self._units,
                                                        flatten=self._flatten, name='fwd')
            else:
                act = F.contrib.intgemm_fully_connected(x, weight, scaling, no_bias=True, num_hidden=self._units,
                                                        flatten=self._flatten, name='fwd')
        else:
            #Newer MXNet allows a numpy array.
            #fc = F.npx.fully_connected if is_np_array() else F.FullyConnected
            act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                     flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act=self.act if self.act else 'linear',
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


def optimize_quantization_mse(tensor, rounds=10):
    """
    Minimize mean squared error of quantizing a tensor, returning the top value
    (i.e. the one that quantizes to 127).  Scaling = 127.0 / return value.

    This is a convex optimization problem.  EM works but makes slow steps.
    Instead of EM, use binary search in the direction minimization suggests.
    """
    best_mse = math.inf
    best_top = None
    maxabs = mx.nd.contrib.intgemm_maxabsolute(tensor)
    low = 0.0
    high = maxabs
    for _ in range(rounds):
        value = (low + high) / 2.0
        quant = mx.nd.contrib.intgemm_prepare_data(tensor, value)
        quant_float = mx.nd.cast(quant, dtype=C.DTYPE_FP32)
        mse = (quant_float * (value / 127.0) - tensor).norm().asscalar() / math.sqrt(float(tensor.size))
        if mse < best_mse:
            best_mse = mse
            best_top = value
        # This optimizes scaling subject to cluster assignment.
        # It can be used for EM but the step is really slow, so use it for direction.
        scale = mx.nd.sum(quant_float * quant_float) / mx.nd.sum(quant_float * tensor)
        top = 127.0 / scale.asscalar()
        if top < value:
            high = value
        else:
            low = value
    return best_top


def extract_quant_max(tensor_param: mx.gluon.parameter.Parameter, scaling_param: mx.gluon.parameter.Parameter) -> float:
    """
    Extract or tune the scaling factor for a parameter.
    """
    scaling = scaling_param.data()
    if scaling.asscalar() < 0:
        # Bogus auto initialized scaling factor.
        b_max = optimize_quantization_mse(tensor_param.data())
        scaling_param.set_data(b_max / 127.0)
    else:
        b_max = scaling * 127.0
    return b_max


def convert_weights_disk_format(params: mx.gluon.parameter.ParameterDict, dtype_store: str):
    """
    Convert weights from float32 MXNet format (B^T in float32) to disk format
    (B^T in int8 format).

    If dtype_store == 'int8' then compute scaling and quantize the model.
    If dtype_store == 'float32' then just annotate with scaling factors.
    :param params model parameters from model.collect_params() in a float32
       model.
    :param dtype_store data type to store on disk.
    """
    logger.info("Optimizing quantization scaling factors")
    for name, param in params.items():
        if name.endswith("_weight"):
            scaling_name = name[0:-6] + "scaling"
            if scaling_name in params:
                b_max = extract_quant_max(param, params[scaling_name])
                if dtype_store == C.DTYPE_INT8:
                    quantized = mx.nd.contrib.intgemm_prepare_data(param.data(), b_max)
                    param.set_data(quantized)
                    param.dtype = C.DTYPE_INT8


def convert_weights_cpu_dependent(params: mx.gluon.parameter.ParameterDict):
    """
    Convert weights from disk format to intgemm's CPU-dependent format for
    quantized matrix multiplication.

    :param params model parameters from model.collect_params() in a model that
        came from convert_weights_disk_format.
    """
    logger.info("Converting weights to CPU format.")
    for name, param in params.items():
        if name.endswith("_weight"):
            scaling_name = name[0:-6] + "scaling"
            if scaling_name in params:
                if param.dtype == C.DTYPE_INT8:
                    # Already fully quantized, just rearrange.
                    weight = mx.nd.contrib.intgemm_prepare_weight(param.data(), already_quantized = True)
                else:
                    # Use offline scaling factor if available.
                    b_max = extract_quant_max(param, params[scaling_name])
                    weight = mx.nd.contrib.intgemm_prepare_weight(param.data(), b_max)
                param.set_data(weight)
                param.dtype = C.DTYPE_INT8

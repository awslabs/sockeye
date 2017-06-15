# Frequently Asked Questions

### What does Sockeye mean?
Sockeye is a salmon found in the Northern Pacific Ocean.

### What does ``mxnet.base.MXNetError: [16:16:21] src/c_api/c_api_ndarray.cc:392: Operator _zeros cannot be run; requires at least one of FCompute<xpu>, NDArrayFunction, FCreateOperator be registered`` mean?
This means you are running a version of MXNet that does not include GPU instructions.
Try installing a version with GPU instructions such as ``mxnet-cu80mkl`` or ``mxnet-cu75mkl``.

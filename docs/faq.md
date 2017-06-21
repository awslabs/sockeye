# Frequently Asked Questions

### What does Sockeye mean?
Sockeye is a salmon found in the Northern Pacific Ocean.

### What does 'Operator _zeros cannot be run; requires at least one of FCompute<xpu>' mean?
If you get the following or a similar error message:
```
mxnet.base.MXNetError: [16:16:21] src/c_api/c_api_ndarray.cc:392: Operator _zeros cannot be run; requires at least one of FCompute<xpu>, NDArrayFunction, FCreateOperator be registered
```
this means you are running a version of MXNet that does not include GPU instructions.
Try installing a version with GPU instructions such as ``mxnet-cu80mkl`` or ``mxnet-cu75mkl``.

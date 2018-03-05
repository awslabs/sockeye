import mxnet as mx


y = mx.nd.array([[  0.,   1.,   2.,   3.,   4.],
     [  5.,   6.,   7.,   8.,   9.],
     [ 10.,  11.,  12.,  13.,  14.],
     [ 15.,  16.,  17.,  18.,  19.]]).astype('float16')

x = mx.nd.array([[ 1.,  3.],
     [ 0.,  2.]]).astype('int32')


sym = mx.sym.Embedding(data=mx.sym.Variable('x'), weight=mx.sym.Variable('y'), input_dim=4, output_dim=5)

result = sym.eval(x=x, y=y)[0]

print(result)
print(result.dtype)

#
# x = mx.sym.Variable(name='x')
# y = mx.sym.Variable(name='y')
#
# sym = mx.sym.dot(x, y)
#
# exec = sym.simple_bind(mx.gpu(0), x=mx.nd.array([1, 2]), y=mx.nd.array([1, 2]))
#
# exec.forward()
#
# print(exec.outputs[0].asnumpy())
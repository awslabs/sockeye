#!/usr/bin/env python3

import mxnet as mx
import sys

file = sys.argv[1]

d = mx.nd.load(file)
for key in d.keys():
    print(key, d[key].shape)

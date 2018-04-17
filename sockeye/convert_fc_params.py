# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


from . import constants as C
import mxnet as mx
import os
import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='model folder')
    args = parser.parse_args()

    param_file = os.path.join(args.model, C.PARAMS_BEST_NAME)

    comb_fc_pat = re.compile("arg:decoder_transformer_[0-9]+_att_enc_kv2h_weight")

    params = mx.nd.load(param_file)
    params_new = dict()
    for name, param in params.items():
        if re.match(comb_fc_pat, name):
            print("Found combined FC weight: %s. Will split..." % name)
            w_k, w_v = mx.nd.split(param, axis=0, num_outputs=2)
            params_new[name.replace('kv2h', "k2h")] = w_k
            params_new[name.replace('kv2h', "v2h")] = w_v
        else:
            params_new[name] = param
    os.rename(param_file, param_file + ".old")
    mx.nd.save(param_file, params_new)

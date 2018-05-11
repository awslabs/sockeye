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

import os
import urllib

import mxnet as mx

from sockeye import constants as C
from sockeye.image_captioning.encoder import ImageLoadedCnnEncoderConfig, \
    ImageLoadedCnnEncoder


def test_image_loaded_cnn_encoder():
    prefixes = ["-0000.params", "-symbol.json"]
    link_address = "http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1"
    model_name = "squeezenet_v1.1"
    model_path = "test/data"
    model_path = os.path.join(model_path, model_name)
    epoch = 0
    layer_name = "conv10"
    encoded_seq_len = 170
    num_embed = 1000
    no_global_descriptor = False
    preextracted_features = False
    source_image_size = (3, 224, 224)
    batch_size = 16
    # Get model
    for p in prefixes:
        if not os.path.exists(model_path + p):
            urllib.request.urlretrieve(link_address + p, model_path + p)
    # Setup encoder
    image_cnn_encoder_config = ImageLoadedCnnEncoderConfig(
                                    model_path=model_path,
                                    epoch=epoch,
                                    layer_name=layer_name,
                                    encoded_seq_len=encoded_seq_len,
                                    num_embed=num_embed,
                                    no_global_descriptor=no_global_descriptor,
                                    preextracted_features=preextracted_features)
    image_cnn_encoder = ImageLoadedCnnEncoder(image_cnn_encoder_config)
    # Prepare for inference
    data_nd = mx.nd.random_normal(shape=(batch_size,) + source_image_size)
    source = mx.sym.Variable(C.SOURCE_NAME)
    embedding, encoded_data_length, seq_len = image_cnn_encoder.encode(source,
                                                                       None,
                                                                       None)
    data_names = ['source']
    module = mx.mod.Module(symbol=embedding,
                           data_names=data_names,
                           label_names=None)
    module.bind(for_training=False,
                data_shapes=[(data_names[0], (batch_size,) + source_image_size)])
    # Pretrained net
    initializers = image_cnn_encoder.get_initializers()
    init = mx.initializer.Mixed(*zip(*initializers))
    module.init_params(init)
    provide_data = [
        mx.io.DataDesc(name=data_names[0],
                       shape=(batch_size,) + source_image_size,  # "NCHW"
                       layout=C.BATCH_MAJOR_IMAGE)
    ]
    batch = mx.io.DataBatch([data_nd], None,
                            pad=0, index=None,
                            provide_data=provide_data)
    # Inference & tests
    module.forward(batch)
    feats = module.get_outputs()[0].asnumpy()
    assert feats.shape == (batch_size, encoded_seq_len, num_embed)

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

import pytest
import random
import string

from test.common_image_captioning import run_extract_features_captioning, tmp_img_captioning_dataset


IMAGE_ENCODER_SETTINGS = [
    # RESNET 152
    ("--source-image-size 3 224 224 --batch-size 16 --image-encoder-layer stage4_unit3_conv3",
     "http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152", 0, "resnet-152"),
    # SQUEEZE-NET
    ("--source-image-size 3 224 224 --batch-size 16 --image-encoder-layer conv10",
     "http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1", 0, "squeezenet_v1.1"),
]


@pytest.mark.parametrize("extract_params, model_address, epoch, model_prefix",
                         IMAGE_ENCODER_SETTINGS)
def test_caption_random_features(extract_params: str, model_address: str, epoch: int, model_prefix: str):

    # generate random names
    source_list = [''.join(random.choice(string.ascii_uppercase) for _ in range(4)) for i in range(15)]
    prefix = "tmp_features"
    use_features = False
    with tmp_img_captioning_dataset(source_list,
                                    prefix,
                                    train_max_length=1,
                                    dev_max_length=1,
                                    test_max_length=1,
                                    use_features=use_features) as data:

        source_files = [data["source"], data["validation_source"], data["test_source"]]
        run_extract_features_captioning(extract_params=extract_params,
                             model_address=model_address,
                             epoch=epoch,
                             model_prefix=model_prefix,
                             source_files=source_files,
                             work_dir=data['work_dir'])

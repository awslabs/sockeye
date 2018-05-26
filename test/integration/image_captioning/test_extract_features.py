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

import random
import string

import pytest

from test.common_image_captioning import run_extract_features_captioning, \
    tmp_img_captioning_dataset

IMAGE_ENCODER_SETTINGS = [
    ("conv1"),
    ("conv2"),
]


@pytest.mark.parametrize("layer",
                         IMAGE_ENCODER_SETTINGS)
def test_caption_random_features(layer: str):
    source_image_size = (3, 20, 20)
    batch_size = 8
    extract_params = "--source-image-size {s1} {s2} {s3} --batch-size {batch_size} " \
                     "--image-encoder-layer {layer}".format(s1=source_image_size[0],
                                                            s2=source_image_size[1],
                                                            s3=source_image_size[2],
                                                            batch_size=batch_size,
                                                            layer=layer)

    # generate random names
    source_list = [
        ''.join(random.choice(string.ascii_uppercase) for _ in range(4)) for i
        in range(8)]
    prefix = "tmp_features"
    use_features = False
    with tmp_img_captioning_dataset(source_list,
                                    prefix,
                                    train_max_length=1,
                                    dev_max_length=1,
                                    test_max_length=1,
                                    use_features=use_features) as data:
        source_files = [data["source"], data["validation_source"],
                        data["test_source"]]
        run_extract_features_captioning(source_image_size=source_image_size,
                                        batch_size=batch_size,
                                        extract_params=extract_params,
                                        source_files=source_files,
                                        image_root=data['work_dir'])

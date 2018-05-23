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

import argparse
import pytest

import sockeye.image_captioning.arguments as arguments
import sockeye.constants as C

from test.unit.test_arguments import _test_args


@pytest.mark.parametrize("test_params, expected_params", [
    ('--image-root test_img_root --input input --output-root test_output_root --output output',
     dict(source_image_size=[3, 224, 224],
          image_root="test_img_root",
          input="input",
          output_root="test_output_root",
          output="output",
          batch_size=64,
          image_positional_embedding_type=C.NO_POSITIONAL_EMBEDDING,
          image_encoder_model_path="/path/to/mxnet/image/model/",
          image_encoder_model_epoch=0,
          image_encoder_layer="stage4_unit3_conv3",
          image_encoder_conv_map_size=49,
          image_encoder_num_hidden=512,
          no_image_encoder_global_descriptor=True,
          load_all_features_to_memory=False,
          device_ids=[-1],
          disable_device_locking=False,
          lock_dir='/tmp',
          use_cpu=False,
          extract_image_features=False
     ))
])
def test_image_extract_features_cli_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_image_extract_features_cli_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('--source-root test_src_root',
     dict(source_root="test_src_root"))
])
def test_image_source_root_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_image_source_root_args)


@pytest.mark.parametrize("test_params, expected_params", [
    ('--validation-source-root test_val_src_root --validation-source val_src --validation-target val_tgt',
     dict(validation_source_root="test_val_src_root",
          validation_source="val_src",
          validation_target="val_tgt",
          validation_source_factors=[]
     ))
])
def test_image_validation_data_params(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_image_validation_data_params)


@pytest.mark.parametrize("test_params, expected_params", [
    ('--load-all-features-to-memory',
     dict(load_all_features_to_memory=True, extract_image_features=False))
])
def test_preextracted_features_args(test_params, expected_params):
    _test_args(test_params, expected_params, arguments.add_preextracted_features_args)


def test_add_image_train_cli_args():
     # Just make sure that it does not fail. We covered above the main tests and
     # the rest are coveder in test/unit/test_arguments.py
     params = argparse.ArgumentParser()
     arguments.add_image_train_cli_args(params)


def test_add_image_caption_cli_args():
     # Just make sure that it does not fail. We covered above the main tests and
     # the rest are coveder in test/unit/test_arguments.py
     params = argparse.ArgumentParser()
     arguments.add_image_caption_cli_args(params)

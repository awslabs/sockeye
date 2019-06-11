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
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

import sockeye.image_captioning.utils as utils


def test_copy_mx_model_to():
    model_path = "test"
    model_epoch = 0

    with TemporaryDirectory() as work_dir:
        # Simulate model files
        model_path = os.path.join(work_dir, model_path)
        json_name = model_path + '-symbol.json'
        params_name = model_path + '-%04d.params' % model_epoch
        open(json_name, 'a').close()
        open(params_name, 'a').close()

        with TemporaryDirectory() as output_folder:
            target_path = utils.copy_mx_model_to(model_path, model_epoch, output_folder)
            assert os.path.exists(target_path + '-symbol.json')
            assert os.path.exists(target_path + '-%04d.params' % model_epoch)


def test_crop_resize_image():
    image_size = [224, 224]
    imarray = np.random.rand(100, 250, 3) * 255
    image = Image.fromarray(imarray.astype('uint8'))
    image_o = utils.crop_resize_image(image, image_size)
    image_o = np.asarray(image_o)

    np.testing.assert_equal(image_o.shape[:2], image_size)


def test_load_preprocess_images():
    image_size = [3, 224, 224]
    image_paths = ['a.jpg', 'b.jpg', 'c.jpg']
    # Generate a set of images
    with TemporaryDirectory() as work_dir:
        filenames = []
        for s in image_paths:
            filename = os.path.join(work_dir, s)
            imarray = np.random.rand(100, 100, 3) * 255
            im = Image.fromarray(imarray.astype('uint8'))
            im.save(filename)
            filenames.append(filename)

        images = utils.load_preprocess_images(filenames, image_size)
        assert len(images)==3
        for img in images:
            np.testing.assert_equal(img.shape, image_size)


def test_load_features():
    feature_size = [10, 2048]
    filenames = ['a.npy', 'b.npy', 'c.npy', 'd.npy']
    # Generate a set of images
    with TemporaryDirectory() as work_dir:
        paths = []
        for s in filenames:
            filename = os.path.join(work_dir, s)
            data = np.random.rand(*feature_size)
            np.save(filename, data)
            paths.append(filename)

        feats = utils.load_features(paths, feature_size)
        assert len(feats)==4
        for f in feats:
            np.testing.assert_equal(f.shape, feature_size)


def test_save_features():
    feature_size = [10, 2048]
    filenames = ['a', 'b', 'c']
    # Generate the list of ndarrays
    datas = []
    for i in range(len(filenames)):
        datas.append(np.random.rand(*feature_size))

    with TemporaryDirectory() as work_dir:
        paths = [os.path.join(work_dir, s) for s in filenames]
        fnames = utils.save_features(paths, datas)
        for i, f in enumerate(fnames):
            assert os.path.exists(f)>0
            data = utils.load_feature(f, feature_size)
            np.testing.assert_almost_equal(datas[i], data)

    # Tests with compression
    with TemporaryDirectory() as work_dir:
        paths = [os.path.join(work_dir, s) for s in filenames]
        fnames = utils.save_features(paths, datas, compressed=True)
        for i, f in enumerate(fnames):
            assert os.path.exists(f)>0
            data = utils.load_feature(f, feature_size)
            np.testing.assert_almost_equal(datas[i], data)

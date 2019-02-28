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

"""
A set of utility methods for images.
"""
import os
import logging
from shutil import copyfile
from typing import List, Optional

import numpy as np

from ..log import setup_main_logger

# Temporary logger, the real one (logging to a file probably, will be created
# in the main function)
logger = logging.getLogger(__name__)

try:  # Try to import pillow
    from PIL import Image  # pylint: disable=import-error
except ImportError as e:
    raise RuntimeError("Please install pillow.")


def copy_mx_model_to(model_path, model_epoch, output_folder):
    """
    Copy mxnet models to new path.

    :param model_path: Model path without -symbol.json and -%04d.params
    :param model_epoch: Epoch of the pretrained model
    :param output_folder: Output folder
    :return: New folder where the files are moved to
    """
    target_path = os.path.join(output_folder, os.path.basename(model_path))
    logger.info("Copying image model from {} to {}".format(model_path,
                                                           target_path))
    suffix = ['-symbol.json', '-%04d.params' % (model_epoch,)]
    for s in suffix:
        copyfile(model_path + s, target_path + s)
    return target_path


def crop_resize_image(image: np.ndarray, size) -> np.ndarray:
    """
    Resize the input image.

    :param image: Original image which is a  PIL object.
    :param size: Tuple of height and width to resize the image to.
    :return: Resized image which is a PIL object
    """
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize(size, Image.ANTIALIAS)
    return image


def load_preprocess_images(image_paths: List[str], image_size: tuple) -> List[np.ndarray]:
    """
    Load and pre-process the images specified with absolute paths.

    :param image_paths: List of images specified with paths.
    :param image_size: Tuple to resize the image to (Channels, Height, Width)
    :return: A list of loaded images (numpy arrays).
    """
    image_size = image_size[1:]  # we do not need the number of channels
    images = []
    for image_path in image_paths:
        images.append(load_preprocess_image(image_path, image_size))
    return images


def load_preprocess_image(image_path: str, image_size: tuple) -> np.ndarray:
    with Image.open(image_path) as image:
        image_o = preprocess_image(image, image_size)
    return image_o


def preprocess_image(image: Image, image_size: tuple) -> np.ndarray:
    # Resize to fixed input
    image_o = crop_resize_image(image, image_size)
    # convert to numpy
    image_o = np.asarray(image_o)
    # Gray-level to 3 channels
    if len(image_o.shape) == 2:
        image_o = np.tile(image_o[:, :, None], (1, 1, 3))
    # (height, width, channel) -> (channel, height, width)
    image_o = np.swapaxes(image_o, 0, 2)
    image_o = np.swapaxes(image_o, 1, 2)
    return image_o


def load_features(paths: List[str],
                  expected_shape: Optional[tuple] = None) -> List[np.ndarray]:
    """
    Load features specified with absolute paths.

    :param paths: List of files specified with paths.
    :param expected_shape: Optional expected shape.
    :return: A list of loaded images (numpy arrays).
    """
    data = []  # type: List[np.ndarray]
    for path in paths:
        data.append(load_feature(path, expected_shape))
    return data


def load_feature(path: str,
                 expected_shape: Optional[tuple] = None) -> np.ndarray:
    try:  # compressed
        data = np.load(path)['data']
    except IndexError:  # uncompressed
        data = np.load(path)
    if expected_shape is not None:
        np.testing.assert_array_equal(data.shape, expected_shape,
                                      err_msg="Loaded feature shape different than provided one. "
                                              "(current: {}, provided{})".format(data.shape,
                                                                                 expected_shape))
    return data


def save_features(paths: List[str], datas: List[np.ndarray],
                  compressed: bool = False) -> List:
    """
    Save features specified with absolute paths.

    :param paths: List of files specified with paths.
    :param datas: List of numpy ndarrays to save into the respective files
    :param compressed: Use numpy compression
    :return: A list of file names.
    """
    fnames = []  # type: List[str]
    for path, data in zip(paths, datas):
        fnames.append(save_feature(path, data, compressed))
    return fnames


def save_feature(path: str,
                 data: np.ndarray,
                 compressed: bool = False) -> str:
    if compressed:
        np.savez_compressed(path, data=data)
        path += ".npz"
    else:
        np.save(path, data)
        path += ".npy"
    return path


def zero_pad_features(features: List[np.ndarray],
                      target_shape: tuple) -> List[np.ndarray]:
    """
    Zero pad to numpy array.

    :param features: List of numpy arrays.
    :param target_shape: Target shape of each numpy array in the list feat. Note:
                   target_shape should be greater that the largest shapes in feat.
    :return: A list of padded numpy arrays.
    """
    pad_features = []
    for feature in features:
        feature_shape = feature.shape
        if len(feature_shape) < len(target_shape):  # add extra dimensions
            for i in range(len(target_shape) - len(feature_shape)):
                feature = np.expand_dims(feature, axis=len(feature.shape) + 1)
                feature_shape = feature.shape
        elif len(feature_shape) > len(target_shape):
            raise ValueError("Provided target shape must be bigger then the original "
                             "shape. (provided: {}, original {})".format(len(target_shape), len(feature_shape)))
        diff_shape = np.subtract(target_shape, feature_shape)  # pylint: disable=assignment-from-no-return
        if np.any(diff_shape < 0):
            raise ValueError("Provided target values must be bigger then the original "
                             "values for each dimension. (provided: {}, original {})".format(target_shape, feature_shape))
        # pad format: ((before_1, after_1), ... (before_N, after_N))
        diff_shape = [[0, d] for d in diff_shape]  # pylint: disable=not-an-iterable
        p = np.pad(feature, diff_shape, 'constant', constant_values=0)
        pad_features.append(p)
    return pad_features

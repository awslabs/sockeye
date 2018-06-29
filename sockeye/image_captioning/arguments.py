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
Defines commandline arguments for the main CLIs with reasonable defaults.
"""
from .. import constants as C
from ..arguments import regular_file, regular_folder, add_training_data_args, \
    add_validation_data_params, add_prepared_data_args, add_bucketing_args, \
    add_vocab_args, add_training_output_args,  add_monitoring_args, \
    add_device_args, int_greater_or_equal, add_model_parameters, \
    add_training_args, add_logging_args, add_max_output_cli_args, \
    add_translate_cli_args


def add_image_source_root_args(params, required=False):
    params.add_argument('--source-root', '-sr',
                        required=required,
                        type=regular_folder(),
                        help='Source root where the training images are located.')


def add_image_validation_data_params(params):
    add_validation_data_params(params)
    params.add_argument('--validation-source-root', '-vsr',
                        type=regular_folder(),
                        help='Source root where the validation images are located.')


def add_image_training_io_args(params):
    params = params.add_argument_group("Data & I/O")
    add_training_data_args(params, required=False)
    add_image_source_root_args(params, required=False)
    add_prepared_data_args(params)
    add_image_validation_data_params(params)
    add_bucketing_args(params)
    add_vocab_args(params)
    add_training_output_args(params)
    add_monitoring_args(params)


def add_image_extract_features_cli_args(params):
    params = params.add_argument_group("Feature extraction")
    add_image_model_parameters(params)
    add_image_size_args(params)
    add_device_args(params)
    params.add_argument('--image-root', '-ir',
                        required=True,
                        type=regular_folder(),
                        help='Source root where the training images are located.')
    params.add_argument('--input', '-i',
                        required=True,
                        type=regular_file(),
                        help='Input file containing the list of images (paths relative to image-root) '
                             'to extract the features for.')
    params.add_argument('--output-root', '-or',
                        required=False,
                        type=str,
                        help='Where the actual features are stored.')
    params.add_argument('--output', '-o',
                        required=False,
                        type=str,
                        help='Output file where the list of features is stored (paths relative to output-root).')
    params.add_argument('--batch-size', '-b',
                        type=int_greater_or_equal(1),
                        default=64,
                        help='Mini-batch size. Default: %(default)s.')


def add_image_size_args(params):
    params.add_argument('--source-image-size', '-sis',
                        nargs='+', type=int,
                        default=[3, 224, 224],
                        help='Source images are resized to this size. It must fit the input shape of the network. Default: %(default)s.')


def add_image_model_parameters(params):
    model_params = params.add_argument_group("ImageModelConfig")

    # Image encoder arguments (pre-trained network)
    model_params.add_argument('--image-positional-embedding-type',
                              choices=C.POSITIONAL_EMBEDDING_TYPES,
                              default=C.NO_POSITIONAL_EMBEDDING,
                              help='The type of positional embedding. Default: %(default)s.')
    model_params.add_argument('--image-encoder-model-path', type=str,
                              default="/path/to/mxnet/image/model/",
                              help="Path to the mxnet pre-trained model for image encoding. The model comes "
                                   "with two files: .json and .params. NOTE: use the prefix only, do not include "
                                   "the sufix -symbol.json or -0000.params.")
    model_params.add_argument('--image-encoder-model-epoch', type=int,
                              default=0,
                              help="Epoch of the model to load. Default: %(default)s.")
    model_params.add_argument('--image-encoder-layer', type=str,
                              default="stage4_unit3_conv3",
                              help="This string specifies the name of the layer from the image model used as "
                                   "representation. The possible names can be found in the model file .json. Default: %(default)s.")
    model_params.add_argument('--image-encoder-conv-map-size', type=int,
                              default=49,
                              help="Expected size of the feature map related to the layer specified in "
                                   "--image-encoder-layer. If the conv map has shape 2048*7*7, the value "
                                   "of this parameter will be 7*7, thus 49. Default: %(default)s.")
    model_params.add_argument('--image-encoder-num-hidden', type=int,
                              default=512,
                              help="Number of hidden units of the fully-connected layer that encode "
                                   "the original features. Suggested to be of dimension which is lower "
                                   "than the original dimension. Default: %(default)s.")
    model_params.add_argument('--no-image-encoder-global-descriptor',
                              action="store_false",
                              help="The image encodes can be augmented with a global descriptor, which is "
                                   "the spatial average of the conv map. This is encoded with fully-connected "
                                   "layer defined with --image-encoder-num-hidden. Use this option to disable it.")
    add_preextracted_features_args(model_params)


def add_preextracted_features_args(model_params):
    model_params.add_argument('--load-all-features-to-memory',
                              action="store_true",
                              help="If we preextracted features, the files are loaded in batch from disk. "
                                   "Enable this option to load all the features to memory in the beginning "
                                   "only once. This speeds up, as long as the features fit to memory.")
    model_params.add_argument('--extract-image-features',
                              action="store_true",
                              help="If True, it extracts features and caption directly from input images,"
                                   "otherwise it will expect pre-extracted features.")


def add_image_train_cli_args(params):
    add_image_training_io_args(params)
    add_model_parameters(params)
    add_image_model_parameters(params)
    add_training_args(params)
    add_device_args(params)
    add_logging_args(params)
    add_max_output_cli_args(params)


def add_image_caption_cli_args(params):
    add_translate_cli_args(params)
    add_image_source_root_args(params, required=False)
    add_max_output_cli_args(params)
    # Used only if images as input instead of features
    add_image_model_parameters(params)
    add_image_size_args(params)
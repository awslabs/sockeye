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
CLI for extracting image features.
"""
import argparse
import os
import pickle
from contextlib import ExitStack

import mxnet as mx
import numpy as np

from . import arguments
from . import encoder
from . import utils
from .. import constants as C
from ..log import setup_main_logger
from ..translate import _setup_context

# Temporary logger, the real one (logging to a file probably, will be created
# in the main function)
logger = setup_main_logger(__name__, file_logging=False, console=True)


def batching(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_pretrained_net(args: argparse.Namespace,
                       context: mx.Context) -> mx.mod.Module:
    # init encoder
    image_cnn_encoder_config = encoder.ImageLoadedCnnEncoderConfig(
        model_path=args.image_encoder_model_path,
        epoch=args.image_encoder_model_epoch,
        layer_name=args.image_encoder_layer,
        encoded_seq_len=0,
        num_embed=100,
        preextracted_features=False
    )  # this num does not matter here

    image_cnn_encoder = encoder.ImageLoadedCnnEncoder(image_cnn_encoder_config)
    symbol = image_cnn_encoder.sym  # this is the net before further encoding
    arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(
        source=(1,) + tuple(args.source_image_size))
    last_layer_shape = out_shapes[-1][1:]

    # Create module
    module = mx.mod.Module(symbol=symbol,
                           data_names=[C.SOURCE_NAME],
                           label_names=None,
                           context=context)
    module.bind(for_training=False, data_shapes=[(C.SOURCE_NAME,
                                                  (args.batch_size,) + tuple(
                                                      args.source_image_size))])

    # Init with pretrained net
    initializers = image_cnn_encoder.get_initializers()
    init = mx.initializer.Mixed(*zip(*initializers))
    module.init_params(init)

    return module, last_layer_shape


def extract_features_forward(im, module, image_root, output_root, batch_size,
                             source_image_size, context):
    batch = mx.nd.zeros((batch_size,) + \
                        tuple(source_image_size), context)
    # Reading
    out_names = []
    for i, v in enumerate(im):
        batch[i] = utils.load_preprocess_image(os.path.join(image_root, v),
                                               source_image_size[1:])
        out_names.append(os.path.join(output_root, v.replace("/", "_")))
    # Forward
    module.forward(mx.io.DataBatch([batch]))
    feats = module.get_outputs()[0].asnumpy()
    # Chunk last batch which might be smaller
    if len(im) < batch_size:
        feats = feats[:len(im)]
    return feats, out_names


def read_list_file(input):
    with open(input, "r") as fd:
        data_list = []
        for i in fd.readlines():
            data_list.append(i.split("\n")[0])
    return data_list


def main():
    params = argparse.ArgumentParser(description='CLI to extract features ' \
                                                 'from images.')
    arguments.add_image_extract_features_cli_args(params)
    args = params.parse_args()

    image_root = os.path.abspath(args.image_root)
    output_root = os.path.abspath(args.output_root)
    output_file = os.path.abspath(args.output)
    size_out_file = os.path.join(output_root, "image_feature_sizes.pkl")
    if os.path.exists(output_root):
        logger.info("Overwriting provided path {}.".format(output_root))
    else:
        os.makedirs(output_root)

    # read image list file
    image_list = read_list_file(args.input)

    # Get pretrained net module (already bind)
    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)
        module, _ = get_pretrained_net(args, context)

        # Extract features
        with open(output_file, "w") as fout:
            for i, im in enumerate(batching(image_list, args.batch_size)):
                logger.info("Processing batch {}/{}".format(i + 1,
                                                            int(np.ceil(len(
                                                                image_list) / args.batch_size))))
                # TODO: enable caching to reuse features and resume computation
                feats, out_names = extract_features_forward(im, module,
                                                            image_root,
                                                            output_root,
                                                            args.batch_size,
                                                            args.source_image_size,
                                                            context)
                # Save to disk
                out_file_names = utils.save_features(out_names, feats)
                # Write to output file
                out_file_names = map(lambda x: os.path.basename(x) + "\n",
                                     out_file_names)
                fout.writelines(out_file_names)

        # Save the image size and feature size
        with open(size_out_file, "wb") as fout:
            pickle.dump({"image_shape": tuple(args.source_image_size),
                         "features_shape": tuple(feats.shape[1:])}, fout)

        # Copy image model to output_folder
        image_encoder_model_path = utils.copy_mx_model_to(
            args.image_encoder_model_path,
            args.image_encoder_model_epoch,
            output_root
        )

        logger.info("Files saved in {}, {} and {}.".format(output_file,
                                                           size_out_file,
                                                           image_encoder_model_path))


if __name__ == "__main__":
    main()

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
Image captioning CLI.
"""
import argparse
import os
import tempfile
from contextlib import ExitStack

import mxnet as mx

from . import arguments as arguments_image
from . import inference as inference_image
from .train import read_feature_shape
from .. import arguments
from .. import constants as C
from .. import inference
from .. import output_handler
from ..image_captioning import utils
from ..image_captioning.extract_features import get_pretrained_net, \
    batching, read_list_file, extract_features_forward
from ..lexicon import TopKLexicon
from ..log import setup_main_logger
from ..translate import read_and_translate, _setup_context
from ..utils import check_condition
from ..utils import log_basic_info

logger = setup_main_logger(__name__, file_logging=False)


def get_pretrained_caption_net(args: argparse.Namespace,
                               context: mx.Context,
                               image_preextracted_features: bool) -> inference_image.ImageCaptioner:
    models, source_vocabs, target_vocab = inference_image.load_models(
        context=context,
        max_input_len=args.max_input_len,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        model_folders=args.models,
        checkpoints=args.checkpoints,
        softmax_temperature=args.softmax_temperature,
        max_output_length_num_stds=args.max_output_length_num_stds,
        decoder_return_logit_inputs=args.restrict_lexicon is not None,
        cache_output_layer_w_b=args.restrict_lexicon is not None,
        source_image_size=tuple(args.feature_size),
        forced_max_output_len=args.max_output_length
    )
    restrict_lexicon = None  # type: TopKLexicon
    store_beam = args.output_type == C.OUTPUT_HANDLER_BEAM_STORE
    if args.restrict_lexicon:
        restrict_lexicon = TopKLexicon(source_vocabs, target_vocab)
        restrict_lexicon.load(args.restrict_lexicon)

    translator = inference_image.ImageCaptioner(context=context,
                                                ensemble_mode=args.ensemble_mode,
                                                bucket_source_width=0,
                                                length_penalty=inference.LengthPenalty(
                                                    args.length_penalty_alpha,
                                                    args.length_penalty_beta),
                                                beam_prune=args.beam_prune,
                                                beam_search_stop=args.beam_search_stop,
                                                models=models,
                                                source_vocabs=None,
                                                target_vocab=target_vocab,
                                                restrict_lexicon=restrict_lexicon,
                                                store_beam=store_beam,
                                                strip_unknown_words=args.strip_unknown_words,
                                                source_image_size=tuple(
                                                    args.feature_size),
                                                source_root=args.source_root,
                                                use_feature_loader=image_preextracted_features)
    return translator


def _extract_features(args, context):
    image_list = read_list_file(args.input)
    image_model, _ = get_pretrained_net(args, context)
    output_root = tempfile.mkdtemp()
    output_file = os.path.join(output_root, "input.features")
    with open(output_file, "w") as fout:
        for i, im in enumerate(batching(image_list, args.batch_size)):
            feats, out_names = extract_features_forward(im, image_model,
                                                        args.source_root,
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
    return output_root, output_file, tuple(feats.shape[1:])


def main():
    params = arguments.ConfigArgumentParser(description='Image Captioning CLI')
    arguments_image.add_image_caption_cli_args(params)
    args = params.parse_args()
    image_preextracted_features = not args.extract_image_features

    if args.output is not None:
        global logger
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models),
                        "must provide checkpoints for each model")

    log_basic_info(args)

    out_handler = output_handler.get_output_handler(args.output_type,
                                                    args.output,
                                                    args.sure_align_threshold)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

        if not image_preextracted_features:
            # Extract features and override input and source_root with tmp location of features
            args.source_root, args.input, args.feature_size = _extract_features(
                args, context)
            image_preextracted_features = True  # now we extracted features
        else:  # Read feature size from disk
            _, args.feature_size = read_feature_shape(args.source_root)

        translator = get_pretrained_caption_net(args, context,
                                                image_preextracted_features)

        read_and_translate(translator=translator,
                           output_handler=out_handler,
                           chunk_size=args.chunk_size,
                           input_file=args.input)


if __name__ == '__main__':
    main()

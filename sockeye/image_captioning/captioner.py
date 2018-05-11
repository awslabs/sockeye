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
from contextlib import ExitStack

import sockeye
import sockeye.constants as C
import sockeye.data_io
import sockeye.output_handler
from sockeye.lexicon import TopKLexicon
from sockeye.log import setup_main_logger
from sockeye.translate import read_and_translate, _setup_context
from sockeye.utils import check_condition
from sockeye.utils import log_basic_info
from . import arguments
from . import inference
from .train import read_feature_shape

logger = setup_main_logger(__name__, file_logging=False)


def main():
    params = argparse.ArgumentParser(description='Translate CLI')
    arguments.add_image_caption_cli_args(params)
    args = params.parse_args()
    # TODO: make code compatible with full net
    args.image_preextracted_features = True

    # Read feature size
    if args.image_preextracted_features:
        _, args.source_image_size = read_feature_shape(args.source_root)

    if args.output is not None:
        global logger
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")

    log_basic_info(args)

    output_handler = sockeye.output_handler.get_output_handler(args.output_type,
                                                               args.output,
                                                               args.sure_align_threshold)

    with ExitStack() as exit_stack:
        context = _setup_context(args, exit_stack)

        models, source_vocabs, target_vocab = inference.load_models(
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
            source_image_size=tuple(args.source_image_size),
            forced_max_output_len=args.max_output_length
        )
        restrict_lexicon = None # type: TopKLexicon
        store_beam = args.output_type == C.OUTPUT_HANDLER_BEAM_STORE
        if args.restrict_lexicon:
            restrict_lexicon = TopKLexicon(source_vocabs, target_vocab)
            restrict_lexicon.load(args.restrict_lexicon)

        translator = inference.ImageCaptioner(context=context,
                                              ensemble_mode=args.ensemble_mode,
                                              bucket_source_width=0,
                                              length_penalty=inference.LengthPenalty(args.length_penalty_alpha,
                                                                                     args.length_penalty_beta),
                                              beam_prune=args.beam_prune,
                                              beam_search_stop=args.beam_search_stop,
                                              models=models,
                                              source_vocabs=None,
                                              target_vocab=target_vocab,
                                              restrict_lexicon=restrict_lexicon,
                                              store_beam=store_beam,
                                              strip_unknown_words=args.strip_unknown_words,
                                              source_image_size=tuple(args.source_image_size),
                                              source_root=args.source_root,
                                              use_feature_loader=args.image_preextracted_features)

        read_and_translate(translator=translator,
                           output_handler=output_handler,
                           chunk_size=args.chunk_size,
                           inp=args.input)


if __name__ == '__main__':
    main()

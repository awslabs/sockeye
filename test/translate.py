import sockeye
import sockeye.utils
import sockeye.translate
import sockeye.inference

import mxnet as mx
import numpy as np

import os


def translate(encoder_fp16=False, decoder_fp16=False):
    context = mx.gpu(0)

    output_handler = sockeye.output_handler.get_output_handler('translation_with_score',
                                                               None,
                                                               0.9)

    models, vocab_source, vocab_target = sockeye.inference.load_models(
        context,
        256,
        20,
        1,
        [
            '<path>/model'
        ],
        None,
        None,
        2,
        decoder_return_logit_inputs=False,
        cache_output_layer_w_b=False,
        encoder_dtype=np.float16 if encoder_fp16 else np.float32,
        decoder_dtype=np.float16 if decoder_fp16 else np.float32)

    translator = sockeye.inference.Translator(context,
                                              'linear',
                                              10,
                                              sockeye.inference.LengthPenalty(1.0,
                                                                              0.0),
                                              models,
                                              vocab_source,
                                              vocab_target)
    sockeye.translate.translate(output_handler, ['Hallo Welt'], translator, 1)


if __name__ == '__main__':
    print(os.path.abspath(sockeye.__file__))
    print(os.path.abspath(mx.__file__))

    mx.profiler.profiler_set_config(mode='all', filename='profile_output.json')
    mx.profiler.profiler_set_state('run')

    translate(True, True)

    mx.profiler.profiler_set_state('stop')
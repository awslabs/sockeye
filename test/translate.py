import sockeye
import sockeye.utils
import sockeye.translate
import sockeye.inference

import mxnet as mx
import numpy as np

import os

def convert_params():
    arg_params, aux_params = sockeye.utils.load_params('/home/ec2-user/kellen/data/model/params.best')
    for k, v in arg_params.items():
        if v.dtype == np.float32:
            print('Converting {} to float16'.format(k))
            arg_params[k] = v.astype(dtype='float16')
    sockeye.utils.save_params(arg_params, '/home/ec2-user/kellen/data/model/params.best', aux_params)
    print('Done converting')

def list_params():
    arg_params, aux_params = sockeye.utils.load_params('/home/ec2-user/kellen/data/model/params.best')
    for k, v in arg_params.items():
        print('Param', k, v.dtype)

def translate():
    context = mx.gpu(0)

    output_handler = sockeye.output_handler.get_output_handler('translation_with_score',
                                                               None,
                                                               0.9)

    models, vocab_source, vocab_target = sockeye.inference.load_models(
        context,
        256,
        20,
        1,
        ['/home/ec2-user/kellen/data/model'],
        None,
        None,
        2,
        decoder_return_logit_inputs=False,
        cache_output_layer_w_b=False,
        use_fp16=True)

    translator = sockeye.inference.Translator(context,
                                              'linear',
                                              10,
                                              sockeye.inference.LengthPenalty(1.0,
                                                                              0.0),
                                              models,
                                              vocab_source,
                                              vocab_target,
                                              use_fp16=True)
    sockeye.translate.translate(output_handler, ['Hallo Welt'], translator, 1)
    # sockeye.translate.read_and_translate(translator, output_handler, None, None)


if __name__ == '__main__':
    #convert_params()
    print(os.path.abspath(sockeye.__file__))
    print(os.path.abspath(mx.__file__))

    mx.profiler.profiler_set_config(mode='all', filename='profile_output.json')
    mx.profiler.profiler_set_state('run')

    list_params()

    translate()

    mx.profiler.profiler_set_state('stop')
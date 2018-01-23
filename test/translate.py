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
    for k in sorted(arg_params):
        print('Param', k, arg_params[k].dtype)

def mse(A, B):
    return ((A - B) ** 2).mean(axis=None, dtype=np.float64)

def diff_arrays():
    fp32_outputs = os.listdir('fp32')

    for file_name in sorted(fp32_outputs):
        fp32_array = mx.nd.load('fp32/' + file_name)[0].asnumpy()
        fp16_array = mx.nd.load('fp16/' + file_name)[0].asnumpy().astype('float32')
        error = mse(fp16_array, fp32_array)
        #if error > 0.001:
        #if 'softmax' in file_name or 'logits' in file_name:
        print(file_name, error)


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
        ['/home/ec2-user/kellen/data/model'],
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
    sockeye.translate.translate(output_handler, ['Hello World'], translator, 1)
    # sockeye.translate.read_and_translate(translator, output_handler, None, None)


if __name__ == '__main__':
    #convert_params()
    print(os.path.abspath(sockeye.__file__))
    print(os.path.abspath(mx.__file__))

    mx.profiler.profiler_set_config(mode='all', filename='profile_output.json')
    mx.profiler.profiler_set_state('run')

    #list_params()

    #diff_arrays()
    translate(True, False)


    mx.profiler.profiler_set_state('stop')
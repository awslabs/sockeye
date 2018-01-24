import os
import sys
import sockeye.train

import mxnet as mx

BASE_PATH = '/home/ec2-user/kellen/sockeye/tutorials/seqcopy/'

def train():
    sys.argv.extend(['-s', BASE_PATH + 'train.source',
                     '-t', BASE_PATH + 'train.target',
                     '-vs', BASE_PATH + 'dev.source',
                     '-vt', BASE_PATH + 'dev.target',
                     '--num-embed', '32',
                     '--rnn-num-hidden', '64',
                     '--rnn-attention-type', 'dot',
                     #'--use-cpu',
                     '--metrics', 'perplexity', 'accuracy',
                     '--max-num-checkpoint-not-improved', '3',
                     '-o', 'seqcopy_model',
                     '--overwrite-output',
                     '--dtype', 'float16'])
    sockeye.train.main()


if __name__ == '__main__':
    print(os.path.abspath(sockeye.__file__))
    print(os.path.abspath(mx.__file__))

    train()

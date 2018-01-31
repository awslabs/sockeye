import os
import sys
import sockeye.train

import mxnet as mx

BASE_PATH = '<path>/tutorials/seqcopy/'


def train_seq():
    sys.argv.extend(['-s', BASE_PATH + 'train.source',
                     '-t', BASE_PATH + 'train.target',
                     '-vs', BASE_PATH + 'dev.source',
                     '-vt', BASE_PATH + 'dev.target',
                     '--num-embed', '32',
                     '--rnn-num-hidden', '64',
                     '--rnn-attention-type', 'dot',
                     '--max-updates', '100',
                     '--metrics', 'perplexity', 'accuracy',
                     '--max-num-checkpoint-not-improved', '3',
                     '--decode-and-evaluate', '500',
                     '-o', 'seqcopy_model',
                     '--overwrite-output',
                     '--dtype', 'float16'])
    sockeye.train.main()


def train_wmt():
    sys.argv.extend(['-s', '<path>/training/corpus.tc.BPE.de.200K',
                     '-t', '<path>/training/corpus.tc.BPE.en.200K',
                     '-vs', '<path>/training/newstest2016.tc.BPE.de',
                     '-vt', '<path>/training/newstest2016.tc.BPE.en',
                     '--num-embed', '256',
                     '--rnn-num-hidden', '512',
                     '--rnn-attention-type', 'dot',
                     '--max-seq-len', '60',
                     '--metrics', 'perplexity', 'accuracy',
                     '--decode-and-evaluate', '500',
                     '-o', 'wmt_model',
                     '--dtype', 'float16'])
    sockeye.train.main()


if __name__ == '__main__':
    print(os.path.abspath(sockeye.__file__))
    print(os.path.abspath(mx.__file__))

    #train_seq()
    train_wmt()

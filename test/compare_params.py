import os

import numpy as np
import mxnet as mx

import matplotlib.pyplot as plt
import re

def plot_diff(labels, data):
    mins = []
    maxes = []
    means = []
    std = []
    for l in data:
        mins.append(l.min())
        maxes.append(l.max())
        means.append(l.mean(dtype=np.float64))
        std.append(l.std(dtype=np.float64))

    mins = np.array(mins)
    maxes = np.array(maxes)
    means = np.array(means)
    std = np.array(std)

    plt.rc('ytick', labelsize=6)

    fig, ax = plt.subplots(figsize=(10, 100), dpi=200)

    ticks = np.arange(len(labels))
    ax.errorbar(y=ticks.copy(), x=means, xerr=std, fmt='ok', lw=3)
    ax.errorbar(y=ticks.copy(), x=means, xerr=[means - mins, maxes - means],
                fmt='.k', ecolor='gray', lw=1)
    # plt.xlim(-1, 8)
    ax.set_yticks(ticks.copy())
    ax.set_yticklabels(labels)
    ax.set_ylim(-1, len(labels))
    #ax.set_xscale('symlog', basex=20)
    ax.set_xlim(-0.1, 0.1)
    fig.tight_layout()
    plt.savefig('diff.png')

def mse(A, B):
    return ((A - B) ** 2).mean(axis=None, dtype=np.float64)


def exculde(name):
    if re.match(r'\d+-swapaxes\d+_output', name) \
        or re.match(r'\d+-sequencemask\d+_output', name):
        return True
    return False

def diff_arrays(dir32, dir16):
    fp32_outputs = os.listdir(dir32)

    data_points = []
    labels = []

    for file_name in sorted(fp32_outputs):
        layer_name = os.path.basename(file_name)
        if exculde(layer_name):
            continue
        fp32_array = mx.nd.load(os.path.join(dir32, file_name))[0].asnumpy()
        fp16_array = mx.nd.load(os.path.join(dir16, file_name))[0].asnumpy().astype('float32')

        data_points.append((fp32_array-fp16_array).flatten())
        labels.append(layer_name)

        error = mse(fp16_array, fp32_array)
        #if error > 0.001:
        #if 'softmax' in file_name or 'logits' in file_name:
        print(file_name, error)

    plot_diff(labels, data_points)

if __name__ == '__main__':
    diff_arrays('float32/decoder', 'float16/decoder')
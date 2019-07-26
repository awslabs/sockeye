# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import collections
import os

import matplotlib.pyplot as plt


PARSE_ENTRY = collections.defaultdict(lambda: str)
PARSE_ENTRY.update({
    'bleu-val': float,
    'chrf-val': float,
    'epoch': int,
    'learning-rate': float,
    'perplexity-train': float,
    'perplexity-val': float,
    'time-elapsed': lambda s: float(s) / (60 * 60),
})

FIND_BEST = collections.defaultdict(lambda: max)
FIND_BEST.update({
    'bleu-val': max,
    'chrf-val': max,
    'learning-rate': min,
    'perplexity-train': min,
    'perplexity-val': min,
})

AX_LABEL = {
    'bleu-val': 'Validation BLEU',
    'chrf-val': 'Validation chrF',
    'epoch': 'Epoch',
    'learning-rate': 'Learning Rate',
    'perplexity-train': 'Training Perplexity',
    'perplexity-val': 'Validation Perplexity',
    'time-elapsed': 'Training Time (Hours)',
}


def ax_label(s):
    if s in AX_LABEL:
        return AX_LABEL[s]
    return s


def read_metrics_file(fname: str):
    metrics = collections.defaultdict(list)
    for line in open(fname, encoding='utf-8'):
        entries = line.split()
        metrics['checkpoint'] = int(entries[0])
        for entry in entries[1:]:
            k, v = entry.split('=')
            v = PARSE_ENTRY[k](v)
            metrics[k].append(v)
    return metrics


def plot_metrics(args: argparse.Namespace):

    fig, ax = plt.subplots()
    overall_best_y = None

    for fname, label in zip(args.input,
                            args.legend if args.legend is not None
                            else (os.path.basename(fname) for fname in args.input)):
        metrics = read_metrics_file(fname)
        ax.plot(metrics[args.x][args.skip:], metrics[args.y][args.skip:], linewidth=1, alpha=0.75, label=label)
        ax.set(xlabel=ax_label(args.x), ylabel=ax_label(args.y), title=args.title)
        if args.best:
            best_y = FIND_BEST[args.y](metrics[args.y][args.skip:])
            if overall_best_y is None:
                overall_best_y = best_y
            else:
                overall_best_y = FIND_BEST[args.y](best_y, overall_best_y)
    if args.best:
        ax.axhline(y=overall_best_y, color='gray', linewidth=1, linestyle='--', zorder=999)

    ax.grid()
    ax.legend()

    fig.savefig(args.output)


def main():
    params = argparse.ArgumentParser(description='Plot data from `metrics` file written during training.')
    params.add_argument('-i', '--input', required=True, nargs='+', help='Input `metrics` file to plot.')
    params.add_argument('-o', '--output', required=True, help='Output file to write (ex: plot.pdf).')
    params.add_argument('-x', default='time-elapsed', help='X axis metric.')
    params.add_argument('-y', default='perplexity-val', help='Y axis metric.')
    params.add_argument('-l', '--legend', nargs='+', help='Labels in legend (one per input file).')
    params.add_argument('-t', '--title', help='Plot title.')
    params.add_argument('-b', '--best', action='store_true', help='Draw horizontal line at best Y value.')
    params.add_argument('-s', '--skip', type=int, default=0, help='Skip the first N points for better readability.')
    args = params.parse_args()
    plot_metrics(args)


if __name__ == '__main__':
    main()

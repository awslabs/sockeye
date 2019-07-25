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

from .utils import smart_open


PARSE_ENTRY = {'converged': lambda x: 1 if x == 'True' else 0,
               'diverged':lambda x: 1 if x == 'True' else 0,
               'epoch': int,
               'gradient-norm': lambda x: 0 if x == 'None' else float(x),
               'learning-rate': float,
               'perplexity-train': float,
               'perplexity-val': float,
               'time-elapsed': lambda x: float(x) / (60 * 60),
               'used-gpu-memory': int}

FIND_BEST = {'converged': max,
             'diverged':max,
             'epoch': max,
             'gradient-norm': max,
             'learning-rate': min,
             'perplexity-train': min,
             'perplexity-val': min,
             'time-elapsed': max,
             'used-gpu-memory': max}


def main():
    params = argparse.ArgumentParser(description='Plot data from `metrics` file written during training.')
    params.add_argument('-i', '--input', required=True, nargs='+', help='Input `metrics` file to plot.')
    params.add_argument('-o', '--output', required=True, help='Output file to write (ex: plot.pdf).')
    params.add_argument('-x', default='time-elapsed', help='X axis metric.')
    params.add_argument('-y', default='perplexity-val', help='Y axis metric.')
    params.add_argument('-b', '--best', action='store_true', help='Draw horizontal line at best Y value.')
    params.add_argument('-s', '--skip', type=int, default=0, help='Skip the first N points for better readability.')
    args = params.parse_args()
    plot_metrics(args)


def read_metrics_file(fname: str):
    metrics = collections.defaultdict(list)
    for line in smart_open(fname):
        entries = line.split()
        metrics['checkpoint'] = int(entries[0])
        for entry in entries[1:]:
            k, v = entry.split('=')
            v = PARSE_ENTRY[k](v)
            metrics[k].append(v)
    return metrics


def plot_metrics(args: argparse.Namespace):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    overall_best_y = None  # type: ignore

    for fname in args.input:
        metrics = read_metrics_file(fname)
        ax.plot(metrics[args.x][args.skip:], metrics[args.y][args.skip:], linewidth=1, alpha=0.75, label=os.path.basename(fname))
        ax.set(xlabel=args.x, ylabel=args.y)
        if args.best:
            best_y = FIND_BEST[args.y](metrics[args.y][args.skip:])
            if overall_best_y is None:
                overall_best_y = best_y
            else:
                overall_best_y = FIND_BEST[args.y](best_y, overall_best_y)
    if args.best:
        ax.axhline(y=overall_best_y, color='gray', linewidth=1, linestyle='--', zorder=9999)

    ax.grid()
    ax.legend()

    fig.savefig(args.output)


if __name__ == '__main__':
    main()

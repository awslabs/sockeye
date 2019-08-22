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

from argparse import ArgumentParser
from bisect import insort
from collections import defaultdict
from os import path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


PARSE_ENTRY = defaultdict(lambda: str)
PARSE_ENTRY.update({
    'bleu-val': float,
    'chrf-val': float,
    'epoch': int,
    'learning-rate': float,
    'perplexity-train': float,
    'perplexity-val': float,
    'time-elapsed': lambda s: float(s) / (60 * 60),
})

FIND_BEST = defaultdict(lambda: max)
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
    'checkpoint': 'Checkpoint',
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


def read_metrics_file(fname):
    metrics = defaultdict(list)
    for line in open(fname, encoding='utf-8'):
        entries = line.split()
        metrics['checkpoint'].append(int(entries[0]))
        for entry in entries[1:]:
            k, v = entry.split('=')
            v = PARSE_ENTRY[k](v)
            metrics[k].append(v)
    return metrics


def average_points(points, num_points, cmp):
    averaged = []
    best = []
    for point in points:
        insort(best, point)
        best = best[:num_points] if cmp is min else best[-num_points:]
        averaged.append(sum(best) / len(best))
    return averaged


def points_since_improvement(points, cmp):
    num_not_improved = []
    best = None
    since_improvement = 0
    for point in points:
        if best is None or (cmp is min and point < best) or (cmp is max and point > best):
            best = point
            since_improvement = 0
        num_not_improved.append(since_improvement)
        since_improvement += 1
    return num_not_improved


def window_improvement(points, num_points, cmp):
    window_improvement_at_point = []
    best_at_point = []
    for point in points:
        if not best_at_point:
            best_at_point.append(point)
        elif (cmp is min and point < best_at_point[-1]) or (cmp is max and point > best_at_point[-1]):
            best_at_point.append(point)
        else:
            best_at_point.append(best_at_point[-1])
        if len(best_at_point) > num_points:
            best_at_point = best_at_point[-num_points:]
        window_improvement_at_point.append(abs(best_at_point[-1] - best_at_point[0]))
    return window_improvement_at_point


def slope(points, num_points):
    # First point has no slope
    slope_at_point = [0]
    # Start computing slope with second point
    for i in range(1, len(points)):
        x, y = list(zip(*enumerate(points[max(i - num_points, 0):i + 1])))
        slope_at_point.append(np.polyfit(x, y, 1)[0])
    return slope_at_point


def plot_metrics(args):

    fig, ax = plt.subplots()
    overall_best_y = None

    if len(args.skip) == 1:
        args.skip *= len(args.input)

    # Paper scaling
    linewidth = 1.25 if args.paper else 1.0
    label_size = 12 if args.paper else None
    title_size = 16 if args.paper else None
    legend_size = 12 if args.paper else None
    tick_size = 12 if args.paper else None

    for fname, label, skip in zip(args.input,
                                  args.legend if args.legend is not None
                                  else (path.basename(fname) for fname in args.input),
                                  args.skip):
        # Read metrics file to dict
        metrics = read_metrics_file(fname)
        x_vals = metrics[args.x][skip:]
        y_vals = metrics[args.y][skip:]
        x_label=ax_label(args.x)
        y_label=ax_label(args.y)
        # Spread points that collapse into one significant digit (ex: epochs)
        for i_label, i_vals in zip([args.x, args.y], [x_vals, y_vals]):
            if i_label in ['epoch']:
                i_vals[:] = np.linspace(i_vals[0], i_vals[-1], len(i_vals))
        # Optionally average best points so far for each Y point
        if args.y_average is not None:
            y_vals = average_points(y_vals, args.y_average, cmp=FIND_BEST[args.y])
            y_label = '{} (Average of {} Points)'.format(y_label, args.y_average)
        # Optionally count points since last improvement for each Y point
        if args.y_since_best:
            y_vals = points_since_improvement(y_vals, cmp=FIND_BEST[args.y])
            y_label = '{} (Checkpoints Since Improvement)'.format(y_label)
        # Optionally compute the window improvement for each Y point
        if args.y_window_improvement is not None:
            y_vals = window_improvement(y_vals, args.y_window_improvement, cmp=FIND_BEST[args.y])
            # Don't plot points for which window improvement is unreliable
            # (fewer than number points used for window)
            x_vals = x_vals[args.y_window_improvement - 1:]
            y_vals = y_vals[args.y_window_improvement - 1:]
            y_label = '{} (Window Improvement over {} Points)'.format(y_label, args.y_window_improvement)
        # Optionally compute current slope for each Y point
        if args.y_slope is not None:
            y_vals = slope(y_vals, args.y_slope)
            # Don't plot points for which slope is unreliable (fewer than number
            # points used to compute slope)
            x_vals = x_vals[args.y_slope - 1:]
            y_vals = y_vals[args.y_slope - 1:]
            y_label = '{} (Slope of {} Points)'.format(y_label, args.y_slope)
        # Plot values for this metrics file
        ax.plot(x_vals, y_vals, linewidth=linewidth, alpha=0.75, label=label)
        plt.xlabel(x_label, fontsize=label_size)
        plt.ylabel(y_label, fontsize=label_size)
        plt.title(args.title, fontsize=title_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        # Optionally track best point so far
        if args.best:
            best_y = FIND_BEST[args.y](y_vals)
            if overall_best_y is None:
                overall_best_y = best_y
            else:
                overall_best_y = FIND_BEST[args.y](best_y, overall_best_y)
    # Optionally mark best Y point across metrics files
    if args.best:
        ax.axhline(y=overall_best_y, color='gray', linewidth=linewidth, linestyle='--', zorder=999)
    # Optionally draw user specified Y line
    if args.y_line is not None:
        ax.axhline(y=args.y_line, color='gray', linewidth=linewidth, linestyle='--', zorder=999)

    ax.grid()
    ax.legend(fontsize=legend_size)

    fig.savefig(args.output)


def main():
    params = ArgumentParser(description='Plot data from \'metrics\' files written during training.')
    params.add_argument('-i', '--input', required=True, nargs='+', help='One or more \'metrics\' files to plot.')
    params.add_argument('-o', '--output', required=True, help='Output file to write (ex: plot.pdf).')
    params.add_argument('-x', default='time-elapsed', help='X axis metric.')
    params.add_argument('-y', default='perplexity-train', help='Y axis metric.')
    params.add_argument('-ya', '--y-average', type=int, help='Average the N best points so far for each Y value.')
    params.add_argument('-ysb', '--y-since-best', action='store_true',
                        help='Use number of points since improvement for each Y value.')
    params.add_argument('-ywi', '--y-window-improvement', type=int,
                        help='Improvement in best over the last N points for each Y value.')
    params.add_argument('-ysl', '--y-slope', type=int, help='Compute current slope for each Y value.')
    params.add_argument('-yli', '--y-line', type=float, help='Draw a horizontal line at specified Y value.')
    params.add_argument('-l', '--legend', nargs='+', help='Labels in legend (one per input file).')
    params.add_argument('-t', '--title', help='Plot title.')
    params.add_argument('-b', '--best', action='store_true', help='Draw horizontal line at best Y value.')
    params.add_argument('-s', '--skip', type=int, nargs='+', default=(0,),
                        help='Skip the first N points for better readability.  Single value or value per input.')
    params.add_argument('-p', '--paper', action='store_true', help='Scale plot elements for inclusion in papers.')
    args = params.parse_args()
    plot_metrics(args)


if __name__ == '__main__':
    main()

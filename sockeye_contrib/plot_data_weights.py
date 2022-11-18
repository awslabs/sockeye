# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from collections import Counter

import matplotlib.pyplot as plt

from sockeye.utils import smart_open


LABEL_SIZE = 14
TITLE_SIZE = 18
TICK_SIZE = 14


def round_base(number: float, base: float = 0.05, precision: int = 8):
    """Round to the nearest multiple of `base` (0.05 by default)"""
    return round(base * round(number / base), precision)


def plot_weights(args):

    # Set up plot/figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Weight', fontsize=LABEL_SIZE)
    ax.set_ylabel('Count', fontsize=LABEL_SIZE)
    plt.title(args.title, fontsize=TITLE_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    # Read/plot weights
    counter = Counter()
    with smart_open(args.input) as inp:
        for line in inp:
            counter[round_base(float(line), base=args.base)] += 1
    ax.hist(x=counter.keys(), weights=counter.values(), bins=args.bins)  # type: ignore

    # Save figure
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)


def main():
    params = ArgumentParser(description='Create histogram of data instance weights.')
    params.add_argument('-i', '--input', required=True, help='Instance weights file with one weight per line.')
    params.add_argument('-o', '--output', required=True, help='Output file to write (e.g., plot.pdf).')
    params.add_argument('-t', '--title', type=str, default='Instance Weights', help='Title for plot.')
    params.add_argument('-b', '--bins', type=int, default=20, help='Number of bins for plotting weights.')
    params.add_argument('--base', type=float, default=0.001, help='Base for rounding when counting weights.')
    params.add_argument('--dpi', type=int, default=300, help='DPI for output file.')
    args = params.parse_args()
    plot_weights(args)


if __name__ == '__main__':
    main()

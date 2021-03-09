#!/usr/bin/env python

# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Dict, Iterator, List, Tuple


def read_benchmark_handler_output(stream: str) -> Iterator[Dict[str, str]]:
    for line in stream:
        fields = line.strip().split('\t')
        entry = dict(field.split('=', 1) for field in fields)
        yield entry


def compute_percentiles(lengths: List[int], length_percentile: int,
                        times: List[float], time_percentile: int) -> Tuple[int, float]:
    # Length percentile
    lp_i = min(int((length_percentile / 100) * len(lengths)), len(lengths) - 1)
    lp = sorted(lengths)[lp_i]

    # Time percentile (of length percentile)
    percentile_points = sorted(zip(lengths, times))[:lp_i + 1]
    percentile_times = [point[1] for point in percentile_points]
    tp_i = min(int((time_percentile / 100) * len(percentile_times)), len(percentile_times) - 1)
    tp = sorted(percentile_times)[tp_i]
    return lp, tp


def percentiles_from_benchmark_output(input_stream, length_percentile: int, time_percentile: int) -> Tuple[int, float]:
    input_lengths = []
    translation_times = []
    for entry in read_benchmark_handler_output(input_stream):
        input_lengths.append(int(entry['input_tokens']))
        translation_times.append(float(entry['translation_time']))
    return compute_percentiles(input_lengths, length_percentile, translation_times, time_percentile)


def main():
    parser = argparse.ArgumentParser(description='Report length and time percentiles for benchmark output')
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help=
        'Input file (result of sockeye.translate with \'benchmark\' output)')
    parser.add_argument(
        '--length-percentile',
        '-lp',
        type=int,
        default=99,
        help='Percentile to report for input length. Default: %(default)s')
    parser.add_argument(
        '--time-percentile',
        '-tp',
        type=int,
        default=99,
        help='Percentile to report for translation time. Default: %(default)s')
    args = parser.parse_args()

    with open(args.input) as inp:
        lp, tp = percentiles_from_benchmark_output(inp, args.length_percentile, args.time_percentile)
    print('P{}\t{:d}'.format(args.length_percentile, lp))
    print('P{}/{}\t{:0.3f}'.format(args.time_percentile, args.length_percentile, tp))


if __name__ == '__main__':
    main()

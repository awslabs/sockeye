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
import json
import os
import pickle
from typing import List, Tuple, Union
from unittest.mock import patch
import warnings

import numpy as np

try:
    # Suppress warnings when importing skopt
    with patch.object(warnings, 'warn', lambda *args, **kwargs: None):
        from skopt import Optimizer
except ImportError:
    raise RuntimeError('Please install Scikit-Optimize: pip install scikit-optimize')


class OptimizerHistory:
    '''
    Storage class for optimizer history: arguments being explored, their
    dimensions, and a list of explored points.  Each point is a flat list of
    index, whether this point has been reported to the optimizer, values, and
    score.
    '''

    def __init__(self):
        self.arguments = []  # type: List[str]
        self.dimensions = []  # type: List[Tuple[Union[int, float], Union[int, float]]]
        self.points = []  # type: List[List[Union[bool, int, float]]]

    def save(self, fname: str):
        with open(fname, 'wt') as fp:
            print(json.dumps(self.arguments), file=fp)
            print(json.dumps(self.dimensions), file=fp)
            for point in self.points:
                print(json.dumps(point), file=fp)

    def load(self, fname: str):
        with open(fname, 'rt') as fp:
            self.arguments = json.loads(fp.readline().strip())
            self.dimensions = json.loads(fp.readline().strip())
            for line in fp:
                self.points.append(json.loads(line.strip()))


def parse_hyper_parameter(hp_spec: str) -> Tuple[str, Tuple[Union[int, float], Union[int, float]]]:
    '''
    Convert spec string in format "argument:min_value:max_value" to tuple of
    (argument, dimension), where dimension is (min_value, max_value) as integers
    or floats.
    '''
    parse = lambda s: float(s) if '.' in s else int(s)
    try:
        argument, min_val, max_val = hp_spec.split(':')
        dimension = (parse(min_val), parse(max_val))
        assert '-' not in argument
        assert type(dimension[0]) is type(dimension[1])
        return argument, dimension
    except:
        raise TypeError('Hyper parameter specifications should be "arg:min:max" where `arg` is a sockeye.train '
                        'argument in Python form (ex: "initial_learning_rate"), `min` and `max` are both floats '
                        '(contain ".") or ints.')


def convert_np(obj: Union[np.int, np.float]) -> Union[int, float]:
    '''
    Convert from NumPy types to standard Python types.
    '''
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.float):
        return float(obj)
    raise TypeError('Required: np.int or np.float')


def format_args(history: OptimizerHistory, values: List[Union[int, float]]) -> str:
    '''
    Format arguments for sockeye.train.
    '''
    formatted = []
    for argument, dimension, value in zip(history.arguments, history.dimensions, values):
        sockeye_arg = '--' + argument.replace('_', '-')
        if dimension == (0, 1):
            # Special case for bool
            if value == 1:
                formatted.append(sockeye_arg)
        else:
            formatted.append('{} {}'.format(sockeye_arg, value))
    return ' '.join(formatted)


def main():
    params = ArgumentParser(description='Explore hyper parameters with Bayesian optimization.')
    params.add_argument('-hp', '--hyper-parameters', nargs='+', metavar='HP_SPEC',
                        help='Hyper parameters to optimize.  One or more specifications of "arg:min:max" where `arg` '
                             'is a sockeye.train argument in Python form (ex: "initial_learning_rate"), `min` and '
                             '`max` are both floats (contain ".") or ints.')
    params.add_argument('-si', '--state-in',
                        help='Current optimizer state dir to read.  Scores for history file should be added manually.')
    params.add_argument('-so', '--state-out', required=True, help='New optimizer state dir to write.')
    params.add_argument('-b', '--batch-size', type=int, default=1, help='Number of points to explore in each batch.')
    params.add_argument('-ip', '--initial-points', type=int, default=8,
                        help='Number of initial random points to explore.')
    params.add_argument('--seed', type=int, default=1, help='Random seed for optimizer run.')
    args = params.parse_args()

    if not any((args.hyper_parameters, args.state_in)) or all((args.hyper_parameters, args.state_in)):
        raise RuntimeError('Please specify either hyper parameters (--hyper-parameters) or current state (--state-in)')

    history = OptimizerHistory()

    # New optimizer run
    if args.hyper_parameters is not None:
        # New history from hyper parameter specifications
        history.arguments, history.dimensions = list(zip(*[parse_hyper_parameter(hp_spec) for hp_spec in args.hyper_parameters]))
        # Set random seed
        np.random.seed(args.seed)
        # Create new optimizer
        optimizer = Optimizer(history.dimensions, n_initial_points=args.initial_points)

    # Continue optimizer run
    if args.state_in:
        # Load existing history from file
        history.load(os.path.join(args.state_in, 'history'))
        # Load RNG state
        with open(os.path.join(args.state_in, 'rng.pkl'), 'rb') as fp:
            np.random.set_state(pickle.load(fp))
        # Load optimizer
        with open(os.path.join(args.state_in, 'opt.pkl'), 'rb') as fp:
            optimizer = pickle.load(fp)

    # Tell optimizer about any new points
    for point in history.points:
        if point[-1] is None:
            # Point still being evaluated
            continue
        if point[1]:
            # Point already reported
            continue
        # Newly evaluated point
        optimizer.tell(x=point[2:-1], y=point[-1])
        point[1] = True

    # Ask optimizer for next points to explore
    next_points = optimizer.ask(n_points=args.batch_size)

    # Print points in Sockeye argument format and add them to the history
    for i, values in enumerate(next_points, start=len(history.points)):
        print('{}: {}'.format(i, format_args(history, values)))
        history.points.append([i, False] + [convert_np(value) for value in values] + [None])

    # Save new state files
    if not os.path.exists(args.state_out):
        os.mkdir(args.state_out)
    history.save(os.path.join(args.state_out, 'history'))
    with open(os.path.join(args.state_out, 'rng.pkl'), 'wb') as fp:
        pickle.dump(np.random.get_state(), fp)
    with open(os.path.join(args.state_out, 'opt.pkl'), 'wb') as fp:
        pickle.dump(optimizer, fp)


if __name__ == '__main__':
    main()

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
import re
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


# Hyper parameter specification "HP:val1:val2:"
HP_REGEX = 'HP:([^:]*?:[^:]*?):'
# Choose next points based on expected improvement
OPTIMIZER_ACQUISITION_FUNCTION = 'EI'


class OptimizerState:
    '''
    Storage class for current optimizer state containing:
    - optimizer: instance of skopt.Optimizer.
    - template: format string for commands.
    - dimensions: list of hyper parameter ranges (dimensions).
    - points: list of points recommended by optimizer.
    - losses: loss for each point (entered by user).

    The NumPy random state is also saved/loaded.
    '''

    FNAME_OPTIMIZER = 'opt.pkl'
    FNAME_TEMPLATE = 'template'
    FNAME_DIMENSIONS = 'dimensions'
    FNAME_POINTS = 'points'
    FNAME_RNG = 'rng.pkl'

    def __init__(self):
        self.optimizer = None  # type: Optional[Optimizer]
        self.template = None  # type: Optional[str]
        self.dimensions = None  # type: Optional[List[Union[Tuple[int, int], Tuple[float, float], Tuple[None, str]]]]
        self.points = []  # type: List[List[Union[int, float]]]
        self.losses = []  # type: List[Union[str, float]]
        self.recorded = []  # type: List[bool]


    def format_command(self, point: List[Union[int, float]]) -> str:
        '''
        Create command string from point, filling in hyper parameter values and
        optional arguments.
        '''
        values = []
        for val, dimension in zip(point, self.dimensions):
            if dimension[0] is None:
                values.append(dimension[1] if val == 1 else '')
            else:
                values.append(val)
        return self.template.format(*values)


    def save(self, fname: str):
        '''
        Write to file: optimizer instance, command template, dimensions, points,
        and rng state.
        '''
        if not os.path.exists(fname):
            os.mkdir(fname)
        with open(os.path.join(fname, OptimizerState.FNAME_OPTIMIZER), 'wb') as fp:
            pickle.dump(self.optimizer, fp)
        with open(os.path.join(fname, OptimizerState.FNAME_TEMPLATE), 'wt') as out:
            print(self.template, file=out)
        with open(os.path.join(fname, OptimizerState.FNAME_DIMENSIONS), 'wt') as out:
            print(json.dumps(self.dimensions), file=out)
        with open(os.path.join(fname, OptimizerState.FNAME_POINTS), 'wt') as out:
            for i, (point, loss, rec) in enumerate(zip(self.points, self.losses, self.recorded)):
                # Index line
                print('{}:'.format(i), file=out)
                # Command line
                print(self.format_command(point), file=out)
                # HP values line (point X)
                print(json.dumps([convert_np(val) for val in point]), file=out)
                # Loss (point Y)
                print(loss, file=out)
                # Is recorded
                print(rec, file=out)
        with open(os.path.join(fname, OptimizerState.FNAME_RNG), 'wb') as fp:
            pickle.dump(np.random.get_state(), fp)

    def load(self, fname: str):
        '''
        Read from file: optimizer instance, command template, dimensions,
        points, and rng state.
        '''
        with open(os.path.join(fname, OptimizerState.FNAME_OPTIMIZER), 'rb') as fp:
            self.optimizer = pickle.load(fp)
        with open(os.path.join(fname, OptimizerState.FNAME_TEMPLATE), 'rt') as fin:
            self.template = fin.readline().strip()
        with open(os.path.join(fname, OptimizerState.FNAME_DIMENSIONS), 'rt') as fin:
            self.dimensions = json.loads(fin.readline().strip())
        with open(os.path.join(fname, OptimizerState.FNAME_POINTS), 'rt') as fin:
            while True:
                # Index line, generated as needed, not stored
                line = fin.readline()
                if line == '':
                    break
                # Command line, generated as needed, not stored
                fin.readline()
                # HP values line (point X)
                self.points.append(json.loads(fin.readline().strip()))
                # Loss (point Y)
                loss = fin.readline().strip()
                self.losses.append(float(loss) if loss else '')
                # Is recorded
                self.recorded.append(fin.readline() == 'True')
        with open(os.path.join(fname, OptimizerState.FNAME_RNG), 'rb') as fp:
            np.random.set_state(pickle.load(fp))


def convert_np(obj: Union[np.int, np.float]) -> Union[int, float]:
    '''
    Convert from NumPy types to standard Python types.
    '''
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.float):
        return float(obj)
    return obj


def parse_command(command: str) -> Tuple[str, List[Union[Tuple[int, int], Tuple[float, float], Tuple[None, str]]]]:
    '''
    Parse full or partial Sockeye command string that includes HP range
    placeholders.  Supported:
    - int: "HP:1:10:"
    - float: "HP:0.:1.:"
    - optional arg: "HP::--arg:"

    Returns a template (format string) and a matching list of dimensions.
    '''
    template = re.sub(HP_REGEX, '{}', command)
    dimensions = []
    matches = re.findall(HP_REGEX, command)
    for match in matches:
        dimensions.append(parse_hp(match))
    return template, dimensions


def parse_hp(hp_spec: str) -> Union[Tuple[int, int], Tuple[float, float], Tuple[None, str]]:
    '''
    Convert spec string to values using either of two formats:
    - "min_value:max_value" -> (min_value, max_value) for integers and floats.
    - ":argument" -> (None, argument) for optional arguments.
    '''
    parse = lambda s: float(s) if '.' in s or 'e' in s else int(s)
    try:
        val1, val2 = hp_spec.split(':')
        if not val1:
            # Optional argument
            return (None, val2)
        dimension = (parse(val1), parse(val2))
        assert type(dimension[0]) is type(dimension[1])
        return dimension
    except:
        raise TypeError('Hyper parameter specifications should be either "min_value:max_value" or ":argument"')


def main():
    params = ArgumentParser(description='Explore Sockeye hyper parameters with Bayesian optimization.')
    params.add_argument('-c', '--command',
                        help='Quoted full or partial Sockeye command with HP range placeholders.  Placeholders can be '
                             'int "HP:1:10:", float "HP:0.:1.:", or optional argument "HP::--arg:".  Full example: '
                             '"python -m sockeye.train -d data -vs src -vt trg -o out --optimizer-params '
                             'beta2:HP:0.98:0.999:,epsilon:HP:1e-9:1e-6: --update-interval HP:1:4: '
                             'HP::--disable-checkpoint-reload:"')
    params.add_argument('-i', '--initial-points', type=int, default=10,
                        help='Number of random initial points to explore before starting optimization.')
    params.add_argument('-si', '--state-in',
                        help='Current optimizer state dir to read.  Losses for points file should be added manually '
                             'before calling this script.')
    params.add_argument('-so', '--state-out', required=True, help='New optimizer state dir to write.')
    params.add_argument('-b', '--batch-size', type=int, default=1, help='Number of points to explore in each batch.')
    params.add_argument('--seed', type=int, default=1, help='Random seed for optimizer run.')
    args = params.parse_args()

    if not any((args.command, args.state_in)) or all((args.command, args.state_in)):
        raise RuntimeError('Please specify either a command (--command) or current state (--state-in)')

    np.random.seed(args.seed)

    state = OptimizerState()

    # New optimizer run
    if args.command is not None:
        # Identify hyper parameters and their ranges (dimensions)
        state.template, state.dimensions = parse_command(args.command)
        # Create new optimizer, convert optional args to 0/1 integer dimensions
        state.optimizer = Optimizer([(0, 1) if dim[0] is None else dim for dim in state.dimensions],
                                    n_initial_points=args.initial_points,
                                    acq_func=OPTIMIZER_ACQUISITION_FUNCTION)

    # Continue optimizer run
    if args.state_in:
        # State from previous run
        state.load(args.state_in)

    # Tell optimizer about any new points
    for i in range(len(state.points)):
        # Point already recorded
        if state.recorded[i]:
            continue
        # Point still being evaluated, not ready to be reported
        if state.losses[i] == '':
            continue
        # Newly evaluated point, ready to report
        state.optimizer.tell(x=state.points[i], y=state.losses[i])
        state.recorded[i] = True

    # Ask optimizer for next points to explore
    next_points = state.optimizer.ask(n_points=args.batch_size)

    # Add new points and print them as formatted commands
    for point in next_points:
        state.points.append(point)
        state.losses.append('')
        state.recorded.append(False)
        print('{}: {}'.format(len(state.points) - 1, state.format_command(point)))

    # Save new state files
    state.save(args.state_out)


if __name__ == '__main__':
    main()

# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
Average parameters from multiple model checkpoints. Checkpoints can be either
specified manually or automatically chosen according to one of several
strategies. The default strategy of simply selecting the top-scoring N points
works well in practice.
"""

import argparse
import itertools
import logging
import os
from typing import Dict, Iterable, List

import torch

from . import arguments
from . import constants as C
from . import utils
from .log import setup_main_logger, log_sockeye_version

logger = logging.getLogger(__name__)


def average(param_paths: Iterable[str]) -> Dict[str, torch.Tensor]:
    """
    Averages parameters from a list of .params file paths.

    :param param_paths: List of paths to parameter files.
    :return: Averaged parameter dictionary.
    """
    all_params = []  # type: List[Dict[str, torch.Tensor]]
    for path in param_paths:
        logger.info("Loading parameters from '%s'", path)
        try:
            params = torch.load(path, map_location=torch.device('cpu'))
        except:
            logger.info('Converting from MXNet')
            from mxnet import npx
            params = npx.load(path)
            params = {k: torch.from_numpy(v.asnumpy()) for k, v in params.items()}
        all_params.append(params)

    logger.info("%d models loaded", len(all_params))
    utils.check_condition(all(all_params[0].keys() == p.keys() for p in all_params),
                          "param names do not match across models")

    avg_params = {}
    # average arg_params
    for k in all_params[0]:
        tensors = [p[k] for p in all_params]
        avg_params[k] = utils.average_tensors(tensors)
    return avg_params


def find_checkpoints(model_path: str, size=4, strategy="best", metric: str = C.PERPLEXITY) -> List[str]:
    """
    Finds N best points from .metrics file according to strategy.

    :param model_path: Path to model.
    :param size: Number of checkpoints to combine.
    :param strategy: Combination strategy.
    :param metric: Metric according to which checkpoints are selected.  Corresponds to columns in model/metrics file.
    :return: List of paths corresponding to chosen checkpoints.
    """
    maximize = C.METRIC_MAXIMIZE[metric]
    points = utils.get_validation_metric_points(model_path=model_path, metric=metric)
    # keep only points for which .param files exist
    param_path = os.path.join(model_path, C.PARAMS_NAME)
    points = [(value, checkpoint) for value, checkpoint in points if os.path.exists(param_path % checkpoint)]

    if strategy == C.AVERAGE_BEST:
        # N best scoring points
        top_n = strategy_best(points, size, maximize)

    elif strategy == C.AVERAGE_LAST:
        # N sequential points ending with overall best
        top_n = strategy_last(points, size, maximize)

    elif strategy == C.AVERAGE_LIFESPAN:
        # Track lifespan of every "new best" point
        # Points dominated by a previous better point have lifespan 0
        top_n = strategy_lifespan(points, size, maximize)
    else:
        raise RuntimeError("Unknown strategy, options: best last lifespan")

    # Assemble paths for params files corresponding to chosen checkpoints
    # Last element in point is always the checkpoint id
    params_paths = [
        os.path.join(model_path, C.PARAMS_NAME % point[-1]) for point in top_n
    ]

    # Report
    logger.info("Found: " + ", ".join(str(point) for point in top_n))

    return params_paths


def strategy_best(points, size, maximize):
    top_n = sorted(points, reverse=maximize)[:size]
    return top_n


def strategy_last(points, size, maximize):
    best = max if maximize else min
    after_top = points.index(best(points)) + 1
    top_n = points[max(0, after_top - size):after_top]
    return top_n


def strategy_lifespan(points, size, maximize):
    top_n = []
    cur_best = points[0]
    cur_lifespan = 0
    for point in points[1:]:
        better = point > cur_best if maximize else point < cur_best
        if better:
            top_n.append(list(itertools.chain([cur_lifespan], cur_best)))
            cur_best = point
            cur_lifespan = 0
        else:
            top_n.append(list(itertools.chain([0], point)))
            cur_lifespan += 1
    top_n.append(list(itertools.chain([cur_lifespan], cur_best)))
    # Sort by lifespan, then by val
    top_n = sorted(
        top_n,
        key=lambda point: [point[0], point[1] if maximize else -point[1]],
        reverse=True)[:size]
    return top_n


def main():
    """
    Commandline interface to average parameters.
    """
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description="Averages parameters from multiple models.")
    arguments.add_average_args(params)
    args = params.parse_args()
    average_parameters(args)


def average_parameters(args: argparse.Namespace):
    log_sockeye_version(logger)

    if len(args.inputs) > 1:
        avg_params = average(args.inputs)
    else:
        param_paths = find_checkpoints(model_path=args.inputs[0],
                                       size=args.n,
                                       strategy=args.strategy,
                                       metric=args.metric)
        avg_params = average(param_paths)

    torch.save(avg_params, args.output)
    logger.info("Averaged parameters written to '%s'", args.output)


if __name__ == "__main__":
    main()

# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Average parameters from multiple model checkpoints.  Checkpoints can be either
specificed manually or automatically chosen according to one of several
strategies.  The default strategy of simply selecting the top-scoring N points
works well in practice.
"""

import argparse
import itertools
import os
from typing import Dict, Iterable, Tuple

import mxnet as mx

import sockeye.constants as C
import sockeye.utils
from sockeye.log import setup_main_logger

logger = setup_main_logger(__name__, console=True, file_logging=False)


def average(param_paths: Iterable[str]) -> Dict[str, mx.nd.NDArray]:
    """
    Averages parameters from a list of .params file paths.

    :param param_paths: List of paths to parameter files.
    :return: Averaged parameter dictionary.
    """
    all_arg_params = []
    all_aux_params = []
    for path in param_paths:
        logger.info("Loading parameters from '%s'", path)
        arg_params, aux_params = sockeye.utils.load_params(path)
        all_arg_params.append(arg_params)
        all_aux_params.append(aux_params)

    logger.info("%d models loaded", len(all_arg_params))
    assert all(
        all_arg_params[0].keys() == p.keys()
        for p in all_arg_params), "arg_param names do not match across models"
    assert all(
        all_aux_params[0].keys() == p.keys()
        for p in all_aux_params), "aux_param names do not match across models"

    avg_params = {}
    # average arg_params
    for k in all_arg_params[0]:
        arrays = [p[k] for p in all_arg_params]
        avg_params["arg:" + k] = sockeye.utils.average_arrays(arrays)
    # average aux_params
    for k in all_aux_params[0]:
        arrays = [p[k] for p in all_aux_params]
        avg_params["aux:" + k] = sockeye.utils.average_arrays(arrays)

    return avg_params


def find_checkpoints(model_path: str, size=4, strategy="best", maximize=False) -> Iterable[str]:
    """
    Finds N best points from .metrics file according to strategy

    :param model_path: Path to model.
    :param size: Number of checkpoints to combine.
    :param strategy: Combination strategy.
    :param maximize: Whether the value of the metric should be maximized.
    :return: List of paths corresponding to chosen checkpoints.
    """
    metrics_path = os.path.join(model_path, C.METRICS_NAME)
    points = read_metrics_points(metrics_path)

    if strategy == "best":
        # N best scoring points
        top_n = sorted(points, reverse=maximize)[:size]

    elif strategy == "last":
        # N sequential points ending with overall best
        best = max if maximize else min
        after_top = points.index(best(points)) + 1
        top_n = points[after_top - size:after_top]

    elif strategy == "lifespan":
        # Track lifespan of every "new best" point
        # Points dominated by a previous better point have lifespan 0
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


def read_metrics_points(path: str) -> Iterable[Tuple[float, int]]:
    """
    Reads lines from .metrics file and return list of elements [val, checkpoint]

    :param path: File to read metric values from.
    :return: List of pairs (metric value, checkpoint).
    """
    points = []
    # First field is checkpoint id
    # Metric on validation (dev) set looks like this: METRIC-val=N
    with open(path, "r") as metrics_in:
        for line in metrics_in:
            fields = line.split()
            checkpoint = int(fields[0])
            for field in fields[1:]:
                key_value = field.split("=")
                if len(key_value) == 2:
                    metric_set = key_value[0].split("-")
                    if len(metric_set) == 2 and metric_set[0] != C.ACCURACY and metric_set[1] == "val":
                        metric_value = float(key_value[1])
                        points.append([metric_value, checkpoint])
    return points


def main():
    """
    Commandline interface to average parameters.
    """
    params = argparse.ArgumentParser(
        description="Averages parameters from multiple models.")
    params.add_argument(
        "inputs",
        metavar="INPUT",
        type=str,
        nargs="+",
        help="either a single model directory (automatic checkpoint selection) "
             "or multiple .params files (manual checkpoint selection)")
    params.add_argument(
        "--max", action="store_true", help="maximize metric (default: min)")
    params.add_argument(
        "-n",
        type=int,
        default=4,
        help="number of checkpoints to find (default: 4)")
    params.add_argument(
        "--output", "-o", required=True, type=str, help="output param file")
    params.add_argument(
        "--strategy",
        choices=["best", "last", "lifespan"],
        default="best",
        help="selection method (default: best)")
    args = params.parse_args()

    if len(args.inputs) > 1:
        avg_params = average(args.inputs)
    else:
        param_paths = find_checkpoints(args.inputs[0], args.n, args.strategy,
                                       args.max)
        avg_params = average(param_paths)

    mx.nd.save(args.output, avg_params)
    logger.info("Averaged parameters written to '%s'", args.output)


if __name__ == "__main__":
    main()

import mxnet as mx
import logging
import os
from . import model
from . import data_io

from typing import Dict, List

logger = logging.getLogger(__name__)


class ElasticWeightConsolidationPenalty(mx.gluon.Block):
    """
    TODO: make this a hybrid block
    """

    def __init__(self,
                 fisher_matrix: Dict[str, mx.nd.NDArray],
                 best_params: Dict[str, mx.nd.NDArray],
                 importance: float = 1):
        super().__init__()
        with self.name_scope():
            self.importance = importance
            self.fisher_matrix = fisher_matrix
            self.best_params = best_params

    def forward(self, params: Dict[str, mx.nd.NDArray]):
        loss = mx.nd.zeros((1,))
        for name, param in params.items():
            _loss = self.fisher_matrix[name] * (param - self.best_params[name]) ** 2
            loss = loss + _loss.sum()
        #print('EWC PENALTY', loss.asscalar())
        return self.importance * loss


def compute_fisher(model: model.SockeyeModel,
                   loss_functions: List,
                   train_iter: data_io.BaseParallelSampleIter,
                   context,
                   output_folder: str,
                   steps: int = 200) -> Dict[str, mx.nd.NDArray]:
    """
    Computes the Fisher information for all trainable parameter matrices and stores them in
    '<output_folder>/params.fisher'. Fisher information is collected over `steps` batches from the `train_iter`.
    No parameter updates are done to the model.
    ATTENTION: this method overwrites the parameter arrays of model to ensure Fisher information is stored under the
    same parameter names as the model parameters. TODO: make this less destructive.
    """

    params = model.collect_params()
    fisher_information = {}
    for name, param in params.items():
        fisher_information[name] = mx.nd.zeros_like(param.data())

    for _ in range(steps):
        batch = train_iter.next()

        # split batch into shards
        batch = batch.split_and_load(ctx=context)

        params.zero_grad()
        with mx.autograd.record():
            for inputs, labels in batch.shards():
                outputs = model(*inputs)  # type: Dict[str, mx.nd.NDArray]
                loss_outputs = [loss_function(outputs, labels) for loss_function in loss_functions]
                loss_values = (v for v, _ in loss_outputs)
                sum_losses = mx.nd.add_n(*loss_values)
                sum_losses.backward()
        for name, param in params.items():
            if param.grad_req != 'null':
                fisher_information[name] = fisher_information[name] + (param.grad() ** 2) / steps

        if not train_iter.iter_next():
            logger.info("End of epoch, resetting data iterator")
            train_iter.reset()

    logger.info("Computed Fisher information for %d parameters over %d batches",
                len(fisher_information), steps)

    # hacky way to store with the same parameter names as model parameters:
    # overwrite model parameters with the fisher information
    # for name, param in params.items():
    #     param.set_data(fisher_information[name])

    filename = os.path.join(output_folder, 'params.fisher')
    mx.nd.save(filename, fisher_information)
    logger.info("Saved Fisher information to '%s'", filename)

    return fisher_information

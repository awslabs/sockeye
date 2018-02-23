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
Provides functionality to track metrics on training and validation data during training and controls
early-stopping.
"""
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union

import mxnet as mx

from . import checkpoint_decoder
from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


class DecoderProcessManager(object):
    """
    TrainingMonitor logs metrics on training and validation data, submits decoding processes to compute BLEU scores,
    and writes metrics to the model output folder.
    It further controls early stopping as it decides based on the specified metric to optimize, whether the model
    has improved w.r.t to the last checkpoint.
    Technically, TrainingMonitor exposes a couple of callback function that are called in the fit() method of
    TrainingModel.

    :param output_folder: Folder where model files are written to.
    :param cp_decoder: Optional CheckpointDecoder instance for BLEU monitoring.
    """

    def __init__(self,
                 output_folder: str,
                 cp_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None) -> None:
        self.output_folder = output_folder
        self.cp_decoder = cp_decoder
        self.ctx = mp.get_context('spawn')  # type: ignore
        self.decoder_metric_queue = self.ctx.Queue()
        self.decoder_process = None  # type: Optional[mp.Process]

    def spawn(self, checkpoint: int):
        assert self.decoder_process is None
        output_name = os.path.join(self.output_folder, C.DECODE_OUT_NAME % checkpoint)
        process = self.ctx.Process(
            target=_decode_and_evaluate,
            args=(self.cp_decoder,
                  checkpoint,
                  output_name,
                  self.decoder_metric_queue))
        process.name = 'Decoder-%d' % checkpoint
        logger.info("Starting process: %s", process.name)
        process.start()
        self.decoder_process = process

    def collect_results(self) -> Optional[Tuple[int, Dict[str, float]]]:
        """
        Get metric results from decoder_process queue and optionally write to tensorboard logs
        """
        assert self.decoder_process is None
        if self.decoder_metric_queue.empty():
            return None
        else:
            decoded_checkpoint, decoder_metrics = self.decoder_metric_queue.get()
            assert self.decoder_metric_queue.empty()
            return decoded_checkpoint, decoder_metrics

    def wait_to_finish(self):
        if self.decoder_process is None:
            return
        if not self.decoder_process.is_alive():
            self.decoder_process = None
            return
        # Wait for the decoder to finish
        logger.warning("Waiting for process %s to finish.", self.decoder_process.name)
        wait_start = time.time()
        self.decoder_process.join()
        self.decoder_process = None
        wait_time = int(time.time() - wait_start)
        logger.warning("Had to wait %d seconds for the checkpoint decoder to finish. Consider increasing the "
                       "checkpoint frequency (updates between checkpoints, see %s) or reducing the size of the "
                       "validation samples that are decoded (see %s)." % (wait_time,
                                                                          C.TRAIN_ARGS_CHECKPOINT_FREQUENCY,
                                                                          C.TRAIN_ARGS_MONITOR_BLEU))


def _decode_and_evaluate(checkpoint_decoder: checkpoint_decoder.CheckpointDecoder,
                         checkpoint: int,
                         output_name: str,
                         queue: mp.Queue):
    """
    Decodes and evaluates using given checkpoint_decoder and puts result in the queue,
    indexed by the checkpoint.
    """
    metrics = checkpoint_decoder.decode_and_evaluate(checkpoint, output_name)
    queue.put((checkpoint, metrics))
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os
import pickle
import random
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from unittest.mock import patch

import mxnet as mx
import numpy as np

import sockeye.average
import sockeye.constants as C
import sockeye.image_captioning.captioner
import sockeye.image_captioning.extract_features
import sockeye.image_captioning.train
from sockeye.evaluate import raw_corpus_bleu, raw_corpus_chrf

try:  # Try to import pillow
    from PIL import Image  # pylint: disable=import-error
except ImportError as e:
    raise RuntimeError("Please install pillow.")

logger = logging.getLogger(__name__)


_DIGITS = "0123456789"
_IMAGE_SHAPE = (100, 100, 3)
_CNN_INPUT_IMAGE_SHAPE = (3, 224, 224)
_FEATURE_SHAPE = (2048, 7, 7)


def generate_img_or_feat(filename, use_features):
    if not use_features:
        imarray = np.random.rand(*_IMAGE_SHAPE) * 255
        im = Image.fromarray(imarray.astype('uint8'))
        im.save(filename)
    else:
        data = np.random.rand(*_FEATURE_SHAPE)
        np.save(filename, data)


def generate_img_text_experiment_files(
                            source_list: List[str],
                            work_dir: str,
                            source_path: str,
                            target_path: str,
                            line_length: int = 9,
                            use_features: bool = False,
                            seed=13):
    random_gen = random.Random(seed)
    with open(source_path, "w") as source_out, open(target_path, "w") as target_out:
        source_list_img = []
        for s in source_list:
            if not use_features:
                filename = s + ".jpg"
            else:
                filename = s + ".npy"
            source_list_img.append(os.path.join(work_dir, filename))
            print(filename, file=source_out)
            digits = [random_gen.choice(_DIGITS) for _ in range(random_gen.randint(1, line_length))]
            print(" ".join(digits), file=target_out)
        # Create random images/features
        for s in source_list_img:
            filename = os.path.join(work_dir, s)
            generate_img_or_feat(filename, use_features)
        # Generate and save the image size and feature size
        size_out_file = os.path.join(work_dir, "image_feature_sizes.pkl")
        with open(size_out_file, "wb") as fout:
            pickle.dump({"image_shape": _CNN_INPUT_IMAGE_SHAPE,
                         "features_shape": _FEATURE_SHAPE}, fout)


@contextmanager
def tmp_img_captioning_dataset(
                        source_list: List[str],
                        prefix: str,
                        train_max_length: int,
                        dev_max_length: int,
                        test_max_length: int,
                        use_features: bool = False,
                        seed_train: int = 13,
                        seed_dev: int = 13):
    with TemporaryDirectory(prefix=prefix) as work_dir:
        # Simple digits files for train/dev data
        train_source_path = os.path.join(work_dir, "train.src")
        train_target_path = os.path.join(work_dir, "train.tgt")
        dev_source_path = os.path.join(work_dir, "dev.src")
        dev_target_path = os.path.join(work_dir, "dev.tgt")
        test_source_path = os.path.join(work_dir, "test.src")
        test_target_path = os.path.join(work_dir, "test.tgt")
        generate_img_text_experiment_files(source_list, work_dir, train_source_path, train_target_path,
                             train_max_length, use_features, seed=seed_train)
        generate_img_text_experiment_files(source_list, work_dir, dev_source_path, dev_target_path,
                                 dev_max_length, use_features, seed=seed_dev)
        generate_img_text_experiment_files(source_list, work_dir, test_source_path, test_target_path,
                                 test_max_length, use_features, seed=seed_dev)
        data = {'work_dir': work_dir,
                'source': train_source_path,
                'target': train_target_path,
                'validation_source': dev_source_path,
                'validation_target': dev_target_path,
                'test_source': test_source_path,
                'test_target': test_target_path}

        yield data


_CAPTION_TRAIN_PARAMS_COMMON = \
    "--use-cpu --max-seq-len {max_len} --source-root {source_root} --source {train_source} --target {train_target}" \
    " --validation-source-root {dev_root} --validation-source {dev_source} --validation-target {dev_target} --output {model} {quiet}" \
    " --seed {seed}"

_CAPTIONER_PARAMS_COMMON = "--use-cpu --models {model}  --source-root {source_root} --input {input} --output {output} {quiet}"

def run_train_captioning(train_params: str,
                        translate_params: str,
                        translate_params_equiv: Optional[str],
                        train_source_path: str,
                        train_target_path: str,
                        dev_source_path: str,
                        dev_target_path: str,
                        test_source_path: str,
                        test_target_path: str,
                        max_seq_len: int = 10,
                        work_dir: Optional[str] = None,
                        seed: int = 13,
                        quiet: bool = False) -> Tuple[float, float, float, float]:
    """
    Train a model and caption a dev set.  Report validation perplexity and BLEU.

    :param train_params: Command line args for model training.
    :param translate_params: First command line args for translation.
    :param translate_params_equiv: Second command line args for captuoning. Should produce the same outputs
    :param train_source_path: Path to the source file.
    :param train_target_path: Path to the target file.
    :param dev_source_path: Path to the development source file.
    :param dev_target_path: Path to the development target file.
    :param test_source_path: Path to the test source file.
    :param test_target_path: Path to the test target file.
    :param max_seq_len: The maximum sequence length.
    :param work_dir: The directory to store the model and other outputs in.
    :param seed: The seed used for training.
    :param quiet: Suppress the console output of training and decoding.
    :return: A tuple containing perplexity, bleu scores for standard and reduced vocab decoding, chrf score.
    """
    source_root = work_dir
    if quiet:
        quiet_arg = "--quiet"
    else:
        quiet_arg = ""
    with TemporaryDirectory(dir=work_dir, prefix="test_train_translate.") as work_dir:
        # Train model
        model_path = os.path.join(work_dir, "model")
        params = "{} {} {}".format(sockeye.image_captioning.train.__file__,
                                   _CAPTION_TRAIN_PARAMS_COMMON.format(
                                       source_root=source_root,
                                       train_source=train_source_path,
                                       train_target=train_target_path,
                                       dev_root=source_root,
                                       dev_source=dev_source_path,
                                       dev_target=dev_target_path,
                                       model=model_path,
                                       max_len=max_seq_len,
                                       seed=seed,
                                       quiet=quiet_arg),
                                   train_params)

        logger.info("Starting training with parameters %s.", train_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.image_captioning.train.main()

        logger.info("Translating with parameters %s.", translate_params)
        # Translate corpus with the 1st params
        out_path = os.path.join(work_dir, "out.txt")
        params = "{} {} {}".format(sockeye.image_captioning.captioner.__file__,
                                   _CAPTIONER_PARAMS_COMMON.format(model=model_path,
                                                                   source_root=source_root,
                                                                   input=test_source_path,
                                                                   output=out_path,
                                                                   quiet=quiet_arg),
                                   translate_params)
        with patch.object(sys, "argv", params.split()):
            sockeye.image_captioning.captioner.main()

        # Translate corpus with the 2nd params
        if translate_params_equiv is not None:
            out_path_equiv = os.path.join(work_dir, "out_equiv.txt")
            params = "{} {} {}".format(sockeye.image_captioning.captioner.__file__,
                                   _CAPTIONER_PARAMS_COMMON.format(model=model_path,
                                                                   source_root=source_root,
                                                                   input=test_source_path,
                                                                   output=out_path_equiv,
                                                                   quiet=quiet_arg),
                                    translate_params_equiv)
            with patch.object(sys, "argv", params.split()):
                sockeye.image_captioning.captioner.main()
            # read-in both outputs, ensure they are the same
            with open(out_path, 'rt') as f:
                lines = f.readlines()
            with open(out_path_equiv, 'rt') as f:
                lines_equiv = f.readlines()
            assert all(a == b for a, b in zip(lines, lines_equiv))

        # test averaging
        points = sockeye.average.find_checkpoints(model_path=model_path,
                                                  size=1,
                                                  strategy='best',
                                                  metric=C.PERPLEXITY)
        assert len(points) > 0
        averaged_params = sockeye.average.average(points)
        assert averaged_params

        # get best validation perplexity
        metrics = sockeye.utils.read_metrics_file(path=os.path.join(model_path, C.METRICS_NAME))
        perplexity = min(m[C.PERPLEXITY + '-val'] for m in metrics)
        hypotheses = open(out_path, "r").readlines()
        references = open(test_target_path, "r").readlines()
        assert len(hypotheses) == len(references)
        # compute metrics
        bleu = raw_corpus_bleu(hypotheses=hypotheses, references=references, offset=0.01)
        chrf = raw_corpus_chrf(hypotheses=hypotheses, references=references)

        return perplexity, bleu, chrf


_EXTRACT_FEATURES_PARAMS_COMMON = \
    "--use-cpu --image-root {image_root} --input {source_file} --output-root {output_root} " \
    "--output {output_file} --image-encoder-model-path {image_encoder_model_path}"


def run_extract_features_captioning(source_image_size: tuple,
                                    batch_size: int,
                                    extract_params: str,
                                    source_files: List[str],
                                    image_root: str) -> None:

    with TemporaryDirectory(dir=image_root, prefix="test_extract_feats") as work_dir:
        model_path = os.path.join(work_dir, '2-conv-layer')
        epoch = 0
        # Create net and save to disk
        create_simple_and_save_to_disk(model_path, epoch, source_image_size, batch_size)

        # Extract features
        for s in source_files:
            with TemporaryDirectory(dir=work_dir, prefix="extracted_feats") as local_work_dir:
                output_root = local_work_dir
                output_file = os.path.join(local_work_dir, "random.features")
                params = "{} {} {}".format(sockeye.image_captioning.extract_features.__file__,
                                           _EXTRACT_FEATURES_PARAMS_COMMON.format(
                                               image_root=image_root,
                                               source_file=s,
                                               output_root=output_root,
                                               output_file=output_file,
                                               image_encoder_model_path=model_path
                                           ),
                                           extract_params)

                logger.info("Starting feature extractopm with parameters %s.", extract_params)
                with patch.object(sys, "argv", params.split()):
                    sockeye.image_captioning.extract_features.main()


def create_simple_and_save_to_disk(prefix, iteration, source_image_size, batch_size):
    # init model
    sym = get_2convnet_symbol()
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (batch_size,) + source_image_size)],
             label_shapes=[('softmax_label', (batch_size, 1))])
    mod.init_params()
    # save
    mod.save_checkpoint(prefix, iteration)


def get_2convnet_symbol():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20, name='conv1')
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50, name='conv2')
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    flatten = mx.symbol.Flatten(data=pool2)
    fc2 = mx.symbol.FullyConnected(data=flatten, num_hidden=1)
    # loss
    outsym = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return outsym
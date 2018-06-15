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

import random
import string

import pytest

from test.common_image_captioning import run_train_captioning, tmp_img_captioning_dataset

_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20

ENCODER_DECODER_SETTINGS = [
    # 2-layer LSTM decoder with attention
    ("--encoder image-pretrain-cnn --image-encoder-num-hidden 8 --decoder rnn --rnn-cell-type lstm "
     "--batch-type sentence --batch-size 2 "
     "--initial-learning-rate 0.0003 --gradient-clipping-threshold 1.0 --bucket-width 2 "
     "--rnn-num-hidden 8 --rnn-decoder-state-init zero --weight-normalization "
     "--checkpoint-frequency 2 --max-updates 2 --num-layers 1:2 ",
     "--beam-size 2"),
    # LSTM decoder with attention: no global, encoder hiddens 8, rnn last, load all feats to mem
    ("--encoder image-pretrain-cnn --image-encoder-num-hidden 8 --no-image-encoder-global-descriptor "
     "--decoder rnn --rnn-cell-type lstm --batch-size 12 --optimizer adam --load-all-features-to-memory "
     "--initial-learning-rate 0.0003 --gradient-clipping-threshold 1.0 --bucket-width 2 "
     "--rnn-num-hidden 8 --rnn-decoder-state-init last --weight-normalization "
     "--checkpoint-frequency 2 --max-updates 2",
     "--beam-size 2"),
    # Transformer decoder
    ("--encoder image-pretrain-cnn --image-encoder-num-hidden 8 --decoder transformer --batch-size 12 --num-embed 4 "
     "--transformer-attention-heads 2 --transformer-model-size 4 --transformer-feed-forward-num-hidden 8 "
     "--initial-learning-rate 0.0003 --gradient-clipping-threshold 1.0 --bucket-width 2 "
     "--checkpoint-frequency 2 --max-updates 2",
     "--beam-size 2"),
    # 2-layer CNN decoder
    ("--encoder image-pretrain-cnn --decoder cnn --num-layers 2 --batch-size 12 "
     "--initial-learning-rate 0.0003 "
     "--cnn-num-hidden 8 --image-encoder-num-hidden 8 --cnn-positional-embedding-type fixed "
     "--checkpoint-frequency 2 --max-updates 2",
     "--beam-size 2")
]


@pytest.mark.parametrize("train_params, translate_params",
                         ENCODER_DECODER_SETTINGS)
def test_caption_random_features(train_params: str, translate_params: str):
    # generate random names
    source_list = [''.join(random.choice(string.ascii_uppercase) for _ in range(4)) for i in range(15)]
    prefix = "tmp_caption_ramdom"
    use_features = True
    with tmp_img_captioning_dataset(source_list,
                                    prefix,
                                    train_max_length=_LINE_MAX_LENGTH,
                                    dev_max_length=_LINE_MAX_LENGTH,
                                    test_max_length=_TEST_MAX_LENGTH,
                                    use_features=use_features) as data:
        # Test model configuration, including the output equivalence of batch and no-batch decoding
        translate_params_batch = translate_params + " --batch-size 2"

        # Ignore return values (perplexity and BLEU) for integration test
        run_train_captioning(train_params=train_params,
                             translate_params=translate_params,
                             translate_params_equiv=translate_params_batch,
                             train_source_path=data['source'],
                             train_target_path=data['target'],
                             dev_source_path=data['validation_source'],
                             dev_target_path=data['validation_target'],
                             test_source_path=data['test_source'],
                             test_target_path=data['test_target'],
                             max_seq_len=_LINE_MAX_LENGTH + 1,
                             work_dir=data['work_dir'])

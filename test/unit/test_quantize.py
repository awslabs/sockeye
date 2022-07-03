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

import argparse
import os
import tempfile

import torch

import sockeye.constants as C
import sockeye.data_io
import sockeye.encoder
import sockeye.model
import sockeye.quantize


def test_quantize():

    config_embed = sockeye.encoder.EmbeddingConfig(vocab_size=20, num_embed=4, dropout=0.0)
    config_encoder = sockeye.encoder.EncoderConfig(model_size=4, attention_heads=1, feed_forward_num_hidden=4,
                                                   act_type='relu', num_layers=1, dropout_attention=0.0,
                                                   dropout_act=0.0, dropout_prepost=0.0,
                                                   positional_embedding_type='fixed', preprocess_sequence='none',
                                                   postprocess_sequence='none', max_seq_len_source=30,
                                                   max_seq_len_target=30)
    config_data = sockeye.data_io.DataConfig(data_statistics=None, max_seq_len_source=30, max_seq_len_target=30,
                                             num_source_factors=0, num_target_factors=0)
    config = sockeye.model.ModelConfig(config_data=config_data, vocab_source_size=20, vocab_target_size=20,
                                       config_embed_source=config_embed, config_embed_target=config_embed,
                                       config_encoder=config_encoder, config_decoder=config_encoder)

    with tempfile.TemporaryDirectory() as model_dir:
        params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
        backup_params_fname = f'{params_fname}.{config.dtype}'

        # Create and save float32 model
        model = sockeye.model.SockeyeModel(config=config)
        assert model.dtype == torch.float32
        for param in model.parameters():
            assert param.dtype == torch.float32
        model.save_config(model_dir)
        model.save_version(model_dir)
        model.save_parameters(params_fname)
        del model

        # Quantize offline to float16
        quantize_args = argparse.Namespace(model=model_dir, dtype=C.DTYPE_FP16, quiet=False, loglevel='INFO')
        for _ in range(2):
            # First call quantizes, second is a no-op
            sockeye.quantize.quantize(quantize_args)

        # Check params saved to disk
        fp16_params = torch.load(params_fname)
        fp32_params = torch.load(backup_params_fname)
        assert set(fp16_params.keys()) == set(fp32_params.keys())
        for key in fp16_params.keys():
            assert fp16_params[key].dtype == torch.float16
            assert fp32_params[key].dtype == torch.float32
        del fp16_params
        del fp32_params

        # Check loaded float16 model
        model, _, _ = sockeye.model.load_model(model_dir)
        assert model.dtype == torch.float16
        for param in model.parameters():
            assert param.dtype == torch.float16

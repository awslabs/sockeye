# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import glob
import itertools
import os.path
import tempfile

import pytest
import torch as pt

import sockeye.constants as C
import sockeye.encoder
import sockeye.model
import sockeye.training


def test_cleanup_param_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n in itertools.chain(range(1, 20, 2), range(21, 41)):
            # Create empty files
            open(os.path.join(tmp_dir, C.PARAMS_NAME % n), "w").close()
        sockeye.training.cleanup_params_files(tmp_dir, 5, 40, 17, False, 8, "perplexity", "best")

        expectedSurviving = set([os.path.join(tmp_dir, C.PARAMS_NAME % n)
                                 for n in [17, 36, 37, 38, 39, 40]])
        # 17 must survive because it is the best one
        assert set(glob.glob(os.path.join(tmp_dir, C.PARAMS_PREFIX + "*"))) == expectedSurviving


def test_cleanup_param_files_keep_first():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n in itertools.chain(range(0, 20, 2), range(21, 41)):
            # Create empty files
            open(os.path.join(tmp_dir, C.PARAMS_NAME % n), "w").close()
        sockeye.training.cleanup_params_files(tmp_dir, 5, 40, 16, True, 8, "perplexity", "best")

        expectedSurviving = set([os.path.join(tmp_dir, C.PARAMS_NAME % n)
                                 for n in [0, 16, 36, 37, 38, 39, 40]])
        # 16 must survive because it is the best one
        # 0 should also survive because we set keep_first to True
        assert set(glob.glob(os.path.join(tmp_dir, C.PARAMS_PREFIX + "*"))) == expectedSurviving


def mock_model():
    config_embed = sockeye.encoder.EmbeddingConfig(vocab_size=20, num_embed=4, dropout=0.0)
    config_encoder = sockeye.encoder.EncoderConfig(model_size=4, attention_heads=1, feed_forward_num_hidden=4,
                                                   act_type='relu', num_layers=1, dropout_attention=0.0,
                                                   dropout_act=0.0, dropout_prepost=0.0,
                                                   positional_embedding_type='fixed', preprocess_sequence='none',
                                                   postprocess_sequence='none', max_seq_len_source=30,
                                                   max_seq_len_target=30)
    config = sockeye.model.ModelConfig(config_data=None, vocab_source_size=20, vocab_target_size=20,
                                       config_embed_source=config_embed, config_embed_target=config_embed,
                                       config_encoder=config_encoder, config_decoder=config_encoder,
                                       weight_tying_type='none')
    model = sockeye.model.SockeyeModel(config=config)
    return model


def test_set_parameters():
    model = mock_model()
    sockeye.model.initialize_parameters(model)
    model_params = dict(model.named_parameters())

    param = pt.nn.Parameter(pt.ones(20, 4))
    name = 'output_layer.weight'
    model.set_parameters({name: param})

    pt.testing.assert_allclose(model_params['output_layer.weight'].data, param.data)


def test_set_parameters_allow_missing():
    model = mock_model()
    sockeye.model.initialize_parameters(model)
    model_params = dict(model.named_parameters())
    model.set_parameters({}, allow_missing=True)
    assert 'embedding_source.embedding.weight' in model_params
    with pytest.raises(AssertionError) as e:
        model.set_parameters({}, allow_missing=False)
    assert str(e.value) == "Parameter 'embedding_source.embedding.weight' is missing in new_params dictionary. " \
                           "Set allow_missing=True to ignore missing parameters."


def test_set_parameters_ignore_extra():
    model = mock_model()
    sockeye.model.initialize_parameters(model)
    model_params = dict(model.named_parameters())

    p = pt.nn.Parameter(pt.ones(20, 4))
    np = 'embedding_source.embedding.weight'
    q = pt.nn.Parameter(pt.zeros(1, 1))
    nq = 'q'
    params = {np: p, nq: q}
    model.set_parameters(params, ignore_extra=True)
    assert 'embedding_source.embedding.weight' in model_params
    assert 'q' not in model_params
    with pytest.raises(ValueError) as e:
        model.set_parameters(params, ignore_extra=False)
    assert str(e.value) == "Parameter 'q' in new_params dictionary is not present in ParameterDict. " \
                           "Set ignore_extra=True to ignore."

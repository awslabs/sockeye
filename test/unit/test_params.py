# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import itertools
import glob
import os.path
import tempfile

import mxnet as mx
import pytest

import sockeye.encoder
import sockeye.model
import sockeye.training
import sockeye.constants as C
import sockeye.utils


def test_cleanup_param_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n in itertools.chain(range(1, 20, 2), range(21, 41)):
            # Create empty files
            open(os.path.join(tmp_dir, C.PARAMS_NAME % n), "w").close()
        sockeye.utils.cleanup_params_files(tmp_dir, 5, 40, 17, False)

        expectedSurviving = set([os.path.join(tmp_dir, C.PARAMS_NAME % n)
                                 for n in [17, 36, 37, 38, 39, 40]])
        # 17 must survive because it is the best one
        assert set(glob.glob(os.path.join(tmp_dir, C.PARAMS_PREFIX + "*"))) == expectedSurviving


def test_cleanup_param_files_keep_first():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n in itertools.chain(range(0, 20, 2), range(21, 41)):
            # Create empty files
            open(os.path.join(tmp_dir, C.PARAMS_NAME % n), "w").close()
        sockeye.utils.cleanup_params_files(tmp_dir, 5, 40, 16, True)

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
                                       config_encoder=config_encoder, config_decoder=config_encoder)
    model = sockeye.model.SockeyeModel(config=config)
    return model


def test_set_parameters():
    model = mock_model()
    model.initialize(init='xavier', ctx=mx.cpu(0))
    p = mx.gluon.Parameter('source_target_embed_weight', shape=(20, 4))
    p.initialize(init='xavier', ctx=mx.cpu(0))
    model.set_parameters({'source_target_embed_weight': p})
    assert mx.test_utils.same(model.params['source_target_embed_weight'].data(), p.data())


def test_set_parameters_allow_missing():
    model = mock_model()
    model.initialize(init='xavier', ctx=mx.cpu(0))
    model.set_parameters({}, allow_missing=True)
    assert 'source_target_embed_weight' in model.params
    with pytest.raises(AssertionError) as e:
        model.set_parameters({}, allow_missing=False)
    assert str(e.value) == "Parameter 'source_target_embed_weight' is missing in new_params dictionary. " \
                           "Set allow_missing=True to ignore missing parameters."


def test_set_parameters_ignore_extra():
    model = mock_model()
    model.initialize(init='xavier', ctx=mx.cpu(0))
    p = mx.gluon.Parameter('source_target_embed_weight', shape=(20, 4))
    p.initialize(init='xavier', ctx=mx.cpu(0))
    q = mx.gluon.Parameter('q', shape=(1, 1))
    q.initialize(init='xavier', ctx=mx.cpu(0))
    params = {'source_target_embed_weight': p, 'q': q}
    model.set_parameters(params, ignore_extra=True)
    assert 'source_target_embed_weight' in model.params
    assert 'q' not in model.params
    with pytest.raises(ValueError) as e:
        model.set_parameters(params, ignore_extra=False)
    assert str(e.value) == "Parameter 'q' in new_params dictionary is not preset in ParameterDict. " \
                           "Set ignore_extra=True to ignore."


def test_set_parameters_context():
    model = mock_model()
    model.initialize(init='xavier', ctx=[mx.cpu(0), mx.cpu(1)])
    p = mx.gluon.Parameter('source_target_embed_weight', shape=(20, 4))
    p.initialize(init='xavier', ctx=mx.cpu(2))
    model.set_parameters({'source_target_embed_weight': p})
    for i in range(2):
        assert mx.test_utils.same(model.params['source_target_embed_weight'].data(mx.cpu(i)), p.data(mx.cpu(2)))


def test_set_parameters_shape():
    model = mock_model()
    model.initialize(init='xavier', ctx=mx.cpu(0))
    p = mx.gluon.Parameter('source_target_embed_weight', shape=(10, 10))
    p.initialize(init='xavier', ctx=mx.cpu(0))
    with pytest.raises(AssertionError) as e:
        model.set_parameters({'source_target_embed_weight': p})
    assert str(e.value) == "Parameter 'source_target_embed_weight' has shape '(20, 4)' in the model but shape " \
                           "'(10, 10)' in the new_params dictionary."


def test_set_parameters_uninitialized():
    model = mock_model()
    model.initialize(init='xavier', ctx=mx.cpu(0))
    p = mx.gluon.Parameter('source_target_embed_weight', shape=(20, 4))
    with pytest.raises(AssertionError) as e:
        model.set_parameters({'source_target_embed_weight': p})
    assert str(e.value) == "Parameter 'source_target_embed_weight' is not initialized in new_params dictionary."
    p.initialize(init='xavier', ctx=mx.cpu(0))
    model = mock_model()
    with pytest.raises(AssertionError) as e:
        model.set_parameters({'source_target_embed_weight': p})
    assert str(e.value) == "Parameter 'source_target_embed_weight' must be initialized before it can be reset using " \
                           "set_parameters."

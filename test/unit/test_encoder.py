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

import pytest

import sockeye.constants as C
import sockeye.encoder
import sockeye.transformer


@pytest.mark.parametrize('dropout, project_to_size, factor_configs, is_source', [
    (0., None, None, False),
    (0.1, 20, [sockeye.encoder.FactorConfig(vocab_size=5, num_embed=5)], True),
])
def test_embedding_encoder(dropout, project_to_size, factor_configs, is_source):
    config = sockeye.encoder.EmbeddingConfig(vocab_size=20, num_embed=10, dropout=dropout, project_to_size=project_to_size, factor_configs=factor_configs)
    embedding = sockeye.encoder.Embedding(config, prefix='embedding', is_source=is_source)
    assert type(embedding) == sockeye.encoder.Embedding


@pytest.mark.parametrize('shared_layer_params, lhuc, sandwich_recipe', [
    (False, False, (0, 0, 0)),
    (True, True, (10, 20, 10))
])
def test_get_transformer_encoder(shared_layer_params, lhuc, sandwich_recipe):
    prefix = "test_"
    config = sockeye.transformer.TransformerConfig(model_size=20,
                                                   attention_heads=10,
                                                   feed_forward_num_hidden=30,
                                                   act_type='test_act',
                                                   num_layers=40,
                                                   dropout_attention=1.0,
                                                   dropout_act=2.0,
                                                   dropout_prepost=3.0,
                                                   positional_embedding_type=C.LEARNED_POSITIONAL_EMBEDDING,
                                                   preprocess_sequence='test_pre',
                                                   postprocess_sequence='test_post',
                                                   max_seq_len_source=50,
                                                   max_seq_len_target=60,
                                                   sandwich_recipe=sandwich_recipe,
                                                   shared_layer_params=shared_layer_params,
                                                   lhuc=lhuc)
    encoder = sockeye.encoder.get_transformer_encoder(config, prefix=prefix)
    encoder.initialize()
    encoder.hybridize(static_alloc=True)

    assert type(encoder) == sockeye.encoder.TransformerEncoder
    assert encoder.prefix == prefix + C.TRANSFORMER_ENCODER_PREFIX

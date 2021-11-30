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

import numpy as onp
import pytest
import torch as pt

import sockeye.constants as C
import sockeye.encoder_pt
import sockeye.transformer_pt


@pytest.mark.parametrize('dropout, factor_configs', [
    (0., None),
    (0.1, [sockeye.encoder_pt.FactorConfig(vocab_size=5,
                                           num_embed=5,
                                           combine=C.FACTORS_COMBINE_SUM,
                                           share_embedding=False)]),
])
def test_embedding_encoder(dropout, factor_configs):
    config = sockeye.encoder_pt.EmbeddingConfig(vocab_size=20, num_embed=10, dropout=dropout,
                                                factor_configs=factor_configs)
    embedding = sockeye.encoder_pt.PyTorchEmbedding(config)
    assert type(embedding) == sockeye.encoder_pt.PyTorchEmbedding


@pytest.mark.parametrize('lhuc', [
    (False,),
    (True,)
])
def test_get_transformer_encoder(lhuc):
    config = sockeye.transformer_pt.TransformerConfig(model_size=20,
                                                      attention_heads=10,
                                                      feed_forward_num_hidden=30,
                                                      act_type='test_act',
                                                      num_layers=40,
                                                      dropout_attention=1.0,
                                                      dropout_act=0.5,
                                                      dropout_prepost=0.2,
                                                      positional_embedding_type=C.LEARNED_POSITIONAL_EMBEDDING,
                                                      preprocess_sequence='test_pre',
                                                      postprocess_sequence='test_post',
                                                      max_seq_len_source=50,
                                                      max_seq_len_target=60,
                                                      use_lhuc=lhuc)
    encoder = sockeye.encoder_pt.pytorch_get_transformer_encoder(config)
    assert type(encoder) == sockeye.encoder_pt.PyTorchTransformerEncoder


def test_mx_pt_eq_transformer_encoder():
    pytest.importorskip("mxnet")
    import sockeye.transformer
    import sockeye.encoder
    import mxnet as mx
    from mxnet import np

    pt.manual_seed(13)
    mx.random.seed(13)
    config = sockeye.transformer.TransformerConfig(model_size=128,
                                                   attention_heads=8,
                                                   feed_forward_num_hidden=256,
                                                   act_type='relu',
                                                   num_layers=12,
                                                   dropout_attention=0,
                                                   dropout_act=0,
                                                   dropout_prepost=0,
                                                   positional_embedding_type=C.LEARNED_POSITIONAL_EMBEDDING,
                                                   preprocess_sequence='n',
                                                   postprocess_sequence='r',
                                                   max_seq_len_source=50,
                                                   max_seq_len_target=60,
                                                   use_lhuc=False)
    encoder_mx = sockeye.encoder.get_transformer_encoder(config, dtype=C.DTYPE_FP32)
    encoder_mx.initialize()

    encoder_pt = sockeye.encoder_pt.pytorch_get_transformer_encoder(config)
    encoder_pt.weights_from_mxnet_block(encoder_mx)

    batch = 12
    seq_len = 45
    data_mx = np.random.uniform(0, 1, (batch, seq_len, config.model_size))
    data_pt = pt.as_tensor(data_mx.asnumpy())
    lengths_mx = np.random.randint(1, seq_len, (batch,))
    lengths_pt = pt.as_tensor(lengths_mx.asnumpy())

    r1_mx, r2_mx = encoder_mx(data_mx, lengths_mx)
    r1_pt, r2_pt = encoder_pt(data_pt, lengths_pt)

    r1_mx, r2_mx = r1_mx.asnumpy(), r2_mx.asnumpy()
    r1_pt, r2_pt = r1_pt.detach().numpy(), r2_pt.detach().numpy()

    print("Max deviation:", onp.abs(r1_mx - r1_pt).max())
    assert np.allclose(r1_mx, r1_pt, atol=1e-04)
    assert np.allclose(r2_mx, r2_pt, atol=1e-04)


@pytest.mark.parametrize('vocab_size, num_embed, factor_configs, sparse',
                         [(300, 32, None, False),
                          (300, 32, None, True),
                          (300, 32, [sockeye.encoder_pt.FactorConfig(300, 8, C.FACTORS_COMBINE_CONCAT, False),
                                     sockeye.encoder_pt.FactorConfig(300, 32, C.FACTORS_COMBINE_AVERAGE, False),
                                     sockeye.encoder_pt.FactorConfig(300, 32, C.FACTORS_COMBINE_AVERAGE, True),
                                     sockeye.encoder_pt.FactorConfig(300, 32, C.FACTORS_COMBINE_SUM, True)], True)])
def test_mx_pt_eq_embedding(vocab_size, num_embed, factor_configs, sparse):
    pytest.importorskip("mxnet")
    import sockeye.encoder
    from mxnet import np

    config = sockeye.encoder.EmbeddingConfig(vocab_size=vocab_size,
                                             num_embed=num_embed,
                                             dropout=0,
                                             factor_configs=factor_configs,
                                             allow_sparse_grad=sparse)

    block_mx = sockeye.encoder.Embedding(config, None, C.DTYPE_FP32)
    block_mx.initialize()
    block_pt = sockeye.encoder_pt.PyTorchEmbedding(config, None)
    block_pt.weights_from_mxnet_block(block_mx)

    batch, seq_len, num_factors = 4, 10, len(factor_configs) + 1 if factor_configs is not None else 1
    # data_mx does not take into account different vocab sizes for factors
    data_mx = np.random.randint(0, vocab_size, (batch, seq_len, num_factors))
    data_pt = pt.as_tensor(data_mx.asnumpy())
    vl_mx = np.ones((1,))  # not used
    vl_pt = pt.as_tensor(vl_mx.asnumpy())

    r_mx, _ = block_mx(data_mx, vl_mx)
    r_pt = block_pt(data_pt)

    r_mx = r_mx.asnumpy()
    r_pt = r_pt.detach().numpy()

    assert np.allclose(r_mx, r_pt)

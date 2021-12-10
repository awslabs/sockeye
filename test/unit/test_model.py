# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
from tempfile import TemporaryDirectory

import pytest
import torch as pt

import sockeye.constants as C
import sockeye.data_io_pt
import sockeye.model_pt
import sockeye.transformer_pt


def test_mx_pt_eq_sockeye_model():
    pytest.importorskip('mxnet')
    from mxnet import np
    import sockeye.transformer
    import sockeye.encoder
    import sockeye.model

    # model setup
    source_vocab_size = target_vocab_size = 32000
    num_embed_source = num_embed_target = model_size = 512
    max_seq_len_source = max_seq_len_target = 100
    num_source_factors = 1
    num_target_factors = 1
    num_layers = 4
    weight_tying = False
    batch_size = 4
    topk_size = 200
    config_encoder = sockeye.transformer.TransformerConfig(model_size=model_size,
                                                           attention_heads=8,
                                                           feed_forward_num_hidden=256,
                                                           act_type='relu',
                                                           num_layers=num_layers,
                                                           dropout_attention=0,
                                                           dropout_act=0,
                                                           dropout_prepost=0,
                                                           positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
                                                           preprocess_sequence='n',
                                                           postprocess_sequence='r',
                                                           max_seq_len_source=max_seq_len_source,
                                                           max_seq_len_target=max_seq_len_target,
                                                           use_lhuc=False)
    config_encoder_pt = sockeye.transformer_pt.TransformerConfig(model_size=model_size,
                                                                 attention_heads=8,
                                                                 feed_forward_num_hidden=256,
                                                                 act_type='relu',
                                                                 num_layers=num_layers,
                                                                 dropout_attention=0,
                                                                 dropout_act=0,
                                                                 dropout_prepost=0,
                                                                 positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
                                                                 preprocess_sequence='n',
                                                                 postprocess_sequence='r',
                                                                 max_seq_len_source=max_seq_len_source,
                                                                 max_seq_len_target=max_seq_len_target,
                                                                 use_lhuc=False)
    config_decoder = sockeye.transformer.TransformerConfig(model_size=model_size,
                                                           attention_heads=8,
                                                           feed_forward_num_hidden=256,
                                                           act_type='relu',
                                                           num_layers=num_layers,
                                                           dropout_attention=0,
                                                           dropout_act=0,
                                                           dropout_prepost=0,
                                                           positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
                                                           preprocess_sequence='n',
                                                           postprocess_sequence='r',
                                                           max_seq_len_source=max_seq_len_source,
                                                           max_seq_len_target=max_seq_len_target,
                                                           depth_key_value=model_size,
                                                           use_lhuc=False)
    config_decoder_pt = sockeye.transformer_pt.TransformerConfig(model_size=model_size,
                                                                 attention_heads=8,
                                                                 feed_forward_num_hidden=256,
                                                                 act_type='relu',
                                                                 num_layers=num_layers,
                                                                 dropout_attention=0,
                                                                 dropout_act=0,
                                                                 dropout_prepost=0,
                                                                 positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
                                                                 preprocess_sequence='n',
                                                                 postprocess_sequence='r',
                                                                 max_seq_len_source=max_seq_len_source,
                                                                 max_seq_len_target=max_seq_len_target,
                                                                 depth_key_value=model_size,
                                                                 use_lhuc=False)
    config_embed_source = sockeye.encoder.EmbeddingConfig(vocab_size=source_vocab_size,
                                                          num_embed=num_embed_source,
                                                          dropout=0,
                                                          factor_configs=None,
                                                          allow_sparse_grad=False)
    config_embed_target = sockeye.encoder.EmbeddingConfig(vocab_size=target_vocab_size,
                                                          num_embed=num_embed_target,
                                                          dropout=0,
                                                          factor_configs=None,
                                                          allow_sparse_grad=False)
    data_statistics = sockeye.data_io_pt.DataStatistics(num_sents=0,
                                                        num_discarded=0,
                                                        num_tokens_source=0,
                                                        num_tokens_target=0,
                                                        num_unks_source=0,
                                                        num_unks_target=0,
                                                        max_observed_len_source=100,
                                                        max_observed_len_target=100,
                                                        size_vocab_source=source_vocab_size,
                                                        size_vocab_target=target_vocab_size,
                                                        length_ratio_mean=1.0,
                                                        length_ratio_std=0.001,
                                                        buckets=[],
                                                        num_sents_per_bucket=[],
                                                        average_len_target_per_bucket=[],
                                                        length_ratio_stats_per_bucket=None)
    data_config = sockeye.data_io_pt.DataConfig(data_statistics=data_statistics,
                                                max_seq_len_source=max_seq_len_source,
                                                max_seq_len_target=max_seq_len_target,
                                                num_source_factors=num_source_factors,
                                                num_target_factors=num_target_factors)
    config_length_task = None
    model_config = sockeye.model.ModelConfig(config_data=data_config,
                                             vocab_source_size=source_vocab_size,
                                             vocab_target_size=target_vocab_size,
                                             config_embed_source=config_embed_source,
                                             config_embed_target=config_embed_target,
                                             config_encoder=config_encoder,
                                             config_decoder=config_decoder,
                                             config_length_task=config_length_task,
                                             weight_tying_type=C.WEIGHT_TYING_NONE,
                                             lhuc=False,
                                             dtype=C.DTYPE_FP32)
    model_config_pt = sockeye.model.ModelConfig(config_data=data_config,
                                                vocab_source_size=source_vocab_size,
                                                vocab_target_size=target_vocab_size,
                                                config_embed_source=config_embed_source,
                                                config_embed_target=config_embed_target,
                                                config_encoder=config_encoder_pt,
                                                config_decoder=config_decoder_pt,
                                                config_length_task=config_length_task,
                                                weight_tying_type=C.WEIGHT_TYING_NONE,
                                                lhuc=False,
                                                dtype=C.DTYPE_FP32)


    # inputs
    source_inputs_mx = np.random.randint(0, max_seq_len_source, (batch_size, max_seq_len_source, num_source_factors))
    source_input_lengths_mx = np.random.randint(0, max_seq_len_source, (batch_size,))
    target_inputs_mx = np.random.randint(0, max_seq_len_target, (batch_size, max_seq_len_target, num_source_factors))
    target_input_lengths_mx = np.random.randint(0, max_seq_len_target, (batch_size,))
    source_inputs_pt = pt.tensor(source_inputs_mx.asnumpy())
    source_input_lengths_pt = pt.tensor(source_input_lengths_mx.asnumpy())
    target_inputs_pt = pt.tensor(target_inputs_mx.asnumpy())
    target_input_lengths_pt = pt.tensor(target_input_lengths_mx.asnumpy())
    step_inputs_mx = np.random.randint(0, target_vocab_size, (batch_size, num_target_factors))
    vocab_slice_ids_mx = np.random.randint(0, target_vocab_size, (topk_size,))
    step_inputs_pt = pt.tensor(step_inputs_mx.asnumpy())
    vocab_slice_ids_pt = pt.tensor(vocab_slice_ids_mx.asnumpy())

    b_mx = sockeye.model.SockeyeModel(model_config, inference_only=False, mc_dropout=False,
                                      forward_pass_cache_size=0)
    b_mx.initialize()

    b_pt = sockeye.model_pt.PyTorchSockeyeModel(model_config_pt, inference_only=False, mc_dropout=False,
                                                forward_pass_cache_size=0)

    assert b_mx.state_structure() == b_pt.state_structure()

    # test forward()
    # first run mx block to complete deferred initialization
    forward_dict_mx = b_mx(source_inputs_mx, source_input_lengths_mx, target_inputs_mx, target_input_lengths_mx)
    # get weights from mx into pt
    b_pt.weights_from_mxnet_block(b_mx)
    forward_dict_pt = b_pt(source_inputs_pt, source_input_lengths_pt, target_inputs_pt, target_input_lengths_pt)

    assert forward_dict_mx.keys() == forward_dict_pt.keys()
    logits_mx = forward_dict_mx[C.LOGITS_NAME].asnumpy()
    logits_pt = forward_dict_pt[C.LOGITS_NAME].detach().numpy()

    assert np.allclose(logits_mx, logits_pt, atol=1e-05)

    # test encode()
    source_encoded_mx, source_encoded_length_mx = b_mx.encode(source_inputs_mx, source_input_lengths_mx)
    source_encoded_pt, source_encoded_length_pt = b_pt.encode(source_inputs_pt, source_input_lengths_pt)
    assert np.allclose(source_encoded_mx.asnumpy(), source_encoded_pt.detach().numpy(), atol=1e-05)
    assert np.allclose(source_encoded_length_mx.asnumpy(), source_encoded_length_pt.detach().numpy(), atol=1e-05)

    # test encode_and_initialize()
    init_states_mx, pred_out_length_mx = b_mx.encode_and_initialize(source_inputs_mx, source_input_lengths_mx,
                                                                    constant_length_ratio=0.0)
    init_states_pt, pred_out_length_pt = b_pt.encode_and_initialize(source_inputs_pt, source_input_lengths_pt,
                                                                    constant_length_ratio=0.0)
    if config_length_task is None:
        assert np.allclose(pred_out_length_mx.asnumpy(), np.zeros_like(source_input_lengths_mx).asnumpy())
        assert np.allclose(pred_out_length_pt.detach().numpy(), pt.zeros_like(source_input_lengths_pt).detach().numpy())
    else:
        assert pred_out_length_mx.asnumpy() == pred_out_length_pt.detach().numpy()

    assert len(init_states_mx) == len(init_states_pt)
    state_structure = b_pt.decoder.state_structure()
    for s_mx, s_pt, structure in zip(init_states_mx, init_states_pt, state_structure):
        if structure != C.MASK_STATE:  # MASK state is new in Pytorch and not equivalent
            assert np.allclose(s_mx.asnumpy(), s_pt.detach().numpy(), atol=1e-05)

    # test decode_step()
    b_pt.eval()
    states_mx = init_states_mx
    states_pt = init_states_pt
    step_output_mx, states_mx, factor_outputs_mx = b_mx.decode_step(step_inputs_mx, states_mx,
                                                                    vocab_slice_ids=vocab_slice_ids_mx)
    step_output_pt, states_pt, factor_outputs_pt = b_pt.decode_step(step_inputs_pt, states_pt,
                                                                    vocab_slice_ids=vocab_slice_ids_pt)

    assert np.allclose(step_output_mx.asnumpy(), step_output_pt.detach().numpy(), atol=1e-05)
    assert step_output_mx.asnumpy().shape == step_output_pt.detach().numpy().shape == (batch_size, topk_size)
    assert len(factor_outputs_mx) == len(factor_outputs_pt)
    # TODO assert factor outputs equality
    assert len(states_mx) == len(states_pt)
    for s_mx, s_pt, structure in zip(states_mx, states_pt, state_structure):
        if structure != C.MASK_STATE:  # MASK state is new in Pytorch and not equivalent
            assert np.allclose(s_mx.asnumpy(), s_pt.detach().numpy(), atol=1e-05)

    from pprint import pprint
    pprint(b_mx.collect_params())
    for param_tensor in b_pt.state_dict():
        print(param_tensor, "\t", b_pt.state_dict()[param_tensor].size())

    # save & load parameters
    with TemporaryDirectory() as work_dir:
        fname = os.path.join(work_dir, 'params.pt')
        b_pt.save_parameters(fname)
        b_pt.load_parameters(fname)

    forward_dict_pt = b_pt(source_inputs_pt, source_input_lengths_pt, target_inputs_pt, target_input_lengths_pt)
    assert forward_dict_mx.keys() == forward_dict_pt.keys()
    logits_mx = forward_dict_mx[C.LOGITS_NAME].asnumpy()
    logits_pt = forward_dict_pt[C.LOGITS_NAME].detach().numpy()
    assert np.allclose(logits_mx, logits_pt, atol=1e-05)

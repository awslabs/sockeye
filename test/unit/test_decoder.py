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

import mxnet as mx
import numpy as onp
import pytest
import torch as pt
from mxnet import np

import sockeye.constants as C
import sockeye.decoder
import sockeye.decoder_pt
import sockeye.transformer


@pytest.mark.parametrize('lhuc', [
    (False,),
    (True,)
])
def test_get_decoder(lhuc):
    config = sockeye.transformer.TransformerConfig(
        model_size=20,
        attention_heads=10,
        feed_forward_num_hidden=30,
        act_type='test_act',
        num_layers=50,
        dropout_attention=0.5,
        dropout_act=0.6,
        dropout_prepost=0.1,
        positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
        preprocess_sequence=C.FIXED_POSITIONAL_EMBEDDING,
        postprocess_sequence='test_post_seq',
        max_seq_len_source=60,
        max_seq_len_target=70,
        use_lhuc=lhuc)
    decoder = sockeye.decoder.get_decoder(config, inference_only=False)

    assert type(decoder) == sockeye.decoder.TransformerDecoder


@pytest.mark.parametrize("inference_only", [False, True])
def test_mx_pt_eq_transformer_decoder(inference_only):
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
                                                   positional_embedding_type=C.FIXED_POSITIONAL_EMBEDDING,
                                                   preprocess_sequence='n',
                                                   postprocess_sequence='r',
                                                   max_seq_len_source=50,
                                                   max_seq_len_target=60,
                                                   use_lhuc=False)
    batch = 12
    encoder_seq_len = 45
    decoder_seq_len = 39 if not inference_only else 1
    encoder_outputs_mx = np.random.uniform(0, 1, (batch, encoder_seq_len, config.model_size))
    encoder_outputs_pt = pt.as_tensor(encoder_outputs_mx.asnumpy())
    encoder_valid_length_mx = np.random.randint(0, encoder_seq_len, (batch,))
    encoder_valid_length_pt = pt.as_tensor(encoder_valid_length_mx.asnumpy())
    inputs_mx = np.random.uniform(0, 1, (batch, decoder_seq_len, config.model_size))
    inputs_pt = pt.as_tensor(inputs_mx.asnumpy())

    # mx
    decoder_mx = sockeye.decoder.get_decoder(config, inference_only=inference_only, dtype=C.DTYPE_FP32)
    decoder_mx.initialize()
    init_states_mx = decoder_mx.init_state_from_encoder(encoder_outputs_mx, encoder_valid_length_mx)
    output_mx, new_states_mx = decoder_mx(inputs_mx, init_states_mx)
    if inference_only:  # do a second decoder step
        output_mx, new_states_mx = decoder_mx(output_mx, new_states_mx)

    # pt
    decoder_pt = sockeye.decoder_pt.pytorch_get_decoder(config, inference_only=inference_only, dtype=C.DTYPE_FP32)
    decoder_pt.weights_from_mxnet_block(decoder_mx)
    init_states_pt = decoder_pt.init_state_from_encoder(encoder_outputs_pt, encoder_valid_length_pt)
    output_pt, new_states_pt = decoder_pt(inputs_pt, init_states_pt)
    if inference_only:  # do a second decoder step
        output_pt, new_states_pt = decoder_pt(output_pt, new_states_pt)

    assert len(init_states_mx) == len(init_states_pt)
    for s_mx, s_pt in zip(init_states_mx, init_states_pt):
        assert np.allclose(s_mx.asnumpy(), s_pt.detach().numpy(), atol=1e-05)

    output_mx, output_mx.asnumpy()
    output_pt = output_pt.detach().numpy()

    print("Max deviation:", onp.abs(output_mx - output_pt).max())
    assert np.allclose(output_mx, output_pt, atol=1e-05)

    assert len(new_states_mx) == len(new_states_pt)
    for i, (s_mx, s_pt) in enumerate(zip(new_states_mx, new_states_pt)):
        assert np.allclose(s_mx.asnumpy(), s_pt.detach().numpy(), atol=1e-05)


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
import sockeye.decoder
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
        lhuc=lhuc)
    decoder = sockeye.decoder.get_decoder(config, inference_only=False, prefix='test_')

    assert type(decoder) == sockeye.decoder.TransformerDecoder
    assert decoder.prefix == 'test_' + C.TRANSFORMER_DECODER_PREFIX

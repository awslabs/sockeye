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

from typing import Optional

import torch as pt

from . import constants as C


class NeuralVocabSelection(pt.nn.Module):
    def __init__(self,
                 model_size: int,
                 vocab_target_size: int,
                 model_type: str = C.NVS_TYPE_LOGIT_MAX,
                 dtype: Optional[pt.dtype] = None) -> None:
        super().__init__()
        self.vocab_target_size = vocab_target_size
        self.model_type = model_type

        self.project_vocab = pt.nn.Linear(model_size, vocab_target_size, bias=True, dtype=dtype)

    def forward(self, source_encoded: pt.Tensor, source_length: pt.Tensor, att_mask: pt.Tensor):
        if self.model_type == C.NVS_TYPE_LOGIT_MAX:
            # ============
            # logit max:
            # ============
            bow_pred = self.project_vocab(source_encoded)
            bow_pred = bow_pred.masked_fill(att_mask.unsqueeze(2), -pt.inf)
            bow_pred, _ = pt.max(bow_pred, dim=1)
        elif C.NVS_TYPE_EOS:
            # ============
            # EOS based:
            # ============
            batch_size, max_len, _ = source_encoded.size()
            source_encoded = source_encoded[pt.arange(0, batch_size, dtype=pt.long), (source_length-1).long()]
            bow_pred = self.project_vocab(source_encoded)
        else:
            raise ValueError("Unknown neural vocabulary selection type.")

        return bow_pred

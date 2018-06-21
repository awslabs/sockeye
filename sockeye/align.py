# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as np
from . import constants as C

class Aligner:

    # todo: this is just a toy implementation at the moment

    def __init__(self, bos_id, eos_id, trg_vocab=None):
        self.vocab = trg_vocab
        self.banned_ids = [bos_id, eos_id]

    def get_copy_alignment_alt(self, src, trg):

        # TODO: we need to make these configurable eventually
        PROXIMITY = 10
        MIN_WORD_LENGHT = 2
        BANNED_WORDS = [C.EOS_SYMBOL, C.BOS_SYMBOL]
        default_index = -1
        copied_pos_list = np.ones(len(trg)) * -1

        for i, current_word in enumerate(trg):
            # TODO: add min word length by passing the vocabulary of the source to the loader
            is_long_enough = True

            if is_long_enough and not current_word in BANNED_WORDS:
                offset = max(0, i - PROXIMITY)
                prox_words = src[offset:min(i + PROXIMITY, len(trg))]
                if current_word in prox_words:
                    positions_matched = []
                    for position, item in enumerate(prox_words):
                        if item == current_word:
                            positions_matched.append(position + offset)
                    for p in positions_matched:
                        if p + 1 not in copied_pos_list:
                            copied_pos_list[i] = p + 1
                            break
        return copied_pos_list

    def get_copy_alignment(self, src, trg):

        # TODO: we need to make these configurable eventually
        PROXIMITY = 10
        MIN_WORD_LENGTH = 2
        default_index = -1
        copied_pos_list = np.ones(len(trg)) * -1
        oov_index = len(self.vocab)

        for current_idx, current_word in enumerate(trg):
            # TODO: add min word length by passing the vocabulary of the source to the loader

            if current_word not in self.banned_ids and current_word not in self.vocab:
                offset = max(0, current_idx - PROXIMITY)
                prox_words = src[offset : min(current_idx + PROXIMITY, len(trg))]
                matched_pos = prox_words.index(current_word) if current_word in prox_words else -1

                if matched_pos >= 0:# and matched_pos + 1 not in copied_pos_list:
                    #copied_pos_list[current_idx] = matched_pos + 1
                    copied_pos_list[current_idx] = oov_index
                    oov_index += 1

        return copied_pos_list

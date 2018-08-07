# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import sys
from . import constants as C
from . import vocab

from typing import List

logger = logging.getLogger(__name__)

class Aligner:
    """
    Alignment used for pointer networks. Rewrites the label stream with indexes past the maximum target vocabulary item,
    which are interpreted as pointing into the source sentence.

    :param source_vocab: The source vocabulary.
    :param target_vocab: The target vocabulary.
    :param window_size: The maximum window size a target word can point to in the source sentence.
    :param min_word_length: The shortest word that can be considered as a pointer.
    """

    def __init__(self,
                 source_vocab: vocab.Vocab,
                 target_vocab: vocab.Vocab,
                 window_size: int = 20,
                 min_word_length: int = 2):
        self.source_vocab = vocab.reverse_vocab(source_vocab)
        self.target_vocab = vocab.reverse_vocab(target_vocab)
        self.vocab_offset = len(target_vocab)

        self.window_size = window_size
        self.min_word_length = min_word_length
        self.banned_words = [C.EOS_SYMBOL, C.BOS_SYMBOL, C.PAD_SYMBOL]

        self.num_pointed = 0
        self.num_total = 0

        logger.info("Pointer networks: window_size=%d min_word_len=%d", self.window_size, self.min_word_length)

    def get_labels(self,
                   source: List[str],
                   target: List[str],
                   label: List[int]):
        """
        Takes the source, target, and current set of labels (shifted target)
        and produces a new set of labels, with some words possibly pointing.

        The source includes the </s> token.
        The target includes <s> but no </s>.
        The labels is the same as target, but dropping <s> and adding </s>.

        :return: The new label with pointed words set to len(target_vocab) + source_index.
        """

        new_label = [x for x in label]

        source = [self.source_vocab[x] for x in source]
        target = [self.target_vocab[x] for x in target]

        already_pointed = {}
        self.num_total += len(source)
        for i, current_word in enumerate(target[1:]):
            if current_word == C.PAD_SYMBOL:
                continue

            is_long_enough = len(current_word) >= self.min_word_length
            if not is_long_enough or current_word in self.banned_words:
                continue
            window_start = max(0, i - self.window_size // 2)
            window_stop = min(i + self.window_size // 2, len(source))
            prox_words = source[window_start:window_stop]
            # print(sentno, i, 'IS', current_word, 'IN', prox_words, current_word in prox_words)
            if current_word in prox_words:
                for source_pos, source_word in enumerate(prox_words, window_start):
                    if source_word == current_word and source_pos not in already_pointed:
                        new_label[i] = source_pos + self.vocab_offset
                        already_pointed[source_pos] = True
                        self.num_pointed += 1
                        break

        # my_labels = [self.target_vocab[x] if x < self.vocab_offset else '[{}/{}]'.format(source[x - self.vocab_offset], x - self.vocab_offset) for x in new_label]
        # print('SENTENCE', source, '\n', target, '\n', my_labels)

        return new_label

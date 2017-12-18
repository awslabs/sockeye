#!/usr/bin/env python

import json
import sys

words = {}

words["<blank>"]=1
words["<unk>"]=2
words["<s>"]=3
words["</s>"]=4

for l in sys.stdin:
    for w in l.strip().split():
        if not w in words:
            words[w] = len(words) + 1

for k in sorted(words.keys(), key=lambda k: words[k]):
    sys.stdout.write("{} {}\n".format(k, words[k]))

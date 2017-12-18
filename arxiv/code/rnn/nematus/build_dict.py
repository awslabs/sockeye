#!/usr/bin/env python

import json
import sys

words = {}
nWords = 0
for l in sys.stdin:
    for w in l.strip().split():
        if not words.has_key(w):
            words[w] = nWords
            nWords += 1
sys.stdout.write(json.dumps(words))

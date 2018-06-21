#!/usr/bin/env python3

import os
import random

random.seed(12)

num_samples = 100000
num_dev = 1000
min_seq_len = 10
max_seq_len = 30
vocab_size = 10

samples = set()
for i in range(0, num_samples):
    seq_len = random.randint(min_seq_len, max_seq_len)
    samples.add(" ".join(str(random.randint(0, vocab_size)) for j in range(0, seq_len)))

samples = list(samples)

train_samples = samples[:num_samples-num_dev]
dev_samples = samples[num_samples-num_dev:]

if not os.path.exists('data'):
    os.mkdir('data')
with open("data/train.source", "w") as source, open("data/train.target", "w") as target:
    for sample in train_samples:
        source.write(sample + "\n")
        target.write(sample + "\n")

with open("data/dev.source", "w") as source, open("data/dev.target", "w") as target:
    for sample in dev_samples:
        source.write(sample + "\n")
        target.write(sample + "\n")


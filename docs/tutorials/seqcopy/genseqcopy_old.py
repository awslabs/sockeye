#!/usr/bin/env python3

import os
import random

random.seed(12)

num_samples = 100000
num_dev = 1000
min_seq_len = 10
max_seq_len = 30
vocab_size = 10


def to_str(l):
    for x in l:
        yield str(x)


samples = set()
for i in range(0, num_samples):
    seq_len = random.randint(min_seq_len, max_seq_len)
    int_seq = [random.randint(0, vocab_size) for j in range(0, seq_len)]
    source_factors = ['l' if x < vocab_size/2 else 'h' for x in int_seq]  # low/high source factor
    target_factors = ['e' if x % 2 == 0 else 'o' for x in int_seq]  # odd/even target factor
    sample = (" ".join(to_str(int_seq)), " ".join(source_factors), " ".join(target_factors))
    samples.add(sample)
samples = list(samples)
train_samples = samples[:num_samples-num_dev]
dev_samples = samples[num_samples-num_dev:]


def write_data(samples, prefix):
    with open("data/%s.source" % prefix, "w") as source, \
            open("data/%s.source.factor" % prefix, "w") as source_factor, \
            open("data/%s.target" % prefix, "w") as target, \
            open("data/%s.target.factor" % prefix, "w") as target_factor:
        for s, sf, tf in samples:
            print(s, file=source)
            print(sf, file=source_factor)
            print(s, file=target)
            print(tf, file=target_factor)


if not os.path.exists('data'):
    os.mkdir('data')
write_data(train_samples, 'train')
write_data(dev_samples, 'dev')

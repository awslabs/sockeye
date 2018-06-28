#!/usr/bin/env python3

import argparse
import os
import random

random.seed(12)

num_samples = 100000
num_dev = 1000
min_seq_len = 10
max_seq_len = 30
vocab_size = 10

parser = argparse.ArgumentParser()
parser.add_argument('--copy-prob', '-c', type=float, default=0.9, help="probability of copying IDs that are copied")
parser.add_argument('--copy-max', '-n', type=int, default=0, help="All numbers less than or equal to this are copied with --copy-prob")
parser.add_argument('--output-dir', '-o', type=str, default='data', help="output directory to write to")
args = parser.parse_args()

print('copying IDs 0..%d with prob %f, writing to %s' % (args.copy_max, args.copy_prob, args.output_dir))

samples = set()
for i in range(0, num_samples):
    seq_len = random.randint(min_seq_len, max_seq_len)
    samples.add(" ".join(str(random.randint(0, vocab_size)) for j in range(0, seq_len)))

samples = list(samples)

train_samples = samples[:num_samples-num_dev]
dev_samples = samples[num_samples-num_dev:]

def int_or_char(sequence: str):
    return ' '.join([i if int(i) <= args.copy_max and random.random() <= args.copy_prob else chr(int(i) + 65) for i in sequence.split()])

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
with open(os.path.join(args.output_dir, 'train.source'), "w") as source, open(os.path.join(args.output_dir, 'train.target'), "w") as target:
    for sample in train_samples:
        print(sample, file=source)
        print(int_or_char(sample), file=target)

with open(os.path.join(args.output_dir, 'dev.source'), "w") as source, open(os.path.join(args.output_dir, 'dev.target'), "w") as target:
    for sample in dev_samples:
        print(sample, file=source)
        print(int_or_char(sample), file=target)

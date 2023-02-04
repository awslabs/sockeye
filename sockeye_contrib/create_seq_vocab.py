"""
Builds a Sockeye-compatible vocab where sequential vocab items are consecutive numbers.
This is for use with numeric target factors and fixed sinusoidal embeddings.
"""

import json
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--min-val", type=int, default=0,
                    help="Minimum value of the numeric vocab items")
parser.add_argument("--max-val", type=int, default=1023,
                    help="Maximum value of the numeric vocab items")
parser.add_argument("--pad-vocab-to-multiple-of",
                    type=int, default=8,
                    help="Pad vocabulary to a multiple of this integer")
parser.add_argument("--output", default="seq_vocab.json",
                    help="Path to write vocab to")

args = parser.parse_args()

PAD_FORMAT = "<pad%d>"
VOCAB_SYMBOLS = ["<pad>", "<unk>", "<s>", "</s>"]

vocab = dict()

ctr = 0
# Insert special vocab items
for item in VOCAB_SYMBOLS:
    vocab[item] = ctr
    ctr += 1

# Insert numeric values
for i in range(args.min_val, args.max_val + 1):
    vocab[str(i)] = ctr
    ctr += 1

# Pad to a multiple of args.pad_vocab_to_multiple_of
if len(vocab.keys()) % args.pad_vocab_to_multiple_of != 0:
    num_pad = args.pad_vocab_to_multiple_of - (len(vocab.keys()) % args.pad_vocab_to_multiple_of)
    for _ in range(num_pad):
        vocab[PAD_FORMAT % ctr] = ctr
        ctr += 1

# Write to file
with open(args.output, 'w') as f_vocab:
    json.dump(vocab, f_vocab, indent=4)


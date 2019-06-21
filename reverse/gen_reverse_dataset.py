# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
CLI for generating datasets for testing reverse sequence copying
"""

import argparse
# import os
# import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="the total number of samples")
    args = parser.parse_args()

    print(args.num_samples)

main()

# random.seed(12)

# num_samples = 100000
# num_dev = 1000
# min_seq_len = 10
# max_seq_len = 30
# vocab_size = 10

# samples = set()
# for i in range(0, num_samples):
#     seq_len = random.randint(min_seq_len, max_seq_len)
#     samples.add(" ".join(str(random.randint(0, vocab_size)) for j in range(0, seq_len)))

# samples = list(samples)

# train_samples = samples[:num_samples-num_dev]
# dev_samples = samples[num_samples-num_dev:]

# if not os.path.exists('data'):
#     os.mkdir('data')
# with open("data/train.source", "w") as source, open("data/train.target", "w") as target:
#     for sample in train_samples:
#         source.write(sample + "\n")
#         target.write(sample + "\n")

# with open("data/dev.source", "w") as source, open("data/dev.target", "w") as target:
#     for sample in dev_samples:
#         source.write(sample + "\n")
#         target.write(sample + "\n")


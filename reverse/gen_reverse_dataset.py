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
CLI for generating datasets for testing reverse sequence copying.
"""

import argparse
import os
import random

def parse_arguments():
    """
    Parses the command line arguments.
    
    :return: An object containing all arguments are fields.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-nt", "--num-train", type=int, default=10000, help="the total number of training samples")
    parser.add_argument("-nv", "--num-val", type=int, default=100, help="the total number of validaton samples")
    parser.add_argument("-sm", "--min-sequence-length", type=int, default=5, help="the minimum length of sequence")
    parser.add_argument("-sx", "--max-sequence-length", type=int, default=15, help="the maximum length of sequence")
    parser.add_argument("-s", "--seed", type=int, default=42, help="the random seed for the data generation")
    parser.add_argument("output", default="data", help="the name of the output directory")
    args = parser.parse_args()

    print("Number of training samples: ", args.num_train)
    print("Number of validation samples: ", args.num_val)
    print("Minimum length of sequences: ", args.min_sequence_length)
    print("Number of training samples: ", args.max_sequence_length)
    print("Random seed: ", args.seed)
    print("Output directory: ", args.output)
    return(args)

def main():
    """
    Main entry point to the app. 
    """

    args = parse_arguments()
    random.seed(args.seed)

    samples = set()
    for i in range(0, args.num_train+args.num_val):
        seq_len = random.randint(args.min_sequence_length, args.max_sequence_length)
        samples.add(" ".join(str(random.randint(0, 9)) for j in range(0, seq_len)))

    samples = list(samples)

    train_samples = samples[1:args.num_train]
    val_samples = samples[(args.num_train+1):]

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    with open(args.output+"/train.source", "w") as source, open(args.output+"/train.target", "w") as target:
        for sample in train_samples:
            source.write(sample + "\n")
            target.write(sample[::-1] + "\n")

    with open(args.output+"/validation.source", "w") as source, open(args.output+"/validation.target", "w") as target:
        for sample in val_samples:
            source.write(sample + "\n")
            target.write(sample[::-1] + "\n")

main()



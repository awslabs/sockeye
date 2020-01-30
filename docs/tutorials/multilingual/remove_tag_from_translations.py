#! /usr/bin/python3

import sys
import argparse
import logging

from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print tag statistics to STDERR after removing tags.")

    args = parser.parse_args()

    return args

def is_tag(token):
    if token[0] == "<" and token[-1] == ">":
        if len(token) == 5:
            return True


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    stat_dict = defaultdict(int)

    for line in sys.stdin:

        tokens = line.strip().split(" ")

        if is_tag(tokens[0]):
            tag = tokens[0]
            tokens.pop(0)
            stat_dict[tag] += 1
        else:
            stat_dict["NO_START_TAG"] += 1

        keep_tokens = []

        for token in tokens:
            if is_tag(token):
                stat_dict["TAG_WITHIN_SENTENCE"] += 1
                continue
            else:
                keep_tokens.append(token)

        line = " ".join(keep_tokens)

        print(line)

    if args.verbose:
        logging.debug("Stats of tags encountered:")
        logging.debug(str(stat_dict))

if __name__ == '__main__':
    main()

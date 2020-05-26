#! /usr/bin/python3

import sys
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", type=str, help="Special tag to indicate language", required=True)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    num_bad = 0

    for line in sys.stdin:

        tokens = line.strip().split(" ")

        if tokens[0][0] == "<" and tokens[0][-1] == ">":
            logging.warning("First token of sentence already seems to be a special language tag: '%s'." % tokens[0])
            num_bad += 1

        if tokens[0] == args.tag:
            logging.error("Sentence already has '%s' as first token. Do not run this script twice." % args.tag)
            sys.exit(1)
        else:
            tokens = [args.tag] + tokens

        line = " ".join(tokens)

        print(line)

    if num_bad > 0:
        logging.debug("Number of times sentences had a first token of the form '<...>': %d." % num_bad)

if __name__ == '__main__':
    main()

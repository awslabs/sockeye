#! /usr/bin/python3

import sys
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", type=str,
                        help="Special tags to indicate target language", required=True)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    num_first_not_tag = 0
    num_tags_within_sentence = 0

    for line in sys.stdin:

        tokens = line.strip().split(" ")

        if tokens[0] == args.tag:
            tokens.pop(0)
        else:
            num_first_not_tag += 1

        keep_tokens = []

        for token in tokens:
            if token == args.tag:
                num_tags_within_sentence += 1
                continue
            else:
                keep_tokens.append(token)

        line = " ".join(keep_tokens)

        print(line)

    logging.debug("First token not the special tag: %d times" % num_first_not_tag)
    logging.debug("Tags found in non-first position: %d times" % num_tags_within_sentence)

if __name__ == '__main__':
    main()

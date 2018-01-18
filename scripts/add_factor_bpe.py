#!/usr/bin/env python3

"""
Adds BIOE state-transition tags for BPE tokens.
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    state = 'O'
    for line in sys.stdin:
        tags = []
        for i, token in enumerate(line.rstrip().split()):
            if token.endswith('@@'):
                if state == "O" or state == "E":
                    state = "B"
                elif state == "B" or state == "I":
                    state = "I"
            else:
                if state == "B" or state == "I":
                    state = "E"
                else:
                    state = "O"
            tags.append(state)

        print(' '.join(tags))

if __name__ == '__main__':
    main()

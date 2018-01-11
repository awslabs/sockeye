#!/usr/bin/env python3

import argparse
import sys

def case(s):
    if s.islower():
        return 'l'
    elif s.isupper():
        return 'U'
    elif s.istitle():
        return 'C'
    else:
        return '-'

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for line in sys.stdin:
        print(' '.join([case(tok) for tok in line.rstrip().split()]))

if __name__ == '__main__':
    main()

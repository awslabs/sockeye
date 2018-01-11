#!/usr/bin/env python3

import argparse
import sys

def case(s):
    # print('case', s)
    if s.islower():
        return 'l'
    elif s.isupper():
        return 'U'
    elif s.istitle():
        return 'C'
    else:
        return '-'

def eat_and_tag(tokens, i, f=case):
    """
    Walks forward merging BPE tokens so the word function can be computed.
    """
    # print('eat_and_tag', i, tokens[i:])
    token = ''
    while tokens[i].endswith('@@'):
        token += tokens[i].replace('@@', '')
        i += 1
    token += tokens[i]
    return i + 1, f(token)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for line in sys.stdin:
        tokens = line.rstrip().split()
        tags = [] * len(tokens)
        i = 0
        while i < len(tokens):
            next_i, tag = eat_and_tag(tokens, i, case)
            tags[i:next_i] = [tag] * (next_i - i)
#            print('tags', i, next_i, tag)
            i = next_i

        print(' '.join(tags))

if __name__ == '__main__':
    main()

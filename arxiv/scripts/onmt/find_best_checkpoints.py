#!/usr/bin/env python3

import os
import re
import sys


def main():

    if len(sys.argv[1:]) != 1:
        print(f'Usage: {__file__} N <onmt/train.log')
        sys.exit(2)

    N = int(sys.argv[1])

    checkpoints = []
    current_valid_ppl = None
    for line in sys.stdin:
        m = re.search(r'Validation perplexity: ([.0-9]+)', line)
        if m:
            current_valid_ppl = float(m.group(1))
        m = re.search(r'Saving checkpoint ([\S]+)', line)
        if m:
            fname = m.group(1)
            if os.path.exists(fname):
                checkpoints.append((current_valid_ppl, fname))

    fnames = [fname for _, fname in sorted(checkpoints)[:N]]

    print(' '.join(fnames))

if __name__ == '__main__':
    main()

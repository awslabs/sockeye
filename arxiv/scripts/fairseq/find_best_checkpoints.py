#!/usr/bin/env python3

import re
import sys


def main():

    if len(sys.argv[1:]) != 1:
        print(f'Usage: {__file__} N <fairseq.big.log')
        sys.exit(2)

    N = int(sys.argv[1])

    checkpoints = []
    for line in sys.stdin:
        m = re.search(r'fairseq.checkpoint_utils \| Saved checkpoint (\S+) .+ score ([.0-9]+)', line)
        if m:
            checkpoints.append((float(m.group(2)), m.group(1)))

    fnames = [fname for _, fname in sorted(checkpoints)[:N]]

    print(' '.join(fnames))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

"""
Initializing Sockeye embedding weights with pretrained word representations.

Quick usage:

    python3 -m contrib.utils.init_embedding        \
            -e embed-in-src.npy embed-in-tgt.npy    \
            -i vocab-in-src.json vocab-in-tgt.json   \
            -o vocab-out-src.json vocab-out-tgt.json  \
            -n source_embed_weight target_embed_weight \
            -f params.init

Optional arguments:

    --embeddings, -e
        list of input embedding weights in .npy format
        shape=(vocab-in-size, embedding-size)

    --vocabularies-in, -i
        list of input vocabularies as token-index dictionaries in .json format

    --vocabularies-out, -o
        list of output vocabularies as token-index dictionaries in .json format
        They can be generated using sockeye.vocab before actual Sockeye training.

    --names, -n
        list of Sockeye parameter names for embedding weights
        Most common ones are source_embed_weight, target_embed_weight and source_target_embed_weight.

    Sizes of above 4 lists should be exactly the same - they are vertically aligned.

    --file, -f
        file to write initialized parameters

    --encoding, -c
        open input vocabularies with specified encoding (default: utf-8)
"""

import argparse
import logging
import sys
import json
import numpy as np
import mxnet as mx

def main():
    arg_parser = argparse.ArgumentParser(description='Quick usage: python3 -m contrib.utils.init_embedding \n'
                                                     '-e embed-in-src.npy embed-in-tgt.npy \n'
                                                     '-i vocab-in-src.json vocab-in-tgt.json \n'
                                                     '-o vocab-out-src.json vocab-out-tgt.json \n'
                                                     '-n source_embed_weight target_embed_weight \n'
                                                     '-f params.init')
    arg_parser.add_argument('--embeddings', '-e', required=True, nargs='+',
                            help='list of input embedding weights in .npy format')
    arg_parser.add_argument('--vocabularies-in', '-i', required=True, nargs='+',
                            help='list of input vocabularies as token-index dictionaries in .json format')
    arg_parser.add_argument('--vocabularies-out', '-o', required=True, nargs='+',
                            help='list of output vocabularies as token-index dictionaries in .json format')
    arg_parser.add_argument('--names', '-n', required=True, nargs='+',
                            help='list of Sockeye parameter names for embedding weights')
    arg_parser.add_argument('--file', '-f', required=True,
                            help='file to write initialized parameters')
    arg_parser.add_argument('--encoding', '-c', type=str, default='utf-8',
                            help='open input vocabularies with specified encoding (default: %(default)s)')
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[INFO]: %(message)s')

    if len(args.embeddings) != len(args.vocabularies_in) or \
       len(args.embeddings) != len(args.vocabularies_out) or \
       len(args.embeddings) != len(args.names):
           logging.error("Exactly the same number of 'input embedding weights', 'input vocabularies', "
                         "'output vocabularies' and 'Sockeye parameter names' should be provided.")
           sys.exit(1)

    params = {} # type: Dict[str, mx.nd.NDArray]
    for embed_file, vocab_in_file, vocab_out_file, name in zip(args.embeddings, args.vocabularies_in, \
                                                               args.vocabularies_out, args.names):
        logging.info('Loading input embedding weight: %s', embed_file)
        embed = np.load(embed_file)
        logging.info('Loading input/output vocabularies: %s %s', vocab_in_file, vocab_out_file)
        with open(vocab_in_file, encoding=args.encoding) as file:
            vocab_in = json.load(file)
        with open(vocab_out_file, encoding=args.encoding) as file:
            vocab_out = json.load(file)
        logging.info('Initializing parameter: %s', name)
        embed_init = np.random.normal(scale=np.std(embed), size=(len(vocab_out), embed.shape[1]))
        for token in vocab_out:
            if token in vocab_in:
                embed_init[vocab_out[token]] = embed[vocab_in[token]]
        params[name] = mx.nd.array(embed_init, dtype='float32')

    logging.info('Saving initialized parameters to %s', args.file)
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in params.items()}
    mx.nd.save(args.file, save_dict)


if __name__ == '__main__':
    main()

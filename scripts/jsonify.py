#!/usr/bin/env python3

import argparse
import random
import json
import sys
import apply_bpe
from sacrebleu import extract_ngrams

parser = argparse.ArgumentParser()
parser.add_argument('--bpe', default='model/bpe.model')
parser.add_argument('--first-word', '-f', action='store_true', default=False)
parser.add_argument('--last-word', '-l', action='store_true', default=False)
parser.add_argument('--random-word', '-r', type=int, default=0)
parser.add_argument('--constraints', '-c', nargs='*', type=str, default=[])
parser.add_argument('--phrase-len', '-p', type=int, default=1)
parser.add_argument('--dictionary', '-d', type=str, default=None, help='dictionary file')
parser.add_argument('--ner', action='store_true', default=False, help='input is NER-tagged')
parser.add_argument('--add-sos', default=False, action='store_true', help='add <s> token')
parser.add_argument('--add-eos', default=False, action='store_true', help='add </s> token')
parser.add_argument('--all', '-a', action='store_true', default=False)
args = parser.parse_args()

bpe = apply_bpe.BPE(open(args.bpe))

termsdict = {}
if args.dictionary is not None:
    with open(args.dictionary) as fh:
        for i, line in enumerate(fh):
            source, target = line.rstrip().split('\t')
            termsdict[source] = target

def get_phrase(words, index, length):
    assert(index < len(words) - length + 1)
    phr = ' '.join(words[index:index+length])
    for i in range(length):
        words.pop(index)

    return phr

for line in sys.stdin:
    constraints = []

    def add_constraint(constraint):
        phrase = ''
        if args.add_sos:
            phrase += '<s> '
        phrase += bpe.segment(constraint)
        if args.add_eos:
            phrase += ' </s>'

        constraints.append(phrase)

    if '\t' in line:
        source, target = line.rstrip().split('\t')

        words = target.split()

        try:
            if args.all:
                constraints.append('<s> ' + bpe.segment(target) + ' </s>')

            else:
                if args.first_word:
                    add_constraint(get_phrase(words, 0, args.phrase_len))
                if args.last_word:
                    add_constraint(get_phrase(words, len(words) - args.phrase_len, args.phrase_len))
                if args.random_word > 0:
                    for i in range(min(len(words), args.random_word)):
                        choice = get_phrase(words, random.randint(0, len(words) - args.phrase_len), args.phrase_len)
                        add_constraint(choice)

        except:
            pass
    else:
        source = line.rstrip()

        if args.ner:
            tagged_source = source
            pairs = [(x[0:x.rindex('/')], x[x.rindex('/')+1:]) for x in tagged_source.split()]
            constraint = []
            for word, tag in pairs:
                if tag == 'PERSON':
                    constraint.append(word)
                else:
                    if len(constraint) > 0:
                        phrase = ' '.join(constraint)
                        if phrase in termsdict:
                            add_constraint(termsdict[phrase])
                        constraint = []
            source = ' '.join([x[0] for x in pairs])

    for phrase in args.constraints:
        add_constraint(phrase)

    if not args.ner and args.dictionary is not None:
        ngrams = sorted(extract_ngrams(source, 4), key=lambda x: len(x.split()), reverse=True)
        for ngram in ngrams:
            if ngram in termsdict:
                add_constraint(termsdict[ngram])
                break

    if len(constraints) > 0:
        print(json.dumps({ 'text': bpe.segment(source), 'constraints': constraints }, ensure_ascii=False))
    else:
        print(json.dumps({ 'text': bpe.segment(source) }, ensure_ascii=False))


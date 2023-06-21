# Copyright 2017--2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import unicodedata
import sys

import gzip

from glob import glob


# pip3 install wget
import wget



"""
Creates a list of sentence IDs with potential testset overlap.

Assumes test sets are in testsets/*
"""



def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

def remove_punctuation(text):
    return text.translate(punct_tbl)


def prep(x):
    return remove_punctuation(remove_accents(x.strip())).split()


def ngrams(text, nn):
    words = prep(text)
    for i in range(len(words)-nn+1):
        yield tuple(words[i:i+nn])


NN = 5 

# Read in sentence data
files = ['https://mtdataexternalpublic.blob.core.windows.net/2023datatask/2023-06-08/sentences-et-include-correct_lang.tsv.gz', # 36M
         'https://mtdataexternalpublic.blob.core.windows.net/2023datatask/2023-06-08/sentences-lt-include-correct_lang.tsv.gz', # 46M
         ]


# ~10min
sentences = dict()
for url in files:
    filename = wget.download(url, out='/tmp/')
    print('filename:', filename)
    for ii, line in enumerate(gzip.open(filename, 'rt')):
        if ii>0:
            fields = line.strip().split('\t')
            sent_id = fields[0]
            sent = fields[1]
            # make sure we can use this as a unique id
            assert sent_id not in sentences
            sentences[sent_id] = sent
        if ii % 1_000_000 == 0:
            print(ii)


print('num sents:', len(sentences))

# Make testset ngrams
test_ngrams = set()
for fname in glob('testsets/*'):
    print('reading test set', fname)
    for line in open(fname, 'rt'):
        line = line.strip()
        for ngram in ngrams(line, NN):
            test_ngrams.add(ngram)

print('number of test set ngrams:', len(test_ngrams))
            
# ~30min
bad_ids = set()
for ii, key in enumerate(sentences):
    for ngram in list(ngrams(sentences[key], NN)):
        if ngram in test_ngrams:
            bad_ids.add(key)
    if ii % 1_000_000 == 0:
        print(ii, 'of', len(sentences))

bad_ids = list(bad_ids)
bad_ids.sort()
with open('exclude_sent_ids.txt', 'wt') as fout:
    for key in bad_ids:
        fout.write(key.strip()+'\n')


print(f'found {len(bad_ids)} sentences with potential testset overlap')

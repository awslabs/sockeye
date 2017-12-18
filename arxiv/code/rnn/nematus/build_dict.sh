#!/bin/bash

. env.sh

cat $DATADIR/train.tok.bpe.{$SOURCR,$TARGET} | build_dict.py > $PAIR.dict.json

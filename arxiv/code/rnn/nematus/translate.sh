#!/bin/bash

fileIn=${1:?First argument must be the file to translate}
model=${2:-model.npz}

echo "Starting at $(date)" > /dev/stderr
THEANO_FLAGS=floatX=float32 \
    $NEMATUS/nematus/translate.py -n --models $model -i $fileIn
echo "Finished at $(date)" > /dev/stderr

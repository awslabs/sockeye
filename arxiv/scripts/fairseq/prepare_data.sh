#!/usr/bin/env bash

ln -s train.src.bpe.filter train.fs.src
ln -s train.trg.bpe.filter train.fs.trg
ln -s dev.src.bpe dev.fs.src
ln -s dev.trg.bpe dev.fs.trg
ln -s test.src.bpe test.fs.src
ln -s test.trg.bpe test.fs.trg

fairseq-preprocess --source-lang src --target-lang trg --trainpref train.fs \
  --validpref dev.fs --testpref test.fs --destdir data-bin/src_trg \
  --thresholdsrc 2 --thresholdtgt 2 --joined-dictionary --workers $(nproc)

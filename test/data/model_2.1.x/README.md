The model was generated with the following command:
```
-s docs/tutorials/seqcopy/data/dev.source
-t docs/tutorials/seqcopy/data/dev.target
-vs docs/tutorials/seqcopy/data/dev.source
-vt docs/tutorials/seqcopy/data/dev.target
--transformer-model-size 16
--num-layers 1:1
--transformer-attention-heads 2
--transformer-feed-forward-num-hidden 16
--overwrite-output
--use-cpu
--batch-type sentence
--batch-size 32
--decode-and-evaluate 400
--checkpoint-interval 500
--initial-learning-rate 0.01
--max-num-checkpoint-not-improved 4
-o model
```

The model_input is just `head dev.source`.

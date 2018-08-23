# How Much Attention Do You Need?A Granular Analysis of Neural Machine Translation Architectures

This branch contains the code and configurations to reproduce the results of the ACL 2018 paper
["How Much Attention Do You Need?A Granular Analysis of Neural Machine Translation Architectures"](http://aclweb.org/anthology/P18-1167).

## Citation

```
@InProceedings{domhant18,
  author = 	"Domhan, Tobias",
  title = 	"How Much Attention Do You Need? A Granular Analysis of Neural Machine Translation Architectures",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1799--1808",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1167"
}
```

## Configurations

### Training commands

Below we specify the architecture definitions for the different experiments of the paper.
Each definition is specified as `--custom-seq-encoder` and `--custom-seq-decoder` arguments to Sockeye training.
For a full training run the following base commands are provided, which is different for the two datasets used, namely IWSLT and WMT.

IWSLT
```bash
export NUM_HIDDEN=256
export FF_NUM_HIDDEN=1024
python -m sockeye.train \
    --prepared-data train_data -vs validation.source -vt validation.target -o model \
    --device-ids=-4 \
    --batch-type=word --batch-size=8192 \
    --encoder=custom-seq --decoder=custom-seq \
    --custom-seq-num-hidden $NUM_HIDDEN --num-embed $NUM_HIDDEN --custom-seq-dropout 0.1 \
    --label-smoothing=0.1 --weight-tying --weight-tying-type=src_trg_softmax \
    --weight-init=xavier --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg
    --optimizer=adam  --gradient-clipping-threshold=-1 \
     --initial-learning-rate=0.0002 --learning-rate-warmup=0 \ --learning-rate-decay-optimizer-states-reset best --learning-rate-decay-param-reset --min-num-epochs=1 --decode-and-evaluate=500 --keep-last-params=60
```

WMT:
```bash
export NUM_HIDDEN=256
export FF_NUM_HIDDEN=4096
python -m sockeye.train \
    --prepared-data train_data -vs validation.source -vt validation.target -o model \
    --device-ids=-4 \
    --batch-type=word --batch-size=8192 \
    --encoder=custom-seq --decoder=custom-seq \
    --custom-seq-num-hidden $NUM_HIDDEN --num-embed $NUM_HIDDEN --custom-seq-dropout 0.1 \
    --label-smoothing=0.1 --weight-tying --weight-tying-type=src_trg_softmax \
    --weight-init=xavier --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg
    --optimizer=adam  --gradient-clipping-threshold=-1 \
     --initial-learning-rate=0.0002 --learning-rate-warmup=0 \ --learning-rate-decay-optimizer-states-reset best --learning-rate-decay-param-reset --min-num-epochs=1 --decode-and-evaluate=500 --keep-last-params=60
```

Additionally, after each training parameter averaging is performed.
This can be done as follows:

```bash
mv model/params.best model/params.single.best
python -m sockeye.average -n 8 --output model/params.best --strategy best model
```

### Example

Say we want to train a standard Transformer model.
The model definition is given by:

```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff->linear)->norm"```
```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm```

Thus the full training commands are given by
```
python -m sockeye.train \
    --prepared-data train_data -vs validation.source -vt validation.target -o model \
    --device-ids=-4 \
    --batch-type=word --batch-size=8192 \
    --encoder=custom-seq --decoder=custom-seq \
    --custom-seq-num-hidden $NUM_HIDDEN --num-embed $NUM_HIDDEN --custom-seq-dropout 0.1 \
    --label-smoothing=0.1 --weight-tying --weight-tying-type=src_trg_softmax \
    --weight-init=xavier --weight-init-scale=3.0 --weight-init-xavier-factor-type=avg
    --optimizer=adam  --gradient-clipping-threshold=-1 \
     --initial-learning-rate=0.0002 --learning-rate-warmup=0 \ --learning-rate-decay-optimizer-states-reset best --learning-rate-decay-param-reset --min-num-epochs=1 --decode-and-evaluate=500 --keep-last-params=60
mv model/params.best model/params.single.best
python -m sockeye.average -n 8 --output model/params.best --strategy best model
```

### Table 3

Transforming an RNN into a Transformer style architecture. + shows the incrementally added variation. / denotes an alternative variation to which the subsequent + is relative to. The BLEU and METEOR scores are given in the paper.

Model | ADL
-- | ---
Transformer | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
RNN | ```--custom-seq-encoder "dropout->res(birnn->dropout)->repeat(5,res(rnn->dropout))"``` <br> ```--custom-seq-decoder "dropout->repeat(6,res(rnn->dropout))->res(mh_dot_att(heads=1))->res(ff($FF_NUM_HIDDEN)->linear)"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ mh | ```--custom-seq-encoder "dropout->res(birnn->dropout)->repeat(5,res(rnn->dropout))"``` <br> ```--custom-seq-decoder "dropout->repeat(6,res(rnn->dropout))->res(mh_dot_att)->res(ff($FF_NUM_HIDDEN)->linear)"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ pos | ```--custom-seq-encoder "pos->res(birnn->dropout)->repeat(5,res(rnn->dropout))"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(rnn->dropout))->res(mh_dot_att)->res(ff($FF_NUM_HIDDEN)->linear)"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ norm | ```--custom-seq-encoder "pos->res(norm->birnn->dropout)->repeat(5,res(norm->rnn->dropout))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->rnn->dropout))->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ multi-att-1h | ```--custom-seq-encoder "pos->res(norm->birnn->dropout)->repeat(5,res(norm->rnn->dropout))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att(heads=1)))->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; / multi-att | ```--custom-seq-encoder "pos->res(norm->birnn->dropout)->repeat(5,res(norm->rnn->dropout))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att))->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ ff | ```--custom-seq-encoder "pos->res(norm->birnn->dropout)->res(norm->ff($FF_NUM_HIDDEN)->linear)->repeat(5,res(norm->rnn->dropout)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```

### Table 4
Transforming a CNN based model into a Transformer style architecture. The BLEU and METEOR scores are given in the paper.

Model | ADL
-- | ---
Transformer | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
CNN GLU | ```--custom-seq-encoder "pos->repeat(6,res(cnn))"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(cnn->res(mh_dot_att(heads=1)))"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ norm | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn)->res(norm->mh_dot_att(heads=1)))->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ mh | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn)->res(norm->mh_dot_att))->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ ff | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
CNN ReLU | ```--custom-seq-encoder "pos->repeat(6,res(cnn($NUM_HIDDEN,3,relu)))"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(cnn($NUM_HIDDEN,3,relu))->res(mh_dot_att(heads=1)))"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ norm | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu)))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->mh_dot_att(heads=1)))->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ mh | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu)))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->mh_dot_att))->norm"```
&nbsp;&nbsp;&nbsp;&nbsp; \+ ff | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```

### Table 5
Different variations of the encoder and decoder self-attention layer. The BLEU and METEOR scores are given in the paper.


Encoder | Decoder | ADL
-- | --- | ----
self-att | self-att | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->norm"```
self-att | RNN | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
self-att | CNN | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
RNN | self-att | ```--custom-seq-encoder "pos->res(norm->birnn->dropout)->res(norm->ff($FF_NUM_HIDDEN)->linear)->repeat(5,res(norm->rnn->dropout)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
CNN | self-att | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
RNN | RNN | ```--custom-seq-encoder "pos->res(norm->birnn->dropout)->res(norm->ff($FF_NUM_HIDDEN)->linear)->repeat(5,res(norm->rnn->dropout)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
CNN | CNN | ```--custom-seq-encoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
self-att | combined | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(2,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->res(norm->rnn->dropout)->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear)->res(norm->cnn($NUM_HIDDEN,3,relu))->res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```
self-att | none | ```--custom-seq-encoder "pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"``` <br> ```--custom-seq-decoder "pos->repeat(6,res(norm->mh_dot_att)->res(norm->ff($FF_NUM_HIDDEN)->linear))->norm"```

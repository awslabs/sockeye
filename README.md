# How Much Attention Do You Need?A Granular Analysis of Neural Machine Translation Architectures

This branch contains the code and configurations to reproduce the results of the ACL 2018 paper
"How Much Attention Do You Need?A Granular Analysis of Neural Machine Translation Architectures".

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

### Table 3

Base command:
```
python -m sockeye.train ...
```

Architecture variations:

[//]: # "RNMT | TODO"
[//]: # "&nbsp;&nbsp;&nbsp;&nbsp; \- input feeding | TODO"

Configuration | Architecture Specification
-- | ---
Transformer | ```--custom-seq-encoder pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->ff(2048)->linear)->norm``` <br> ```--custom-seq-decoder pos->repeat(6,res(norm->mh_dot_self_att)->res(norm->mh_dot_att)->res(norm->ff(2048)->linear)->norm```
RNN | ```--custom-seq-encoder dropout->res(birnn->dropout)->repeat(5,res(rnn->dropout))``` <br> ```--custom-seq-decoder dropout->repeat(6,res(rnn->dropout))->res(mh_dot_att(heads=1))->res(ff(2048)->linear)```
&nbsp;&nbsp;&nbsp;&nbsp; \+ mh | ```--custom-seq-encoder dropout->res(birnn->dropout)->repeat(5,res(rnn->dropout))``` <br> ```--custom-seq-decoder dropout->repeat(6,res(rnn->dropout))->res(mh_dot_att)->res(ff(2048)->linear)```
&nbsp;&nbsp;&nbsp;&nbsp; \+ pos | ```--custom-seq-encoder pos->res(birnn->dropout)->repeat(5,res(rnn->dropout))``` <br> ```--custom-seq-decoder pos->repeat(6,res(rnn->dropout))->res(mh_dot_att)->res(ff(2048)->linear)```
&nbsp;&nbsp;&nbsp;&nbsp; \+ norm | ```--custom-seq-encoder pos->res(norm->birnn->dropout)->repeat(5,res(norm->rnn->dropout))->norm``` <br> ```--custom-seq-decoder pos->repeat(6,res(norm->rnn->dropout))->res(norm->mh_dot_att)->res(norm->ff(2048)->linear)->norm```
&nbsp;&nbsp;&nbsp;&nbsp; \+ multi-att-1h | ```--custom-seq-encoder pos->res(norm->birnn->dropout)->repeat(5,res(norm->rnn->dropout))->norm``` <br> ```--custom-seq-decoder pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att(heads=1)))->res(norm->ff(2048)->linear)->norm```
&nbsp;&nbsp;&nbsp;&nbsp; / multi-att | ```--custom-seq-encoder pos->res(norm->birnn->dropout)->repeat(5,res(norm->rnn->dropout))->norm``` <br> ```--custom-seq-decoder pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att))->res(norm->ff(2048)->linear)->norm```
&nbsp;&nbsp;&nbsp;&nbsp; \+ ff | ```--custom-seq-encoder pos->res(norm->birnn->dropout)->res(norm->ff(2048)->linear)->repeat(5,res(norm->rnn->dropout)->res(norm->ff(2048)->linear))->norm``` <br> ```--custom-seq-decoder pos->repeat(6,res(norm->rnn->dropout)->res(norm->mh_dot_att)->res(norm->ff(2048)->linear))->norm```

### Table 4


### Table 5

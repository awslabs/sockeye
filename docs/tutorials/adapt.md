# Domain adaptation of NMT models

Although the quality of machine translation systems is nowadays remarkably good, sometimes it is important to specialize the MT output to the specifics of certain domains.
These customizations may include preferring some word translation over others or adapting the style of the text, among others.
In this tutorial, we show two methods on how to perform domain adaptation of a general translation system using Sockeye.

We assume you already have a trained Sockeye model, for example the one trained from the [WMT tutorial](wmt.md).
We also assume that you have two training sets, one composed of general or out-of-domain (OOD) data, and one composed of in-domain (ID) data on which you want to adapt your system.
Note that both datasets need to be pre-processed in the same way.

## Preparing the data

First, you must be careful to prepare the in-domain training data using the same vocabulary as the out-of-domain data.
Assuming your prepared OOD data resides in `ood_data`

    python -m sockeye.prepare_data \
        -s data/id.train.src.bpe \
        -t data/id.train.trg.bpe \
        -o id_data \
        --source_vocab ood_data/vocab.src.0.json \
        --target_vocab ood_data/vocab.trg.0.json

Note: If your in-domain data is small, you may skip this step and add the corresponding arguments to the `sockeye.train` calls.

## Continuation of training

This method fine-tunes a trained model and starts a second training run on in-domain data, initialized with the parameters obtained from the out-domain data.
Thus you "continue training" on the data you are more interested in.
Freitag and Al-Onaizan (2016) showed that this straightforward technique can achieve good results.

When training a model, you can load a set of parameters with the `--params` argument in Sockeye, specifying an already trained model.
Assuming the trained model resides in `ood`, a possible invocation could be

    python -m sockeye.train \
        --config ood/args.yaml \
        -d id_data \
        -vs data/id.dev.src.bpe \
        -vt data/id.dev.trg.bpe \
        --params ood/params.best \
        -o id_continuation

Depending on the size of your training data you may want to adjust the parameters of the learning algorithm (learning rate, decay, etc.) and perhaps the checkpoint interval.

## Learning Hidden Unit Contribution

Learning Hidden Unit Contribution (LHUC) is a method proposed by Vilar (2018), where the output of the hidden units in a network are expanded with an additional multiplicative unit.
This unit can the strengthen or dampen the output of the corresponding unit.

The usage is very similar as the call shown above, but you have to specify an additional `--lhuc` argument.
This argument accepts a (space separated) list of components where to apply the LHUC units (`encoder`, `decoder` or `state_init`) or you can specify `all` for adding it to all supported components:

    python -m sockeye.train \
        --config ood/args.yaml \
        -d id_data \
        -vs data/id.dev.src.bpe \
        -vt data/id.dev.trg.bpe \
        --params ood/params.best \
        --lhuc all \
        -o id_lhuc

Again it may be beneficial to adjust the learning parameters for the adaptation run.

## References

> Markus Freitag and Yaser Al-Onaizan. 2016.
> [Fast Domain Adaptation for Neural Machine Translation](http://arxiv.org/pdf/1612.06897v1)
> ArXiv e-prints.

> David Vilar. 2018.
> [Learning Hidden Unit Contribution for Adapting Neural Machine Translation Models](http://aclweb.org/anthology/N18-2080)
> Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers).

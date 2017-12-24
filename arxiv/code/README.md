# Sockeye Paper Toolkit Comparison

The scripts underneath this subdirectory capture the code used to run
the comparisons in the paper:

    @article{Sockeye:17,
        author = {Hieber, Felix and Domhan, Tobias and Denkowski, Michael
                  and Vilar, David and Sokolov, Artem, and Clifton, Ann and Post, Matt},
        title = "{Sockeye: A Toolkit for Neural Machine Translation}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1712.05690},
        primaryClass = "cs.CL",
        keywords = {Computer Science - Computation and Language, Computer Science - Learning, Statistics - Machine Learning},
        year = 2017,
        month = dec,
        url = {https://arxiv.org/abs/1712.05690}
    }

The top-level script, `prepare_train.sh`, will download and preprocess
the data as was done for our experiments. You'll have to download the
official datasets from the
[WMT download page](http://statmt.org/wmt17/translation-task.html);
you can match them to the source prefixes listed in `train.en-de.txt`
and `train.lv-en.txt`. These prefixes are used directly by
`prepare_train.sh`, so you may also have to adjust paths. It also
makes use of `prepare_devtest.sh`, which applies preprocessing and BPE
to dev and test data.

The training scripts are parameterized to load a file in the current
directory, `env.sh`, which defines a single environment variable,
`$PAIR`. For our experiments this was either "en-de" or "lv-en", but
this could easily be extended to other languages, of course.

You will also want to edit the top-level `env.sh` to define locations
of the toolkits and, especially, `$DATADIR`, which points to where the
data was downloaded and preprocessed.

For evaluation with BLEU, you will need to install `sacreBLEU`:

    pip3 install sacrebleu

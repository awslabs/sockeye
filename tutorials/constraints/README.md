# Decoding with lexical constraints

Lexical constraints provide a way to force the model to include certain words in the output.
Given a set of constraints, the decoder will find the best output that includes the constraints.
This file describes how to use lexical constraints; for more technical information, please see our paper:

```
@InProceedings{post2018:fast,
    author = "Post, Matt and Vilar, David",
    title = "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation",
    booktitle = "Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    year = "2018",
    publisher = "Association for Computational Linguistics",
    pages = "1314--1324",
    location = "New Orleans, Louisiana",
    url = "http://aclweb.org/anthology/N18-1119",
}
```

## Example

You need a [trained model](../wmt/README.md).

You need to be careful to apply the same preprocessing to your test data that you applied at training time, including
any [subword processing](http://github.com/rsennrich/subword-nmt), since Sockeye itself does not do this.

Constraints must be encoded with a JSON object.
This JSON object can be produced with the provided script:

    echo -e "This is a test .\tconstraint\tmulti@@ word const@@ raint" \
      | python3 -m sockeye.lexical_constraints

The script creates a Python object with the constraints encoded as follows:

    { 'text': 'This is a test .', 'constraints': ['constr@@ aint', 'multi@@ word constr@@ aint'] }]

You can pass the output of this to Sockeye.
Make sure that you (a) the JSON object is on a *single line* and (b) you pass `--json-input` to Sockeye, so that it knows to parse the input (without that flag, it will treat the JSON input as a regular sentence).
We also recommend that you increase the beam a little bit and enable beam pruning:

    echo -e "This is a test .\tconstraint\tmultiword constraint" \
      | python3 -m sockeye.lexical_constraints \
      | python3 -m sockeye.translate -m /path/to/model --json-input --beam-size 20 --beam-prune 20 [other args]

You will get a translation with the required constraints as part of the output.

## Scoring

Lexical constraints can also be used as a rudimentary scoring mechanism, by providing the entire reference (with `<s>` and `</s>` tokens) as a constraint.
For example:

    echo '{ "text": "This is a test", "constraints": ["<s> Dies ist ein Test </s>"] }' \
      python3 -m sockeye.translate -m /path/to/model --json-input --beam-size 1 --output-type translation_with_score

This will output tab-delimited pairs of (score, translation).
As always, don't forget to apply source- and target-side preprocessing to your input and your constraint.

However, it is probably better to use [Sockeye's scoring module](../scoring.md) directly, since it makes use of the training time computation graph and is therefore much faster.

## Negative constraints

Negative constraints---phrases that must *not* appear in the output---are also supported.
To use them, use the "avoid" key in the JSON object instead of "constraints".
An example JSON object:

    { 'text': 'This is a test .', 'avoid': ['Test'] }

Multiple negative constraints and multi-word negative constraints are supported, just like for positive constraints.
You can also add `<s>` and `</s>` to constraints, to specify that phrases that must not start or end a sentence.
Don't forget to apply your preprocessing!

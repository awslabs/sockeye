# Decoding with lexical constraints

Lexical constraints provide a way to force the model to include certain words in the output.
Given a set of constraints, the decoder will find the best output that includes the constraints.
This file describes how to use lexical constraints; for more technical information, please see our paper:

  Fast Lexically Constrained Decoding With Dynamic Beam Allocation for Neural Machine Translation
  Matt Post & David Vilar
  [NAACL 2018](http://naacl2018.org/)
  [PDF](https://arxiv.org/pdf/1804.06609.pdf)

## Example

You need a [trained model](../wmt/README.md).

You need to be careful to apply the same preprocessing to your test data that you applied at training time, including
any [subword processing](http://github.com/rsennrich/subword-nmt), since Sockeye itself does not do this.

Constraints must be encoded with a JSON object.
This JSON object can be produced with the provided script:

    echo -e "This is a test .\tconstraint\tmulti@@ word const@@ raint" \
      | python3 -m sockeye.lexical_constraints

The script creates a Python object with the constraints encoded as follows (except on one line):

    { 'text': 'This is a test .',
      'constraints': ['constr@@ aint',
                      'multi@@ word constr@@ aint'] }

You can pass the output of this to Sockeye. Make sure that you specify `--json-input` so that Sockeye knows to parse the
input (without that flag, it will treat the JSON input as a regular sentence). We also recommend that you increase the
beam a little bit and enable beam pruning:

    echo -e "This is a test .\tconstraint\tmultiword constraint" \
      | python3 -m sockeye.lexical_constraints \
      | python3 -m sockeye.translate -m /path/to/model --json-input --beam-size 20 --beam-prune 20 [other args]

You will get a translation with the required constraints as part of the output.

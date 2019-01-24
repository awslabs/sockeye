---
layout: default
---

# Translation

Decoding (a.k.a. inference or translation) in sockeye is made available through the `sockeye.translate` module.

```bash
> python -m sockeye.translate
```

The only required argument is `--models`, which should point to an `<model_dir>` folder of trained models.
By default, sockeye chooses the parameters from the best checkpoint and uses these for translation.
You can specify parameters from a specific checkpoint by using `--checkpoints X`.

You can control the size of the beam using `--beam-size` and the maximum input length by `--max-input-length`.
Sentences that are longer than `max-input-length` are stripped.

Input is read from the standard input and the output is written to the standard output.
The CLI will log translation speed once the input is consumed.
Like in the training module, the first GPU device is used by default.
Note however that multi-GPU translation is not currently supported. For CPU decoding use `--use-cpu`.

Use the `--help` option to see a full list of options for translation.

## Ensemble Decoding

Sockeye supports ensemble decoding by specifying multiple model directories and multiple checkpoints.
The given lists must have the same length, such that the first given checkpoint will be taken from the first model directory, the second specified checkpoint from the second directory, etc.

```bash
> python -m sockeye.translate --models [<m1prefix> <m2prefix>] --checkpoints [<cp1> <cp2>]
```

## Visualization

The default mode of the translate CLI is to output translations to STDOUT.
You can also print out an ASCII matrix of the alignments using `--output-type align_text`, or save the alignment matrix as a PNG plot using `--output-type align_plot`.
The PNG files will be written to files beginning with the prefix given by the `--align-plot-prefix` option, one for each input sentence, indexed by the sentence id.

## Source factors

If your [model was trained with source factors](training.html#source-factors), you will need to supply them at test-time, too.
Factors can be provided in three formats: (a) separate, token-parallel files (as in training), (b) direct annotations on words, or (c) in a JSON object.

### Parallel files

You can also provide parallel files, [in the same style as training](training.html#source-factors).
Factor files are token-parallel to the source and are passed in to `sockeye.translate` via the `--input-factors` flag.
(In this scenario, the source is another file, passed via `--input`).

### Direct annotations

Here, factors are appended to each token and delimited with a `|` symbol.
For example:

    The|O boy|O ate|O the|O waff@@|B le|E .|O

Any number of factors can be supplied; just delimit them with `|`.
Factor representation are dense; each word must be annotated for all factors.

### Input and output with JSON

Sockeye supports JSON for both input and output.
JSON input is enabled by adding the `--json-input` to the call to `sockeye.translate`.
In this case, Sockeye will take the text to translate from the "text" field.
Sockeye expects a complete JSON object on each line of input.
This JSON object can also specify the source factors as a list of token-parallel strings, e.g.,

```python
{ "text": "The boy ate the waff@@ le .", "factors": ["O O O O B E O"] }
```

JSON output is enabled with the `--output-type json` flag.
The translation itself will appear in a `translation` field, along with other fields such as `sentence_id`.

If both JSON input and output are enabled, Sockeye will push through all fields in the input object that it doesn't overwrite.
For example, if your input is:

```json
{ "text": "The boy ate the waff@@ le .", "sentiment_id": "positive" }
```

The output may be:

```json
{ "sentence_id": 1, "sentiment_id": "positive", "text": "The boy ate the waff@@ le .", "translation": "Der Junge aß die Waffel." }
```

## N-best translations

Sockeye can return the n best hypotheses per input (*nbest lists*).
Such nbest lists can for example be used in reranking (`python -m sockeye.rerank`).

When `--nbest-size > 1`, each line in the output of `translate` will contain the following JSON object:
```python
{"alignments": [<1st alignment>, <2nd alignment>, ...], "scores": [<1st score>, <2nd score>, ...], "translations": ["<1st hypothesis>", "<2nd hypothesis>", ...]}
```
Note that `--nbest-size` must be smaller or equal to `--beam-size` and `--beam-search-stop` must be set to `all`.

## Lexical constraints

Lexical constraints provide a way to force the model to include certain words in the output.
Given a set of constraints, the decoder will find the best output that includes the constraints.
This file describes how to use lexical constraints; for more technical information, please see our paper:

> Matt Post and David Vilar. 2018.
> [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](http://aclweb.org/anthology/N18-1119).
> Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers).

### Example

You need a [trained model](tutorials/wmt.html).

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

### Negative constraints

Negative constraints---phrases that must *not* appear in the output---are also supported.
To use them, use the "avoid" key in the JSON object instead of "constraints".
An example JSON object:

    { 'text': 'This is a test .', 'avoid': ['Test'] }

Multiple negative constraints and multi-word negative constraints are supported, just like for positive constraints.
You can also add `<s>` and `</s>` to constraints, to specify that phrases that must not start or end a sentence.
Don't forget to apply your preprocessing!

### Scoring

Lexical constraints can also be used as a rudimentary scoring mechanism, by providing the entire reference (with `<s>` and `</s>` tokens) as a constraint.
(However, you'll get significantly faster results with [Sockeye's scoring module](scoring.html)).
For example:

    echo '{ "text": "This is a test", "constraints": ["<s> Dies ist ein Test </s>"] }' \
      python3 -m sockeye.translate -m /path/to/model --json-input --beam-size 1 --output-type translation_with_score

This will output tab-delimited pairs of (score, translation).
As always, don't forget to apply source- and target-side preprocessing to your input and your constraint.

## CPU process per core translation

On multi-core computers, translation per core separately can speedup translation performance, due to some operation can't be handled parallel in one process.
Using this method, translation on each core can be parallel.

One [python script example](https://raw.githubusercontent.com/awslabs/sockeye/master/docs/tutorials/cpu_process_per_core_translation.py) is given and you can run it as follows:

```bash
> python cpu_process_per_core_translation.py -m model -i input_file_name -o output_file_name -bs batch_size -t true
```

Options:

- `-t true`: each core translate the whole input file.

- `-t false`: each core translate (input file line/core number) lines , then merge the translated file into one complete output file.

## Sampling

Instead of filling the beam with the best items at each step of the decoder, Sockeye can sample from the target distributions of each hypothesis using `--sample [N]`.
If the optional parameter `N` is specified, the sampling will be limited to the top `N` vocabulary items.
The default, `N = 0`, which means to sample from the full distribution over all target vocabulary items.
Limiting `N` to a value that is much smaller than the target vocabulary size (say, 5%) can lead to much more sensible samples.

You can use this with `--nbest-size` to output multiple samples for each input.
However, note that since each beam item is sampled independently, there is no guarantee that sampled items will be unique.
You can use `--softmax-temperature T` to make the target distributions more peaked (`T < 1.0`) or smoother (`T > 1.0`).

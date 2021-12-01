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

If your [model was trained with source factors](training.md#source-factors), you will need to supply them at test-time, too.
Factors can be provided in three formats: (a) separate, token-parallel files (as in training), (b) direct annotations on words, or (c) in a JSON object.

### Parallel files

You can also provide parallel files, [in the same style as training](training.md#source-factors).
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

## Decoding with brevity penalty

To nudge Sockeye towards longer translations, you can enable a penalty for short translations by setting `--brevity-penalty-type` to `learned` or `constant`.
With the former setting, provided the training was done with `--length-task`, Sockeye will predict the reference length individually for each sentence
and use it to calculate the (logarithmic) brevity penalty `weight * min(0.0, 1 - |ref|/|hyp|)` that will be subtracted from the scores to reward longer sentences.
The latter setting, by default, will use a constant length ratio for all sentences that was estimated on the training data.
The value of the constant can be changed with `--brevity-penalty-constant-length-ratio`.

## Sampling

Instead of filling the beam with the best items at each step of the decoder, Sockeye can sample from the target distributions of each hypothesis using `--sample [N]`.
If the optional parameter `N` is specified, the sampling will be limited to the top `N` vocabulary items.
If `--sample` is used without an integer, the default `N = 0` applies. `N = 0` means to sample from the full distribution over all target vocabulary items.
Limiting `N` to a value that is much smaller than the target vocabulary size (say, 5%) can lead to much more sensible samples.
Likewise, you can use `--softmax-temperature T` to make the target distributions more peaked (`T < 1.0`) or smoother (`T > 1.0`).

You can use this with `--nbest-size` to output multiple samples for each input.
However, note that since each beam item is sampled independently, there is no guarantee that sampled items will be unique.
Also note that the samples in an nbest list will be sorted according to model scores.

# Scoring existing translations

Sockeye provides a fast scoring module that permits the scoring of existing translations.
It works by making use of the training computation graph, but turning off caching of gradients and loss computation.
Just like when training models, the scorer works with raw plain-text data passed in via `--source` and `--target`.
It can easily therefore taken any pretrained model, just like in inference.

## Example

To score a source and target dataset, first make sure that all source and target preprocessing have been applied.
Then run this command:

    python3 -m sockeye.score -m MODEL --source SOURCE --target TARGET

Sockeye will output a score (a negative log probability) for each sentence pair.

## Command-line arguments

The scorer takes a number of arguments:

-  `--score-type logprob`. Use this to get log probabilities instead of negative log probabilities.
-  `--batch-size X`. Word-based batching is used.
   You can use this flag to change the batch size from its default of 500.
   If you run out of memory, try lowering this.
-  `--output-type {score,pair_with_score}`. The output type: either the score alone, or the score with the translation pair.
   Fields will be separated by a tab.
- `--max-seq-len M:N`. The maximum sequences length (`M` the source length, `N` the target).
- `--softmax-temperature X`. Scales the logits by dividing by this argument before computing softmax.
- `--length-penalty-alpha`, `--length-penalty-beta`. Parameters for length normalization.
  Set `--length-penalty-alpha 0` to disable normalization.

## Caveat emptor

Some things to watch out for:

- Scoring reads the maximum sentence lengths from the model.
  Sentences longer than these will be skipped, meaning the scored output will not be parallel with the input.
  A warning message will be printed to STDERR, but beware.

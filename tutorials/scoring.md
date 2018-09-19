# Scoring existing translations

Sockeye provides a fast scoring module that permits the scoring of existing translations.
Just like when training models, the scorer can make use of prepared data (via `--prepare DATA_DIR`) or can take raw source and target data (via `--source` and `--target`).
In fact, Sockeye's scoring model makes use of the training computation graph, but it turns off the caching of gradients and the computation of the loss.
It can easily therefore taken any pretrained model, just like in inference.

## Example

To score a source and target dataset, first make sure that all source and target preprocessing have been applied.
Then run this command:

    python3 -m sockeye.score -m MODEL --source SOURCE --target TARGET

Sockeye will output a score (a negative log probability) for each sentence pair.
You can also provide prepared data:

    python3 -m sockeye.score -m MODEL --prepared-data DATA_DIR

## Command-line arguments

The scorer takes a number of arguments:

-  `--score-type logprob`. Use this to get log probabilities instead of negative log probabilities.
-  `--batch-size X`. You can change the batch size from its default of 500.
   If you run out of memory, try changing this.
-  `--output FIELDS`. The fields to output.
   Fields will be separated by a tab.
   The options are `id` (the sentence ID), `source` (the source sentence), `target` (the target sentence), and `score` (formatted according to `--score-type`).

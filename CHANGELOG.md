# Changelog
All notable changes to the project are documented in this file.

Version numbers are of the form `1.0.0`.
Any version bump in the last digit is backwards-compatible, in that a model trained with the previous version can still
be used for translation with the new version.
Any bump in the second digit indicates a backwards-incompatible change,
e.g. due to changing the architecture or simply modifying model parameter names.
Note that Sockeye has checks in place to not translate with an old model that was trained with an incompatible version.

Each version section may have have subsections for: _Added_, _Changed_, _Removed_, _Deprecated_, and _Fixed_.

## [1.18.59]
### Added
- Full training state is now returned from EarlyStoppingTrainer's fit().
### Changed
- Training state cleanup will not be performed for training runs that did not converge yet.
- Switched to portalocker for locking files (Windows compatibility).

## [1.18.58]
### Added
- Added nbest translation, exposed as `--nbest-size`. Nbest translation means to not only output the most probable translation according to a model, but the top n most probable hypotheses. If `--nbest-size > 1` and the option `--output-type` is not explicitly specified, the output type will be changed to one JSON list of nbest translations per line. `--nbest-size` can never be larger than `--beam-size`.

### Changed
- Changed `sockeye.rerank` CLI to be compatible with nbest translation JSON output format.

## [1.18.57]
### Added
- Added `sockeye.score` CLI for quickly scoring existing translations ([documentation](tutorials/scoring.md)).
### Fixed
- Entry-point clean-up after the contrib/ rename

## [1.18.56]
### Changed
- Update to MXNet 1.3.0.post0

## [1.18.55]
- Renamed `contrib` to less-generic `sockeye_contrib`

## [1.18.54]
### Added
- `--source-factor-vocabs` can be set to provide source factor vocabularies.

## [1.18.53]
### Added
- Always skipping softmax for greedy decoding by default, only for single models.
- Added option `--skip-topk` for greedy decoding.

## [1.18.52]
### Fixed
- Fixed bug in constrained decoding to make sure best hypothesis satifies all constraints.

## [1.18.51]
### Added
- Added a CLI for reranking of an nbest list of translations.

## [1.18.50]
### Fixed
- Check for equivalency of training and validation source factors was incorrectly indented.

## [1.18.49]
### Changed
- Removed dependence on the nvidia-smi tool. The number of GPUs is now determined programatically.

## [1.18.48]
### Changed
- Translator.max_input_length now reports correct maximum input length for TranslatorInput objects, independent of the internal representation, where an additional EOS gets added. 

## [1.18.47]
### Changed
- translate CLI: no longer rely on external, user-given input id for sorting translations. Also allow string ids for sentences.

## [1.18.46]
### Fixed
- Fixed issue with `--num-words 0:0` in image captioning and another issue related to loading all features to memory with variable length.

## [1.18.45]
### Added
- Added an 8 layer LSTM model similar (but not exactly identical) to the 'GNMT' architecture to autopilot.

## [1.18.44]
### Fixed
- Fixed an issue with `--max-num-epochs` causing training to stop before the update/batch that actually completes the epoch was made.

## [1.18.43]
### Added
- `<s>` now supported as the first token in a multi-word negative constraint
  (e.g., `<s> I think` to prevent a sentence from starting with `I think`)
### Fixed
- Bugfix in resetting the state of a multiple-word negative constraint

## [1.18.42]
### Changed
- Simplified gluon blocks for length calculation

## [1.18.41]
### Changed
- Require numpy 1.14 or later to avoid MKL conflicts between numpy as mxnet-mkl.

## [1.18.40]
### Fixed
- Fixed bad check for existence of negative constraints.
- Resolved conflict for phrases that are both positive and negative constraints.
- Fixed softmax temperature at inference time.

## [1.18.39]
### Added
- Image Captioning now supports constrained decoding.
- Image Captioning: zero padding of features now allows input features of different shape for each image.

## [1.18.38]
### Fixed
- Fixed issue with the incorrect order of translations when empty inputs are present and translating in chunks.

## [1.18.37]
### Fixed
- Determining the max output length for each sentence in a batch by the bucket length rather than the actual in order to match the behavior of a single sentence translation.

## [1.18.36]
### Changed
- Updated to [MXNet 1.2.1](https://github.com/apache/incubator-mxnet/tree/1.2.1)

## [1.18.35]
### Added
- ROUGE scores are now available in `sockeye-evaluate`.
- Enabled CHRF as an early-stopping metric.

## [1.18.34]
### Added
- Added support for `--beam-search-stop first` for decoding jobs with `--batch-size > 1`.

## [1.18.33]
### Added
- Now supports negative constraints, which are phrases that must *not* appear in the output.
   - Global constraints can be listed in a (pre-processed) file, one per line: `--avoid-list FILE`
   - Per-sentence constraints are passed using the `avoid` keyword in the JSON object, with a list of strings as its field value.

## [1.18.32]
### Added
- Added option to pad vocabulary to a multiple of x: e.g. `--pad-vocab-to-multiple-of 16`.

## [1.18.31]
### Added
- Pre-training the RNN decoder. Usage:
  1. Train with flag `--decoder-only`.
  2. Feed identical source/target training data.

## [1.18.30]
### Fixed
- Preserving max output length for each sentence to allow having identical translations for both with and without batching.

## [1.18.29]
### Changed
- No longer restrict the vocabulary to 50,000 words by default, but rather create the vocabulary from all words which occur at least `--word-min-count` times. Specifying `--num-words` explicitly will still lead to a restricted
  vocabulary.

## [1.18.28]
### Changed
- Temporarily fixing the pyyaml version to 3.12 as version 4.1 introduced some backwards incompatible changes.

## [1.18.27]
### Fixed
- Fix silent failing of NDArray splits during inference by using a version that always returns a list. This was causing incorrect behavior when using lexicon restriction and batch inference with a single source factor.

## [1.18.26]
### Added
- ROUGE score evaluation. It can be used as the stopping criterion for tasks such as summarization.

## [1.18.25]
### Changed
- Update requirements to use MKL versions of MXNet for fast CPU operation.

## [1.18.24]
### Added
- Dockerfiles and convenience scripts for running `fast_align` to generate lexical tables.
These tables can be used to create top-K lexicons for faster decoding via vocabulary selection ([documentation](https://github.com/awslabs/sockeye/tree/master/contrib/fast_align)).

### Changed
- Updated default top-K lexicon size from 20 to 200.

## [1.18.23]
### Fixed
- Correctly create the convolutional embedding layers when the encoder is set to `transformer-with-conv-embed`. Previously
no convolutional layers were added so that a standard Transformer model was trained instead.

## [1.18.22]
### Fixed
- Make sure the default bucket is large enough with word based batching when the source is longer than the target (Previously
there was an edge case where the memory usage was sub-optimal with word based batching and longer source than target sentences).

## [1.18.21]
### Fixed
- Constrained decoding was missing a crucial cast
- Fixed test cases that should have caught this

## [1.18.20]
### Changed
- Transformer parametrization flags (model size, # of attention heads, feed-forward layer size) can now optionally
  defined separately for encoder & decoder. For example, to use a different transformer model size for the encoder,
  pass `--transformer-model-size 1024:512`.

## [1.18.19]
### Added
- LHUC is now supported in transformer models

## [1.18.18]
### Added
- \[Experimental\] Introducing the image captioning module. Type of models supported: ConvNet encoder - Sockeye NMT decoders. This includes also a feature extraction script,
an image-text iterator that loads features, training and inference pipelines and a visualization script that loads images and captions.
See [this tutorial](tutorials/image_captioning) for its usage. This module is experimental therefore its maintenance is not fully guaranteed.

## [1.18.17]
### Changed
- Updated to MXNet 1.2
- Use of the new LayerNormalization operator to save GPU memory.

## [1.18.16]
### Fixed
- Removed summation of gradient arrays when logging gradients.
  This clogged the memory on the primary GPU device over time when many checkpoints were done.
  Gradient histograms are now logged to Tensorboard separated by device.

## [1.18.15]
### Added
- Added decoding with target-side lexical constraints (documentation in `tutorials/constraints`).

## [1.18.14]
### Added
- Introduced Sockeye Autopilot for single-command end-to-end system building.
See the [Autopilot documentation]((https://github.com/awslabs/sockeye/tree/master/contrib/autopilot)) and run with: `sockeye-autopilot`.
Autopilot is a `contrib` module with its own tests that are run periodically.
It is not included in the comprehensive tests run for every commit.

## [1.18.13]
### Fixed
- Fixed two bugs with training resumption:
  1. removed overly strict assertion in the data iterator for model states before the first checkpoint.
  2. removed deletion of Tensorboard log directory.

### Added
- Added support for config files. Command line parameters have precedence over the values read from the config file.
  Minimal working example:
  `python -m sockeye.train --config config.yaml` with contents of `config.yaml` as follows:
  ```yaml
  source: source.txt
  target: target.txt
  output: out
  validation_source: valid.source.txt
  validation_target: valid.target.txt
  ```
### Changed
  The full set of arguments is serialized to `out/args.yaml` at the beginning of training (before json was used).

## [1.18.12]
### Changed
- All source side sequences now get appended an additional end-of-sentence (EOS) symbol. This change is backwards
  compatible meaning that inference with older models will still work without the EOS symbol.

## [1.18.11]
### Changed
- Default training parameters have been changed to reflect the setup used in our arXiv paper. Specifically, the default
  is now to train a 6 layer Transformer model with word based batching. The only difference to the paper is that weight
  tying is still turned off by default, as there may be use cases in which tying the source and target vocabularies is
  not appropriate. Turn it on using `--weight-tying --weight-tying-type=src_trg_softmax`. Additionally, BLEU scores from
  a checkpoint decoder are now monitored by default.

## [1.18.10]
### Fixed
- Re-allow early stopping w.r.t BLEU

## [1.18.9]
### Fixed
- Fixed a problem with lhuc boolean flags passed as None.

### Added
- Reorganized beam search. Normalization is applied only to completed hypotheses, and pruning of
  hypotheses (logprob against highest-scoring completed hypothesis) can be specified with
  `--beam-prune X`
- Enabled stopping at first completed hypothesis with `--beam-search-stop first` (default is 'all')

## [1.18.8]
### Removed
- Removed tensorboard logging of embedding & output parameters at every checkpoint. This used a lot of disk space.

## [1.18.7]
### Added
- Added support for LHUC in RNN models (David Vilar, "Learning Hidden Unit
  Contribution for Adapting Neural Machine Translation Models" NAACL 2018)

### Fixed
- Word based batching with very small batch sizes.

## [1.18.6]
### Fixed
- Fixed a problem with learning rate scheduler not properly being loaded when resuming training.

## [1.18.5]
### Fixed
- Fixed a problem with trainer not waiting for the last checkpoint decoder (#367).

## [1.18.4]
### Added
- Added options to control training length w.r.t number of updates/batches or number of samples:
  `--min-updates`, `--max-updates`, `--min-samples`, `--max-samples`.

## [1.18.3]
### Changed
- Training now supports training and validation data that contains empty segments. If a segment is empty, it is skipped
  during loading and a warning message including the number of empty segments is printed.

## [1.18.2]
### Changed
- Removed combined linear projection of keys & values in source attention transformer layers for
  performance improvements.
- The topk operator is performed in a single operation during batch decoding instead of running in a loop over each
sentence, bringing speed benefits in batch decoding.

## [1.18.1]
### Added
- Added Tensorboard logging for all parameter values and gradients as histograms/distributions. The logged values
  correspond to the current batch at checkpoint time.

### Changed
- Tensorboard logging now is done with the MXNet compatible 'mxboard' that supports logging of all kinds of events
  (scalars, histograms, embeddings, etc.). If installed, training events are written out to Tensorboard compatible
  even files automatically.

### Removed
- Removed the `--use-tensorboard` argument from `sockeye.train`. Tensorboard logging is now enabled by default if
  `mxboard` is installed.

## [1.18.0]
### Changed
- Change default target vocab name in model folder to `vocab.trg.0.json`
- Changed serialization format of top-k lexica to pickle/Numpy instead of JSON.
- `sockeye-lexicon` now supports two subcommands: create & inspect.
  The former provides the same functionality as the previous CLI.
  The latter allows users to pass source words to the top-k lexicon to inspect the set of allowed target words.

### Added
- Added ability to choose a smaller `k` at decoding runtime for lexicon restriction.

## [1.17.5]
### Added
- Added a flag `--strip-unknown-words` to `sockeye.translate` to remove any `<unk>` symbols from the output strings.

## [1.17.4]
### Added
- Added a flag `--fixed-param-names` to prevent certain parameters from being optimized during training.
  This is useful if you want to keep pre-trained embeddings fixed during training.
- Added a flag `--dry-run` to `sockeye.train` to not perform any actual training, but print statistics about the model
  and mode of operation.

## [1.17.3]
### Changed
- `sockeye.evaluate` can now handle multiple hypotheses files by simply specifying `--hypotheses file1 file2...`.
For each metric the mean and standard deviation will be reported across files.

## [1.17.2]
### Added
- Optionally store the beam search history to a `json` output using the `beam_store` output handler.

### Changed
- Use stack operator instead of expand_dims + concat in RNN decoder. Reduces memory usage.

## [1.17.1]
### Changed
 - Updated to [MXNet 1.1.0](https://github.com/apache/incubator-mxnet/tree/1.1.0)

## [1.17.0]
### Added
 - Source factors, as described in

   Linguistic Input Features Improve Neural Machine Translation (Sennrich \& Haddow, WMT 2016)
   [PDF](http://www.aclweb.org/anthology/W16-2209.pdf) [bibtex](http://www.aclweb.org/anthology/W16-2209.bib)

   Additional source factors are enabled by passing `--source-factors file1 [file2 ...]` (`-sf`), where file1, etc. are
   token-parallel to the source (`-s`).
   An analogous parameter, `--validation-source-factors`, is used to pass factors for validation data.
   The flag `--source-factors-num-embed D1 [D2 ...]` denotes the embedding dimensions and is required if source factor
   files are given. Factor embeddings are concatenated to the source embeddings dimension (`--num-embed`).

   At test time, the input sentence and its factors can be passed in via STDIN or command-line arguments.
   - For STDIN, the input and factors should be in a token-based factored format, e.g.,
     `word1|factor1|factor2|... w2|f1|f2|... ...1`.
   - You can also use file arguments, which mirrors training: `--input` takes the path to a file containing the source,
     and `--input-factors` a list of files containing token-parallel factors.
   At test time, an exception is raised if the number of expected factors does not
   match the factors passed along with the input.

 - Removed bias parameters from multi-head attention layers of the transformer.

## [1.16.6]
### Changed
 - Loading/Saving auxiliary parameters of the models. Before aux parameters were not saved or used for initialization.
 Therefore the parameters of certain layers were ignored (e.g., BatchNorm) and randomly initialized. This change
 enables to properly load, save and initialize the layers which use auxiliary parameters.

## [1.16.5]
### Changed
 - Device locking: Only one process will be acquiring GPUs at a time.
 This will lead to consecutive device ids whenever possible.

## [1.16.4]
### Changed
 - Internal change: Standardized all data to be batch-major both at training and at inference time.

## [1.16.3]
### Changed
 - When a device lock file exists and the process has no write permissions for the lock file we assume that the device
 is locked. Previously this lead to an permission denied exception. Please note that in this scenario we an not detect
 if the original Sockeye process did not shut down gracefully. This is not an issue when the sockeye process has write
 permissions on existing lock files as in that case locking is based on file system locks, which cease to exist when a
 process exits.

## [1.16.2]
### Changed
 - Changed to a custom speedometer that tracks samples/sec AND words/sec. The original MXNet speedometer did not take
 variable batch sizes due to word-based batching into account.

## [1.16.1]
### Fixed
 - Fixed entry points in `setup.py`.

## [1.16.0]
### Changed
 - Update to [MXNet 1.0.0](https://github.com/apache/incubator-mxnet/tree/1.0.0) which adds more advanced indexing
features, benefitting the beam search implementation.
 - `--kvstore` now accepts 'nccl' value. Only works if MXNet was compiled with `USE_NCCL=1`.

### Added
 - `--gradient-compression-type` and `--gradient-compression-threshold` flags to use gradient compression.
  See [MXNet FAQ on Gradient Compression](https://mxnet.incubator.apache.org/versions/master/faq/gradient_compression.html).

## [1.15.8]
### Fixed
 - Taking the BOS and EOS tag into account when calculating the maximum input length at inference.

## [1.15.7]
### Fixed
 - fixed a problem with `--num-samples-per-shard` flag not being parsed as int.

## [1.15.6]
### Added
 - New CLI `sockeye.prepare_data` for preprocessing the training data only once before training,
 potentially splitting large datasets into shards. At training time only one shard is loaded into memory at a time,
 limiting the maximum memory usage.

### Changed
 - Instead of using the ```--source``` and ```--target``` arguments ```sockeye.train``` now accepts a
 ```--prepared-data``` argument pointing to the folder containing the preprocessed and sharded data. Using the raw
 training data is still possible and now consumes less memory.

## [1.15.5]
### Added
 - Optionally apply query, key and value projections to the source and target hidden vectors in the CNN model
 before applying the attention mechanism. CLI parameter: `--cnn-project-qkv`.

## [1.15.4]
### Added
 - A warning will be printed if the checkpoint decoder slows down training.

## [1.15.3]
### Added
 - Exposing the xavier random number generator through `--weight-init-xavier-rand-type`.

## [1.15.2]
### Added
 - Exposing MXNet's Nesterov Accelerated Gradient, Adadelta and Adadelta optimizers.

## [1.15.1]
### Added
 - A tool that initializes embedding weights with pretrained word representations, `sockeye.init_embedding`.

## [1.15.0]
### Added
- Added support for Swish-1 (SiLU) activation to transformer models
([Ramachandran et al. 2017: Searching for Activation Functions](https://arxiv.org/pdf/1710.05941.pdf),
[Elfwing et al. 2017: Sigmoid-Weighted Linear Units for Neural Network Function Approximation
in Reinforcement Learning](https://arxiv.org/pdf/1702.03118.pdf)).  Use `--transformer-activation-type swish1`.
- Added support for GELU activation to transformer models ([Hendrycks and Gimpel 2016: Bridging Nonlinearities and
Stochastic Regularizers with Gaussian Error Linear Units](https://arxiv.org/pdf/1606.08415.pdf).
Use `--transformer-activation-type gelu`.

## [1.14.3]
### Changed
- Fast decoding for transformer models. Caches keys and values of self-attention before softmax.
Changed decoding flag `--bucket-width` to apply only to source length.

## [1.14.2]
### Added
 - Gradient norm clipping (`--gradient-clipping-type`) and monitoring.
### Changed
 - Changed `--clip-gradient` to `--gradient-clipping-threshold` for consistency.

## [1.14.1]
### Changed
 - Sorting sentences during decoding before splitting them into batches.
 - Default chunk size: The default chunk size when batching is enabled is now batch_size * 500 during decoding to avoid
  users accidentally forgetting to increase the chunk size.

## [1.14.0]
### Changed
 - Downscaled fixed positional embeddings for CNN models.
 - Renamed `--monitor-bleu` flag to `--decode-and-evaluate` to illustrate that it computes
 other metrics in addition to BLEU.

### Added
 - `--decode-and-evaluate-use-cpu` flag to use CPU for decoding validation data.
 - `--decode-and-evaluate-device-id` flag to use a separate GPU device for validation decoding. If not specified, the
 existing and still default behavior is to use the last acquired GPU for training.

## [1.13.2]
### Added
 - A tool that extracts specified parameters from params.x into a .npz file for downstream applications or analysis.

## [1.13.1]
### Added
 - Added chrF metric
([Popovic 2015: chrF: character n-gram F-score for automatic MT evaluation](http://www.statmt.org/wmt15/pdf/WMT49.pdf)) to Sockeye.
sockeye.evaluate now accepts `bleu` and `chrf` as values for `--metrics`

## [1.13.0]
### Fixed
 - Transformer models do not ignore `--num-embed` anymore as they did silently before.
 As a result there is an error thrown if `--num-embed` != `--transformer-model-size`.
 - Fixed the attention in upper layers (`--rnn-attention-in-upper-layers`), which was previously not passed correctly
 to the decoder.
### Removed
 - Removed RNN parameter (un-)packing and support for FusedRNNCells (removed `--use-fused-rnns` flag).
 These were not used, not correctly initialized, and performed worse than regular RNN cells. Moreover,
 they made the code much more complex. RNN models trained with previous versions are no longer compatible.
- Removed the lexical biasing functionality (Arthur ETAL'16) (removed arguments `--lexical-bias`
 and `--learn-lexical-bias`).

## [1.12.2]
### Changed
 - Updated to [MXNet 0.12.1](https://github.com/apache/incubator-mxnet/releases/tag/0.12.1), which includes an important
 bug fix for CPU decoding.

## [1.12.1]
### Changed
 - Removed dependency on sacrebleu pip package. Now imports directly from `contrib/`.

## [1.12.0]
### Changed
 - Transformers now always use the linear output transformation after combining attention heads, even if input & output
 depth do not differ.

## [1.11.2]
### Fixed
 - Fixed a bug where vocabulary slice padding was defaulting to CPU context.  This was affecting decoding on GPUs with
 very small vocabularies.

## [1.11.1]
### Fixed
 - Fixed an issue with the use of `ignore` in `CrossEntropyMetric::cross_entropy_smoothed`. This was affecting
 runs with Eve optimizer and label smoothing. Thanks @kobenaxie for reporting.

## [1.11.0]
### Added
 - Lexicon-based target vocabulary restriction for faster decoding. New CLI for top-k lexicon creation, sockeye.lexicon.
 New translate CLI argument `--restrict-lexicon`.

### Changed
 - Bleu computation based on Sacrebleu.

## [1.10.5]
### Fixed
 - Fixed yet another bug with the data iterator.

## [1.10.4]
### Fixed
 - Fixed a bug with the revised data iterator not correctly appending EOS symbols for variable-length batches.
 This reverts part of the commit added in 1.10.1 but is now correct again.

## [1.10.3]
### Changed
 - Fixed a bug with max_observed_{source,target}_len being computed on the complete data set, not only on the
 sentences actually added to the buckets based on `--max_seq_len`.

## [1.10.2]
### Added
 - `--max-num-epochs` flag to train for a maximum number of passes through the training data.

## [1.10.1]
### Changed
 - Reduced memory footprint when creating data iterators: integer sequences
 are streamed from disk when being assigned to buckets.

## [1.10.0]
### Changed
 - Updated MXNet dependency to 0.12 (w/ MKL support by default).
 - Changed `--smoothed-cross-entropy-alpha` to `--label-smoothing`.
 Label smoothing should now require significantly less memory due to its addition to MXNet's `SoftmaxOutput` operator.
 - `--weight-normalization` now applies not only to convolutional weight matrices, but to output layers of all decoders.
 It is also independent of weight tying.
 - Transformers now use `--embed-dropout`. Before they were using `--transformer-dropout-prepost` for this.
 - Transformers now scale their embedding vectors before adding fixed positional embeddings.
 This turns out to be crucial for effective learning.
 - `.param` files now use 5 digit identifiers to reduce risk of overflowing with many checkpoints.

### Added
 - Added CUDA 9.0 requirements file.
 - `--loss-normalization-type`. Added a new flag to control loss normalization. New default is to normalize
 by the number of valid, non-PAD tokens instead of the batch size.
 - `--weight-init-xavier-factor-type`. Added new flag to control Xavier factor type when `--weight-init=xavier`.
 - `--embed-weight-init`. Added new flag for initialization of embeddings matrices.

### Removed
 - `--smoothed-cross-entropy-alpha` argument. See above.
 - `--normalize-loss` argument. See above.

## [1.9.0]
### Added
 - Batch decoding. New options for the translate CLI: ``--batch-size`` and ``--chunk-size``. Translator.translate()
 now accepts and returns lists of inputs and outputs.

## [1.8.4]
### Added
 - Exposing the MXNet KVStore through the ``--kvstore`` argument, potentially enabling distributed training.

## [1.8.3]
### Added
 - Optional smart rollback of parameters and optimizer states after updating the learning rate
 if not improved for x checkpoints. New flags: ``--learning-rate-decay-param-reset``,
 ``--learning-rate-decay-optimizer-states-reset``

## [1.8.2]
### Fixed
 - The RNN variational dropout mask is now independent of the input
 (previously any zero initial state led to the first state being canceled).
 - Correctly pass `self.dropout_inputs` float to `mx.sym.Dropout` in `VariationalDropoutCell`.

## [1.8.1]
### Changed
 - Instead of truncating sentences exceeding the maximum input length they are now translated in chunks.

## [1.8.0]
### Added
 - Convolutional decoder.
 - Weight normalization (for CNN only so far).
 - Learned positional embeddings for the transformer.

### Changed
 - `--attention-*` CLI params renamed to `--rnn-attention-*`.
 - `--transformer-no-positional-encodings` generalized to `--transformer-positional-embedding-type`.



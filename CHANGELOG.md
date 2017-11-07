# Changelog
All notable changes to this project will be documented in this file.

We use version numbers with three digits such as 1.0.0.
Any version bump in the last digit is backwards-compatible, in that a model trained with the previous version can still
be used for translation with the new version.
Any bump in the second digit indicates potential backwards incompatibilities, e.g. due to changing the architecture or
simply modifying weight names.
Note that Sockeye has checks in place to not translate with an old model that was trained with an incompatible version.

For each item we will potentially have subsections for: _Added_, _Changed_, _Removed_, _Deprecated_, and _Fixed_.

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


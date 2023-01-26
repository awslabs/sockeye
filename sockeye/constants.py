# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Defines various constants used throughout the project
"""
import torch as pt
import numpy as np

BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
PAD_ID = 0
PAD_FORMAT = "<pad%d>"
TOKEN_SEPARATOR = " "
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
UNK_ID = VOCAB_SYMBOLS.index(UNK_SYMBOL)
BOS_ID = VOCAB_SYMBOLS.index(BOS_SYMBOL)
EOS_ID = VOCAB_SYMBOLS.index(EOS_SYMBOL)
# reserve extra space for the EOS or BOS symbol that is added to both source and target
SPACE_FOR_XOS = 1

ARG_SEPARATOR = ":"

# If true, target factors are shifted to the right by 1 at training time, and unshifted in inference.
# TODO: make this configurable in the model, separately per target factor.
TARGET_FACTOR_SHIFT = True

ENCODER_PREFIX = "encoder"
DECODER_PREFIX = "decoder"
DEFAULT_OUTPUT_LAYER_PREFIX = "output_layer"
NVS_LAYER_PREFIX = "nvs"

# SSRU
SSRU_PREFIX = "ssru_"

# embedding prefixes
SOURCE_EMBEDDING_PREFIX = "embedding_source"
TARGET_EMBEDDING_PREFIX = "embedding_target"

# source factors
FACTORS_COMBINE_SUM = 'sum'
FACTORS_COMBINE_AVERAGE = 'average'
FACTORS_COMBINE_CONCAT = 'concat'
FACTORS_COMBINE_CHOICES = [FACTORS_COMBINE_SUM,
                           FACTORS_COMBINE_AVERAGE,
                           FACTORS_COMBINE_CONCAT]

# encoder names (arguments)
TRANSFORMER_TYPE = "transformer"

# available encoders
ENCODERS = [TRANSFORMER_TYPE]

# TODO replace options list (e.g ENCODERS, DECODERS, ...) with Enum classes
# available decoders
SSRU_TRANSFORMER = SSRU_PREFIX + TRANSFORMER_TYPE
DECODERS = [TRANSFORMER_TYPE, SSRU_TRANSFORMER]

# positional embeddings
NO_POSITIONAL_EMBEDDING = "none"
FIXED_POSITIONAL_EMBEDDING = "fixed"
LEARNED_POSITIONAL_EMBEDDING = "learned"
POSITIONAL_EMBEDDING_TYPES = [NO_POSITIONAL_EMBEDDING, FIXED_POSITIONAL_EMBEDDING, LEARNED_POSITIONAL_EMBEDDING]

DEFAULT_NUM_EMBED = 512

# weight tying components
WEIGHT_TYING_SRC = 'src'
WEIGHT_TYING_TRG = 'trg'
WEIGHT_TYING_SOFTMAX = 'softmax'
# weight tying types (combinations of above components):
WEIGHT_TYING_NONE = 'none'
WEIGHT_TYING_TRG_SOFTMAX = 'trg_softmax'
WEIGHT_TYING_SRC_TRG = 'src_trg'
WEIGHT_TYING_SRC_TRG_SOFTMAX = 'src_trg_softmax'
WEIGHT_TYING_TYPES = [WEIGHT_TYING_NONE, WEIGHT_TYING_SRC_TRG_SOFTMAX, WEIGHT_TYING_SRC_TRG, WEIGHT_TYING_TRG_SOFTMAX]

NVS_TYPE_LOGIT_MAX = "logit_max"
NVS_TYPE_EOS = "eos"
NVS_TYPES = [NVS_TYPE_LOGIT_MAX, NVS_TYPE_EOS]

# Activation types
RELU = "relu"
# Swish-1/SiLU (https://arxiv.org/pdf/1710.05941.pdf, https://arxiv.org/pdf/1702.03118.pdf)
SWISH1 = "swish1"
# Gaussian Error Linear Unit (https://arxiv.org/pdf/1606.08415.pdf)
GELU = "gelu"
TRANSFORMER_ACTIVATION_TYPES = [RELU, SWISH1, GELU]

# default I/O variable names
TARGET_LABEL_NAME = "target_label"
TARGET_FACTOR_LABEL_NAME = "target_factor%d_label"
LENRATIO_LABEL_NAME = "length_ratio_label"
LENRATIO_NAME = "length_ratio"

LOGITS_NAME = "logits"
FACTOR_LOGITS_NAME = "factor%d_logits"

NVS_PRED_NAME = "nvs_pred"

MEASURE_SPEED_EVERY = 50  # measure speed and metrics every X batches

# Inference constants
DEFAULT_BEAM_SIZE = 5
DEFAULT_NBEST_SIZE = 1
CHUNK_SIZE_NO_BATCHING = 1
CHUNK_SIZE_PER_BATCH_SEGMENT = 500
BEAM_SEARCH_STOP_FIRST = 'first'
BEAM_SEARCH_STOP_ALL = 'all'

# State structure constants
STEP_STATE = 's'
MASK_STATE = 'm'
ENCODER_STATE = 'e'
DECODER_STATE = 'd'

# Inference Input JSON constants
JSON_TEXT_KEY = "text"
JSON_FACTORS_KEY = "factors"
JSON_SOURCE_PREFIX_KEY = "source_prefix"
JSON_SOURCE_PREFIX_FACTORS_KEY = "source_prefix_factors"
JSON_TARGET_PREFIX_KEY = "target_prefix"
JSON_TARGET_PREFIX_FACTORS_KEY = "target_prefix_factors"
JSON_USE_TARGET_PREFIX_ALL_CHUNKS_KEY = "use_target_prefix_all_chunks"
JSON_KEEP_TARGET_PREFIX_KEY = "keep_target_prefix"
JSON_RESTRICT_LEXICON_KEY = "restrict_lexicon"
JSON_CONSTRAINTS_KEY = "constraints"
JSON_AVOID_KEY = "avoid"

VERSION_NAME = "version"
CONFIG_NAME = "config"
LOG_NAME = "log"
JSON_SUFFIX = ".json"
VOCAB_SRC_PREFIX = "vocab.src"
VOCAB_SRC_NAME = VOCAB_SRC_PREFIX + ".%d" + JSON_SUFFIX
VOCAB_TRG_PREFIX = "vocab.trg"
VOCAB_TRG_NAME = VOCAB_TRG_PREFIX + ".%d" + JSON_SUFFIX
VOCAB_ENCODING = "utf-8"
PARAMS_PREFIX = "params."
PARAMS_NAME = PARAMS_PREFIX + "%05d"
PARAMS_BEST_NAME = "params.best"
DECODE_OUT_NAME = "decode.output.{{factor}}.{checkpoint:05d}"
DECODE_IN_NAME = "decode.source.{factor}"
DECODE_REF_NAME = "decode.target.{factor}"
METRICS_NAME = "metrics"
TENSORBOARD_NAME = "tensorboard"
DEEPSPEED_LATEST_NAME = "latest"

# training resumption constants
TRAINING_STATE_DIRNAME = "training_state"
TRAINING_STATE_TEMP_DIRNAME = "tmp.training_state"
TRAINING_STATE_TEMP_DELETENAME = "delete.training_state"
TRAINING_STATE_DEEPSPEED = "deepspeed"

OPT_STATE_LAST = "optimizer_last.pkl"
OPT_STATE_BEST = "optimizer_best.pkl"

LR_SCHEDULER_LAST = "lr_scheduler_last.pkl"
LR_SCHEDULER_BEST = "lr_scheduler_best.pkl"

BUCKET_ITER_STATE_NAME = "bucket.pkl"
RNG_STATE_NAME = "rng.pkl"
TRAINING_STATE_NAME = "training.pkl"
GRAD_SCALER_STATE_NAME = "grad_scaler.pkl"
APEX_AMP_STATE_NAME = "apex_amp_state.pkl"
TRAINING_STATE_PARAMS_NAME = "params"
ARGS_STATE_NAME = "args.yaml"

# Arguments that may differ and still resume training
ARGS_MAY_DIFFER = ["device_id", "device_ids", "overwrite_output", "use_tensorboard", "quiet", "align_plot_prefix",
                   "sure_align_threshold", "keep_last_params", "seed", "max_updates", "min_updates", "max_num_epochs",
                   "min_num_epochs", "max_samples", "min_samples", "max_checkpoints", "max_seconds", "local_rank", "tf32"]

# Other argument constants
TRAINING_ARG_SOURCE = "--source"
TRAINING_ARG_TARGET = "--target"
TRAINING_ARG_PREPARED_DATA = "--prepared-data"
TRAINING_ARG_MAX_SEQ_LEN = "--max-seq-len"

VOCAB_ARG_SHARED_VOCAB = "--shared-vocab"

INFERENCE_ARG_INPUT_LONG = "--input"
INFERENCE_ARG_INPUT_SHORT = "-i"
INFERENCE_ARG_OUTPUT_LONG = "--output"
INFERENCE_ARG_OUTPUT_SHORT = "-o"
INFERENCE_ARG_INPUT_FACTORS_LONG = "--input-factors"
INFERENCE_ARG_INPUT_FACTORS_SHORT = "-if"
TRAIN_ARGS_MONITOR_BLEU = "--decode-and-evaluate"
TRAIN_ARGS_CHECKPOINT_INTERVAL = "--checkpoint-interval"
TRAIN_ARGS_STOP_ON_DECODER_FAILURE = "--stop-training-on-decoder-failure"

# Used to delimit factors on STDIN for inference
DEFAULT_FACTOR_DELIMITER = '|'

BATCH_TYPE_SENTENCE = "sentence"
BATCH_TYPE_WORD = "word"
BATCH_TYPE_MAX_WORD = "max-word"
BATCH_TYPES = [BATCH_TYPE_SENTENCE, BATCH_TYPE_WORD, BATCH_TYPE_MAX_WORD]

# Training constants
OPTIMIZER_ADAM = "adam"
OPTIMIZER_SGD = "sgd"
OPTIMIZERS = [OPTIMIZER_ADAM, OPTIMIZER_SGD]

LR_SCHEDULER_NONE = 'none'
LR_SCHEDULER_INV_SQRT_DECAY = 'inv-sqrt-decay'
LR_SCHEDULER_LINEAR_DECAY = 'linear-decay'
LR_SCHEDULER_PLATEAU_REDUCE = 'plateau-reduce'
LR_SCHEDULERS = [LR_SCHEDULER_NONE,
                 LR_SCHEDULER_INV_SQRT_DECAY,
                 LR_SCHEDULER_LINEAR_DECAY,
                 LR_SCHEDULER_PLATEAU_REDUCE]

GRADIENT_CLIPPING_TYPE_ABS = 'abs'
GRADIENT_CLIPPING_TYPE_NORM = 'norm'
GRADIENT_CLIPPING_TYPE_NONE = 'none'
GRADIENT_CLIPPING_TYPES = [GRADIENT_CLIPPING_TYPE_ABS, GRADIENT_CLIPPING_TYPE_NORM, GRADIENT_CLIPPING_TYPE_NONE]

DIST_ENV_LOCAL_RANK = 'LOCAL_RANK'
DIST_SECONDARY_WORKERS_LOGDIR = 'secondary_worker_logs'

# output handler
OUTPUT_HANDLER_TRANSLATION = "translation"
OUTPUT_HANDLER_TRANSLATION_WITH_SCORE = "translation_with_score"
OUTPUT_HANDLER_SCORE = "score"
OUTPUT_HANDLER_PAIR_WITH_SCORE = "pair_with_score"
OUTPUT_HANDLER_BENCHMARK = "benchmark"
OUTPUT_HANDLER_JSON = "json"
OUTPUT_HANDLER_TRANSLATION_WITH_FACTORS = "translation_with_factors"
OUTPUT_HANDLERS = [OUTPUT_HANDLER_TRANSLATION,
                   OUTPUT_HANDLER_SCORE,
                   OUTPUT_HANDLER_TRANSLATION_WITH_SCORE,
                   OUTPUT_HANDLER_TRANSLATION_WITH_FACTORS,
                   OUTPUT_HANDLER_BENCHMARK,
                   OUTPUT_HANDLER_JSON]
OUTPUT_HANDLERS_SCORING = [OUTPUT_HANDLER_SCORE,
                           OUTPUT_HANDLER_PAIR_WITH_SCORE]

# metrics
ACCURACY = 'accuracy'
PERPLEXITY = 'perplexity'
PERPLEXITY_SHORT_NAME = 'ppl'
LENRATIO_MSE = 'length-ratio-mse'
BLEU = 'bleu'
CHRF = 'chrf'
ROUGE1 = 'rouge1'
ROUGE2 = 'rouge2'
ROUGEL = 'rougel'
BOW_PERPLEXITY = 'bow-perplexity'
TER = 'ter'
LENRATIO = 'length-ratio-mse'
AVG_TIME = "avg-sec-per-sent"
DECODING_TIME = "decode-walltime"
METRICS = [PERPLEXITY, ACCURACY, LENRATIO_MSE, BLEU, CHRF, ROUGE1, BOW_PERPLEXITY, TER]
METRIC_MAXIMIZE = {ACCURACY: True, BLEU: True, CHRF: True, ROUGE1: True, PERPLEXITY: False, LENRATIO_MSE: False,
                   TER: False, BOW_PERPLEXITY: False}
METRIC_WORST = {ACCURACY: 0.0, BLEU: 0.0, CHRF: 0.0, ROUGE1: 0.0, PERPLEXITY: np.inf, BOW_PERPLEXITY: np.inf, TER: np.inf}
METRICS_REQUIRING_DECODER = [BLEU, CHRF, ROUGE1, ROUGE2, ROUGEL, TER]
EVALUATE_METRICS = [BLEU, CHRF, ROUGE1, ROUGE2, ROUGEL, TER]

# loss
CROSS_ENTROPY = 'cross-entropy'
BINARY_CROSS_ENTROPY = 'binary-cross-entropy'
LINK_NORMAL = 'normal'
LINK_POISSON = 'poisson'
LENGTH_TASK_RATIO = 'ratio'
LENGTH_TASK_LENGTH = 'length'

TARGET_MAX_LENGTH_FACTOR = 2
DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH = 2

DTYPE_BF16 = 'bfloat16'
DTYPE_FP16 = 'float16'
DTYPE_FP32 = 'float32'
DTYPE_INT8 = 'int8'
DTYPE_INT16 = 'int16'
DTYPE_INT32 = 'int32'
LARGE_POSITIVE_VALUE = 99999999.
LARGE_VALUES = {
    # Rounds to 1.0014e+08
    # https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Range_and_precision
    DTYPE_BF16: LARGE_POSITIVE_VALUE,
    pt.bfloat16: LARGE_POSITIVE_VALUE,

    # Something at the middle of 32768<x<65519. Will be rounded to a multiple of 32.
    # https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_integer_values
    DTYPE_FP16: 49152.0,
    np.float16: 49152.0,
    pt.float16: 49152.0,

    # Will be rounded to 1.0e8.
    # https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Precision_limitations_on_integer_values
    DTYPE_FP32: LARGE_POSITIVE_VALUE,
    np.float32: LARGE_POSITIVE_VALUE,
    pt.float32: LARGE_POSITIVE_VALUE,
    # with --tf32, rounds to 0b1011111011 * 1024 * 128 (10 bits precision) = 1.00007e8

    # Rounds to 1.0014e+08
    DTYPE_BF16: LARGE_POSITIVE_VALUE,
    pt.bfloat16: LARGE_POSITIVE_VALUE,
}

# lhuc application points
LHUC_ENCODER = "encoder"
LHUC_DECODER = "decoder"
LHUC_ALL = "all"
LHUC_CHOICES = [LHUC_ENCODER, LHUC_DECODER, LHUC_ALL]

# Strategies for fixing various parameters.
FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER = "all_except_decoder"
FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTER_LAYERS = "all_except_outer_layers"
FIXED_PARAM_STRATEGY_ALL_EXCEPT_EMBEDDINGS = "all_except_embeddings"
FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTPUT_PROJ = "all_except_output_proj"
FIXED_PARAM_STRATEGY_ALL_EXCEPT_FEED_FORWARD = "all_except_feed_forward"
FIXED_PARAM_STRATEGY_ENCODER_AND_SOURCE_EMBEDDINGS = "encoder_and_source_embeddings"
FIXED_PARAM_STRATEGY_ENCODER_HALF_AND_SOURCE_EMBEDDINGS = "encoder_half_and_source_embeddings"

FIXED_PARAM_STRATEGY_CHOICES = [FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER,
                                FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTER_LAYERS,
                                FIXED_PARAM_STRATEGY_ALL_EXCEPT_EMBEDDINGS,
                                FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTPUT_PROJ,
                                FIXED_PARAM_STRATEGY_ALL_EXCEPT_FEED_FORWARD,
                                FIXED_PARAM_STRATEGY_ENCODER_AND_SOURCE_EMBEDDINGS,
                                FIXED_PARAM_STRATEGY_ENCODER_HALF_AND_SOURCE_EMBEDDINGS]

# data sharding
SHARD_NAME = "shard.%05d"
SHARD_SOURCE = SHARD_NAME + ".source"
SHARD_TARGET = SHARD_NAME + ".target"
DATA_INFO = "data.info"
DATA_CONFIG = "data.config"
PREPARED_DATA_VERSION_FILE = "data.version"
PREPARED_DATA_VERSION = 6

# reranking metric options
RERANK_BLEU = "bleu"
RERANK_CHRF = "chrf"
RERANK_ISOMETRIC = "isometric"
RERANK_ISOMETRIC_RATIO = RERANK_ISOMETRIC + "-ratio"
RERANK_ISOMETRIC_DIFF = RERANK_ISOMETRIC + "-diff"
RERANK_ISOMETRIC_LC = RERANK_ISOMETRIC + "-lc"
RERANK_METRICS = [RERANK_BLEU, RERANK_CHRF,
                  RERANK_ISOMETRIC_RATIO, RERANK_ISOMETRIC_DIFF, RERANK_ISOMETRIC_LC]

# token segmentation symbols
TOKEN_SEGMENTATION_SW = "@@"  # sub-word nmt
TOKEN_SEGMENTATION_SP = "▁"   # sentence piece
TOKEN_SEGMENTATION_MARKERS = [TOKEN_SEGMENTATION_SW, TOKEN_SEGMENTATION_SP]

# sequence length count types
SEQ_LEN_IN_CHARACTERS = "char"
SEQ_LEN_IN_TOKENS = "token"

# scoring
SCORING_TYPE_NEGLOGPROB = 'neglogprob'
SCORING_TYPE_LOGPROB = 'logprob'
SCORING_TYPE_DEFAULT = SCORING_TYPE_NEGLOGPROB
SCORING_TYPE_CHOICES = [SCORING_TYPE_NEGLOGPROB, SCORING_TYPE_LOGPROB]

# parameter averaging
AVERAGE_BEST = 'best'
AVERAGE_LAST = 'last'
AVERAGE_LIFESPAN = 'lifespan'
AVERAGE_CHOICES = [AVERAGE_BEST, AVERAGE_LAST, AVERAGE_LIFESPAN]

# brevity penalty
BREVITY_PENALTY_CONSTANT = 'constant'
BREVITY_PENALTY_LEARNED = 'learned'
BREVITY_PENALTY_NONE = 'none'

# k-nn
KNN_STATE_DATA_STORE_NAME = "keys.npy"
KNN_WORD_DATA_STORE_NAME = "vals.npy"
KNN_WORD_DATA_STORE_DTYPE = DTYPE_INT32
KNN_CONFIG_NAME = "config.yaml"
KNN_INDEX_NAME = "key_index"
KNN_EPSILON = 1e-6
DEFAULT_DATA_STORE_BLOCK_SIZE = 1024 * 1024
DEFAULT_KNN_LAMBDA = 0.8

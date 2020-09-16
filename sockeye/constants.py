# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sys

import mxnet as mx
import numpy as np

# MXNet environment variables
MXNET_SAFE_ACCUMULATION = 'MXNET_SAFE_ACCUMULATION'

# Horovod environment variables
HOROVOD_HIERARCHICAL_ALLREDUCE = 'HOROVOD_HIERARCHICAL_ALLREDUCE'
HOROVOD_HIERARCHICAL_ALLGATHER = 'HOROVOD_HIERARCHICAL_ALLGATHER'

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

ENCODER_PREFIX = "encoder_"
DECODER_PREFIX = "decoder_"
EMBEDDING_PREFIX = "embed_"
ATTENTION_PREFIX = "att_"
COVERAGE_PREFIX = "cov_"
TRANSFORMER_ENCODER_PREFIX = ENCODER_PREFIX + "transformer_"
DEFAULT_OUTPUT_LAYER_PREFIX = "target_output_"
LENRATIOS_OUTPUT_LAYER_PREFIX = "length_ratio_layer_"

# SSRU
SSRU_PREFIX = "ssru_"

# embedding prefixes
SOURCE_EMBEDDING_PREFIX = "source_" + EMBEDDING_PREFIX
SOURCE_POSITIONAL_EMBEDDING_PREFIX = "source_pos_" + EMBEDDING_PREFIX
TARGET_EMBEDDING_PREFIX = "target_" + EMBEDDING_PREFIX
TARGET_POSITIONAL_EMBEDDING_PREFIX = "target_pos_" + EMBEDDING_PREFIX
SHARED_EMBEDDING_PREFIX = "source_target_" + EMBEDDING_PREFIX

# source factors
SOURCE_FACTORS_COMBINE_SUM = 'sum'
SOURCE_FACTORS_COMBINE_AVERAGE = 'average'
SOURCE_FACTORS_COMBINE_CONCAT = 'concat'
SOURCE_FACTORS_COMBINE_CHOICES = [SOURCE_FACTORS_COMBINE_SUM,
                                  SOURCE_FACTORS_COMBINE_AVERAGE,
                                  SOURCE_FACTORS_COMBINE_CONCAT]

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

DEFAULT_INIT_PATTERN = ".*"

# init types
INIT_XAVIER = 'xavier'
INIT_UNIFORM = 'uniform'
INIT_TYPES = [INIT_XAVIER, INIT_UNIFORM]

INIT_XAVIER_FACTOR_TYPE_IN = "in"
INIT_XAVIER_FACTOR_TYPE_OUT = "out"
INIT_XAVIER_FACTOR_TYPE_AVG = "avg"
INIT_XAVIER_FACTOR_TYPES = [INIT_XAVIER_FACTOR_TYPE_IN, INIT_XAVIER_FACTOR_TYPE_OUT, INIT_XAVIER_FACTOR_TYPE_AVG]

RAND_TYPE_UNIFORM = 'uniform'
RAND_TYPE_GAUSSIAN = 'gaussian'

# Embedding init types
EMBED_INIT_PATTERN = '(%s|%s|%s)weight' % (SOURCE_EMBEDDING_PREFIX, TARGET_EMBEDDING_PREFIX, SHARED_EMBEDDING_PREFIX)
EMBED_INIT_DEFAULT = 'default'
EMBED_INIT_NORMAL = 'normal'
EMBED_INIT_TYPES = [EMBED_INIT_DEFAULT, EMBED_INIT_NORMAL]
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

# default decoder prefixes
TRANSFORMER_DECODER_PREFIX = DECODER_PREFIX + "transformer_"
TRANSFORMER_SSRU_DECODER_PREFIX = DECODER_PREFIX + SSRU_TRANSFORMER

# Activation types
RELU = "relu"
# Swish-1/SiLU (https://arxiv.org/pdf/1710.05941.pdf, https://arxiv.org/pdf/1702.03118.pdf)
SWISH1 = "swish1"
# Gaussian Error Linear Unit (https://arxiv.org/pdf/1606.08415.pdf)
GELU = "gelu"
TRANSFORMER_ACTIVATION_TYPES = [RELU, SWISH1, GELU]

# default I/O variable names
SOURCE_NAME = "source"
SOURCE_LENGTH_NAME = "source_length"
TARGET_NAME = "target"
TARGET_LABEL_NAME = "target_label"
LENRATIO_LABEL_NAME = "length_ratio_label"
LENRATIO_LABEL_OUTPUT_NAME = "length_ratio_label" + "_output"
LENRATIO_NAME = "length_ratio"
LENRATIO_LOSS_NAME = LENRATIO_NAME + "_loss"
LENRATIO_OUTPUT_NAME = LENRATIO_NAME + "_output"
LEXICON_NAME = "lexicon"

SOURCE_ENCODED_NAME = "encoded_source"
TARGET_PREVIOUS_NAME = "prev_target_word_id"
HIDDEN_PREVIOUS_NAME = "prev_hidden"
SOURCE_DYNAMIC_PREVIOUS_NAME = "prev_dynamic_source"

LOGIT_INPUTS_NAME = "logit_inputs"
LOGITS_NAME = "logits"
SOFTMAX_NAME = "softmax"
SOFTMAX_OUTPUT_NAME = SOFTMAX_NAME + "_output"

MEASURE_SPEED_EVERY = 50  # measure speed and metrics every X batches

# Monitor constants
STAT_FUNC_DEFAULT = "mx_default"  # default MXNet monitor stat func: mx.nd.norm(x)/mx.nd.sqrt(x.size)
STAT_FUNC_MAX = 'max'
STAT_FUNC_MIN = 'min'
STAT_FUNC_MEAN = 'mean'
MONITOR_STAT_FUNCS = {STAT_FUNC_DEFAULT: None,
                      STAT_FUNC_MAX: lambda x: mx.nd.max(x),
                      STAT_FUNC_MEAN: lambda x: mx.nd.mean(x)}

# Inference constants
DEFAULT_BEAM_SIZE = 5
DEFAULT_NBEST_SIZE = 1
CHUNK_SIZE_NO_BATCHING = 1
CHUNK_SIZE_PER_BATCH_SEGMENT = 500
BEAM_SEARCH_STOP_FIRST = 'first'
BEAM_SEARCH_STOP_ALL = 'all'

# State structure constants
STEP_STATE = 's'
BIAS_STATE = 'b'
ENCODER_STATE = 'e'
DECODER_STATE = 'd'

# Inference Input JSON constants
JSON_TEXT_KEY = "text"
JSON_FACTORS_KEY = "factors"
JSON_RESTRICT_LEXICON_KEY = "restrict_lexicon"
JSON_CONSTRAINTS_KEY = "constraints"
JSON_AVOID_KEY = "avoid"
JSON_ENCODING = "utf-8"

# Lexical constraints
BANK_ADJUSTMENT = 'even'

VERSION_NAME = "version"
CONFIG_NAME = "config"
CONFIG_NAME_FLOAT32 = CONFIG_NAME + ".float32"
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
PARAMS_BEST_NAME_FLOAT32 = PARAMS_BEST_NAME + ".float32"
DECODE_OUT_NAME = "decode.output.%05d"
DECODE_IN_NAME = "decode.source.%d"
DECODE_REF_NAME = "decode.target"
SYMBOL_NAME = "symbol" + JSON_SUFFIX
METRICS_NAME = "metrics"
TENSORBOARD_NAME = "tensorboard"

# training resumption constants
TRAINING_STATE_DIRNAME = "training_state"
TRAINING_STATE_TEMP_DIRNAME = "tmp.training_state"
TRAINING_STATE_TEMP_DELETENAME = "delete.training_state"
OPT_STATES_LAST = "mx_optimizer_last.pkl"
OPT_STATES_BEST = "mx_optimizer_best.pkl"
OPT_STATES_INITIAL = "mx_optimizer_initial.pkl"
BUCKET_ITER_STATE_NAME = "bucket.pkl"
RNG_STATE_NAME = "rng.pkl"
TRAINING_STATE_NAME = "training.pkl"
AMP_LOSS_SCALER_STATE_NAME = "amp_loss_scaler.pkl"
SCHEDULER_STATE_NAME = "scheduler.pkl"
TRAINING_STATE_PARAMS_NAME = "params"
ARGS_STATE_NAME = "args.yaml"

# Arguments that may differ and still resume training
ARGS_MAY_DIFFER = ["overwrite_output", "use_tensorboard", "quiet",
                   "align_plot_prefix", "sure_align_threshold",
                   "keep_last_params", "seed",
                   "max_updates", "min_updates",
                   "max_num_epochs", "min_num_epochs",
                   "max_samples", "min_samples", "max_checkpoints", "max_seconds"]

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
TRAIN_ARGS_CHECKPOINT_FREQUENCY = "--checkpoint-frequency"
TRAIN_ARGS_STOP_ON_DECODER_FAILURE = "--stop-training-on-decoder-failure"

# Used to delimit factors on STDIN for inference
DEFAULT_FACTOR_DELIMITER = '|'

BATCH_TYPE_SENTENCE = "sentence"
BATCH_TYPE_WORD = "word"
BATCH_TYPE_MAX_WORD = "max-word"
BATCH_TYPES = [BATCH_TYPE_SENTENCE, BATCH_TYPE_WORD, BATCH_TYPE_MAX_WORD]

KVSTORE_DEVICE = "device"
KVSTORE_LOCAL = "local"
KVSTORE_SYNC = "dist_sync"
KVSTORE_DIST_DEVICE_SYNC = "dist_device_sync"
KVSTORE_DIST_ASYNC = "dist_async"
KVSTORE_NCCL = 'nccl'
KVSTORE_TYPES = [KVSTORE_DEVICE, KVSTORE_LOCAL, KVSTORE_SYNC,
                 KVSTORE_DIST_DEVICE_SYNC, KVSTORE_DIST_ASYNC,
                 KVSTORE_NCCL]

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

GRADIENT_COMPRESSION_NONE = None
GRADIENT_COMPRESSION_2BIT = "2bit"
GRADIENT_COMPRESSION_TYPES = [GRADIENT_CLIPPING_TYPE_NONE, GRADIENT_COMPRESSION_2BIT]

HOROVOD_SECONDARY_WORKERS_DIRNAME = 'secondary_workers'

# output handler
OUTPUT_HANDLER_TRANSLATION = "translation"
OUTPUT_HANDLER_TRANSLATION_WITH_SCORE = "translation_with_score"
OUTPUT_HANDLER_SCORE = "score"
OUTPUT_HANDLER_PAIR_WITH_SCORE = "pair_with_score"
OUTPUT_HANDLER_BENCHMARK = "benchmark"
OUTPUT_HANDLER_BEAM_STORE = "beam_store"
OUTPUT_HANDLER_JSON = "json"
OUTPUT_HANDLERS = [OUTPUT_HANDLER_TRANSLATION,
                   OUTPUT_HANDLER_SCORE,
                   OUTPUT_HANDLER_TRANSLATION_WITH_SCORE,
                   OUTPUT_HANDLER_BENCHMARK,
                   OUTPUT_HANDLER_JSON]
OUTPUT_HANDLERS_SCORING = [OUTPUT_HANDLER_SCORE,
                           OUTPUT_HANDLER_PAIR_WITH_SCORE]

# metrics
ACCURACY = 'accuracy'
PERPLEXITY = 'perplexity'
LENRATIO_MSE = 'length-ratio-mse'
BLEU = 'bleu'
CHRF = 'chrf'
ROUGE = 'rouge'
ROUGE1 = 'rouge1'
ROUGE2 = 'rouge2'
ROUGEL = 'rougel'
LENRATIO = 'length-ratio-mse'
AVG_TIME = "avg-sec-per-sent"
DECODING_TIME = "decode-walltime"
METRICS = [PERPLEXITY, ACCURACY, LENRATIO_MSE, BLEU, CHRF, ROUGE1]
METRIC_MAXIMIZE = {ACCURACY: True, BLEU: True, CHRF: True, ROUGE1: True, PERPLEXITY: False}
METRIC_WORST = {ACCURACY: 0.0, BLEU: 0.0, CHRF: 0.0, ROUGE1: 0.0, PERPLEXITY: np.inf}
METRICS_REQUIRING_DECODER = [BLEU, CHRF, ROUGE1, ROUGE2, ROUGEL]
EVALUATE_METRICS = [BLEU, CHRF, ROUGE1, ROUGE2, ROUGEL]

# loss
CROSS_ENTROPY = 'cross-entropy'
CROSS_ENTROPY_WITOUT_SOFTMAX_OUTPUT = 'cross-entropy-without-softmax-output'
LENRATIO_REGRESSION = 'length-ratio-regression'

LINK_NORMAL = 'normal'
LINK_POISSON = 'poisson'
LENGTH_TASK_RATIO = 'ratio'
LENGTH_TASK_LENGTH = 'length'

LOSS_NORM_BATCH = 'batch'
LOSS_NORM_VALID = "valid"

TARGET_MAX_LENGTH_FACTOR = 2
DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH = 2

DTYPE_INT8 = 'int8'
DTYPE_FP16 = 'float16'
DTYPE_FP32 = 'float32'
LARGE_POSITIVE_VALUE = 99999999.
LARGE_NEGATIVE_VALUE = -LARGE_POSITIVE_VALUE
LARGE_VALUES = {
    # Something at the middle of 32768<x<65519. Will be rounded to a multiple of 32.
    # https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_integer_values
    DTYPE_FP16: 49152.0,
    np.float16: 49152.0,

    # Will be rounded to 1.0e8.
    # https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Precision_limits_on_integer_values.
    DTYPE_FP32: LARGE_POSITIVE_VALUE,
    np.float32: LARGE_POSITIVE_VALUE
}
LARGEST_INT = sys.maxsize

# see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
# TODO: better to use dynamic loss scaling for FP16, but unclear how to do this with SoftmaxOutput loss for CE.
FIXED_GRAD_SCALE_FP16 = 1024.0

LHUC_PREFIX = "lhuc_"
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
PREPARED_DATA_VERSION = 3

# reranking
RERANK_BLEU = "bleu"
RERANK_CHRF = "chrf"
RERANK_METRICS = [RERANK_BLEU, RERANK_CHRF]

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

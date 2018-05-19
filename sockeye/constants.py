# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Defines various constants used througout the project
"""
import mxnet as mx
import numpy as np

BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
PAD_ID = 0
TOKEN_SEPARATOR = " "
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
# reserve extra space for the EOS or BOS symbol that is added to both source and target
SPACE_FOR_XOS = 1

ARG_SEPARATOR = ":"

ENCODER_PREFIX = "encoder_"
DECODER_PREFIX = "decoder_"
EMBEDDING_PREFIX = "embed_"
ATTENTION_PREFIX = "att_"
COVERAGE_PREFIX = "cov_"
BIDIRECTIONALRNN_PREFIX = ENCODER_PREFIX + "birnn_"
STACKEDRNN_PREFIX = ENCODER_PREFIX + "rnn_"
FORWARD_PREFIX = "forward_"
REVERSE_PREFIX = "reverse_"
TRANSFORMER_ENCODER_PREFIX = ENCODER_PREFIX + "transformer_"
CNN_ENCODER_PREFIX = ENCODER_PREFIX + "cnn_"
CHAR_SEQ_ENCODER_PREFIX = ENCODER_PREFIX + "char_"
DEFAULT_OUTPUT_LAYER_PREFIX = "target_output_"

# embedding prefixes
SOURCE_EMBEDDING_PREFIX = "source_" + EMBEDDING_PREFIX
SOURCE_POSITIONAL_EMBEDDING_PREFIX = "source_pos_" + EMBEDDING_PREFIX
TARGET_EMBEDDING_PREFIX = "target_" + EMBEDDING_PREFIX
TARGET_POSITIONAL_EMBEDDING_PREFIX = "target_pos_" + EMBEDDING_PREFIX
SHARED_EMBEDDING_PREFIX = "source_target_" + EMBEDDING_PREFIX

# encoder names (arguments)
RNN_NAME = "rnn"
RNN_WITH_CONV_EMBED_NAME = "rnn-with-conv-embed"
TRANSFORMER_TYPE = "transformer"
CONVOLUTION_TYPE = "cnn"
TRANSFORMER_WITH_CONV_EMBED_TYPE = "transformer-with-conv-embed"

# available encoders
ENCODERS = [RNN_NAME, RNN_WITH_CONV_EMBED_NAME, TRANSFORMER_TYPE, TRANSFORMER_WITH_CONV_EMBED_TYPE, CONVOLUTION_TYPE]

# available decoder
DECODERS = [RNN_NAME, TRANSFORMER_TYPE, CONVOLUTION_TYPE]

# rnn types
LSTM_TYPE = 'lstm'
LNLSTM_TYPE = 'lnlstm'
LNGLSTM_TYPE = 'lnglstm'
GRU_TYPE = 'gru'
LNGRU_TYPE = 'lngru'
LNGGRU_TYPE = 'lnggru'
CELL_TYPES = [LSTM_TYPE, LNLSTM_TYPE, LNGLSTM_TYPE, GRU_TYPE, LNGRU_TYPE, LNGGRU_TYPE]

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

# RNN init types
RNN_INIT_PATTERN = ".*h2h.*"
RNN_INIT_ORTHOGONAL = 'orthogonal'
RNN_INIT_ORTHOGONAL_STACKED = 'orthogonal_stacked'
# use the default initializer used also for all other weights
RNN_INIT_DEFAULT = 'default'

# RNN decoder state init types
RNN_DEC_INIT_ZERO = "zero"
RNN_DEC_INIT_LAST = "last"
RNN_DEC_INIT_AVG = "avg"
RNN_DEC_INIT_CHOICES = [RNN_DEC_INIT_ZERO, RNN_DEC_INIT_LAST, RNN_DEC_INIT_AVG]

# attention types
ATT_BILINEAR = 'bilinear'
ATT_DOT = 'dot'
ATT_MH_DOT = 'mhdot'
ATT_FIXED = 'fixed'
ATT_LOC = 'location'
ATT_MLP = 'mlp'
ATT_COV = "coverage"
ATT_TYPES = [ATT_BILINEAR, ATT_DOT, ATT_MH_DOT, ATT_FIXED, ATT_LOC, ATT_MLP, ATT_COV]

# weight tying components
WEIGHT_TYING_SRC = 'src'
WEIGHT_TYING_TRG = 'trg'
WEIGHT_TYING_SOFTMAX = 'softmax'
# weight tying types (combinations of above components):
WEIGHT_TYING_TRG_SOFTMAX = 'trg_softmax'
WEIGHT_TYING_SRC_TRG = 'src_trg'
WEIGHT_TYING_SRC_TRG_SOFTMAX = 'src_trg_softmax'

# default decoder prefixes
RNN_DECODER_PREFIX = DECODER_PREFIX + "rnn_"
TRANSFORMER_DECODER_PREFIX = DECODER_PREFIX + "transformer_"
CNN_DECODER_PREFIX = DECODER_PREFIX + "cnn_"

# Activation types
# Gaussian Error Linear Unit (https://arxiv.org/pdf/1606.08415.pdf)
GELU = "gelu"
# Gated Linear Unit (https://arxiv.org/pdf/1705.03122.pdf)
GLU = "glu"
RELU = "relu"
SIGMOID = "sigmoid"
SOFT_RELU = "softrelu"
# Swish-1/SiLU (https://arxiv.org/pdf/1710.05941.pdf, https://arxiv.org/pdf/1702.03118.pdf)
SWISH1 = "swish1"
TANH = "tanh"
TRANSFORMER_ACTIVATION_TYPES = [GELU, RELU, SWISH1]
CNN_ACTIVATION_TYPES = [GLU, RELU, SIGMOID, SOFT_RELU, TANH]

# Convolutional block pad types:
CNN_PAD_LEFT = "left"
CNN_PAD_CENTERED = "centered"

# default I/O variable names
SOURCE_NAME = "source"
SOURCE_LENGTH_NAME = "source_length"
TARGET_NAME = "target"
TARGET_LABEL_NAME = "target_label"
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
CHUNK_SIZE_NO_BATCHING = 1
CHUNK_SIZE_PER_BATCH_SEGMENT = 500
BEAM_SEARCH_STOP_FIRST = 'first'
BEAM_SEARCH_STOP_ALL = 'all'

# Inference Input JSON constants
JSON_TEXT_KEY = "text"
JSON_FACTORS_KEY = "factors"
JSON_ENCODING = "utf-8"

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
SCHEDULER_STATE_NAME = "scheduler.pkl"
TRAINING_STATE_PARAMS_NAME = "params"
ARGS_STATE_NAME = "args.yaml"

# Arguments that may differ and still resume training
ARGS_MAY_DIFFER = ["overwrite_output", "use-tensorboard", "quiet",
                   "align_plot_prefix", "sure_align_threshold",
                   "keep_last_params"]

# Other argument constants
TRAINING_ARG_SOURCE = "--source"
TRAINING_ARG_TARGET = "--target"
TRAINING_ARG_PREPARED_DATA = "--prepared-data"

VOCAB_ARG_SHARED_VOCAB = "--shared-vocab"

INFERENCE_ARG_INPUT_LONG = "--input"
INFERENCE_ARG_INPUT_SHORT = "-i"
INFERENCE_ARG_OUTPUT_LONG = "--output"
INFERENCE_ARG_OUTPUT_SHORT = "-o"
INFERENCE_ARG_INPUT_FACTORS_LONG = "--input-factors"
INFERENCE_ARG_INPUT_FACTORS_SHORT = "-if"
TRAIN_ARGS_MONITOR_BLEU = "--decode-and-evaluate"
TRAIN_ARGS_CHECKPOINT_FREQUENCY = "--checkpoint-frequency"

# Used to delimit factors on STDIN for inference
DEFAULT_FACTOR_DELIMITER = '|'

# data layout strings
BATCH_MAJOR = "NTC"
TIME_MAJOR = "TNC"

BATCH_TYPE_SENTENCE = "sentence"
BATCH_TYPE_WORD = "word"

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
OPTIMIZER_EVE = "eve"
OPTIMIZER_NADAM = "nadam"
OPTIMIZER_RMSPROP = "rmsprop"
OPTIMIZER_SGD = "sgd"
OPTIMIZER_NAG = "nag"
OPTIMIZER_ADAGRAD = "adagrad"
OPTIMIZER_ADADELTA = "adadelta"
OPTIMIZERS = [OPTIMIZER_ADAM, OPTIMIZER_EVE, OPTIMIZER_NADAM, OPTIMIZER_RMSPROP, OPTIMIZER_SGD, OPTIMIZER_NAG,
              OPTIMIZER_ADAGRAD, OPTIMIZER_ADADELTA]

LR_SCHEDULER_FIXED_RATE_INV_SQRT_T = "fixed-rate-inv-sqrt-t"
LR_SCHEDULER_FIXED_RATE_INV_T = "fixed-rate-inv-t"
LR_SCHEDULER_FIXED_STEP = "fixed-step"
LR_SCHEDULER_PLATEAU_REDUCE = "plateau-reduce"
LR_SCHEDULERS = [LR_SCHEDULER_FIXED_RATE_INV_SQRT_T,
                 LR_SCHEDULER_FIXED_RATE_INV_T,
                 LR_SCHEDULER_FIXED_STEP,
                 LR_SCHEDULER_PLATEAU_REDUCE]

LR_DECAY_OPT_STATES_RESET_OFF = 'off'
LR_DECAY_OPT_STATES_RESET_INITIAL = 'initial'
LR_DECAY_OPT_STATES_RESET_BEST = 'best'
LR_DECAY_OPT_STATES_RESET_CHOICES = [LR_DECAY_OPT_STATES_RESET_OFF,
                                     LR_DECAY_OPT_STATES_RESET_INITIAL,
                                     LR_DECAY_OPT_STATES_RESET_BEST]

GRADIENT_CLIPPING_TYPE_ABS = 'abs'
GRADIENT_CLIPPING_TYPE_NORM = 'norm'
GRADIENT_CLIPPING_TYPE_NONE = 'none'
GRADIENT_CLIPPING_TYPES = [GRADIENT_CLIPPING_TYPE_ABS, GRADIENT_CLIPPING_TYPE_NORM, GRADIENT_CLIPPING_TYPE_NONE]

GRADIENT_COMPRESSION_NONE = None
GRADIENT_COMPRESSION_2BIT = "2bit"
GRADIENT_COMPRESSION_TYPES = [GRADIENT_CLIPPING_TYPE_NONE, GRADIENT_COMPRESSION_2BIT]

# output handler
OUTPUT_HANDLER_TRANSLATION = "translation"
OUTPUT_HANDLER_TRANSLATION_WITH_SCORE = "translation_with_score"
OUTPUT_HANDLER_TRANSLATION_WITH_ALIGNMENTS = "translation_with_alignments"
OUTPUT_HANDLER_TRANSLATION_WITH_ALIGNMENT_MATRIX = "translation_with_alignment_matrix"
OUTPUT_HANDLER_BENCHMARK = "benchmark"
OUTPUT_HANDLER_ALIGN_PLOT = "align_plot"
OUTPUT_HANDLER_ALIGN_TEXT = "align_text"
OUTPUT_HANDLER_BEAM_STORE = "beam_store"
OUTPUT_HANDLERS = [OUTPUT_HANDLER_TRANSLATION,
                   OUTPUT_HANDLER_TRANSLATION_WITH_SCORE,
                   OUTPUT_HANDLER_TRANSLATION_WITH_ALIGNMENTS,
                   OUTPUT_HANDLER_TRANSLATION_WITH_ALIGNMENT_MATRIX,
                   OUTPUT_HANDLER_BENCHMARK,
                   OUTPUT_HANDLER_ALIGN_PLOT,
                   OUTPUT_HANDLER_ALIGN_TEXT,
                   OUTPUT_HANDLER_BEAM_STORE]

# metrics
ACCURACY = 'accuracy'
PERPLEXITY = 'perplexity'
BLEU = 'bleu'
CHRF = 'chrf'
BLEU_VAL = BLEU + "-val"
CHRF_VAL = CHRF + "-val"
AVG_TIME = "avg-sec-per-sent-val"
DECODING_TIME = "decode-walltime-val"
METRICS = [PERPLEXITY, ACCURACY, BLEU]
METRIC_MAXIMIZE = {ACCURACY: True, BLEU: True, PERPLEXITY: False}
METRIC_WORST = {ACCURACY: 0.0, BLEU: 0.0, PERPLEXITY: np.inf}

# loss
CROSS_ENTROPY = 'cross-entropy'

LOSS_NORM_BATCH = 'batch'
LOSS_NORM_VALID = "valid"

TARGET_MAX_LENGTH_FACTOR = 2
DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH = 2

DTYPE_FP16 = 'float16'
DTYPE_FP32 = 'float32'
LARGE_POSITIVE_VALUE = 99999999.
LARGE_NEGATIVE_VALUE = -LARGE_POSITIVE_VALUE
LARGE_VALUES = {
    # Something at the middle of 32768<x<65519. Will be rounded to a multiple of 32.
    # https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_integer_values
    DTYPE_FP16: 49152.0,

    # Will be rounded to 1.0e8.
    # https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Precision_limits_on_integer_values.
    DTYPE_FP32: LARGE_POSITIVE_VALUE
}

LHUC_NAME = "lhuc"
# lhuc application points
LHUC_ENCODER = "encoder"
LHUC_DECODER = "decoder"
LHUC_STATE_INIT = "state_init"
LHUC_ALL = "all"
LHUC_CHOICES = [LHUC_ENCODER, LHUC_DECODER, LHUC_STATE_INIT, LHUC_ALL]

# data sharding
SHARD_NAME = "shard.%05d"
SHARD_SOURCE = SHARD_NAME + ".source"
SHARD_TARGET = SHARD_NAME + ".target"
DATA_INFO = "data.info"
DATA_CONFIG = "data.config"
PREPARED_DATA_VERSION_FILE = "data.version"
PREPARED_DATA_VERSION = 2

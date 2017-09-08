# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


ARG_SEPARATOR = ":"

ENCODER_PREFIX = "encoder_"
EMBEDDING_PREFIX = "embed_"
ATTENTION_PREFIX = "att_"
COVERAGE_PREFIX = "cov_"
BIDIRECTIONALRNN_PREFIX = ENCODER_PREFIX + "birnn_"
STACKEDRNN_PREFIX = ENCODER_PREFIX + "rnn_"
FORWARD_PREFIX = "forward_"
REVERSE_PREFIX = "reverse_"
TRANSFORMER_ENCODER_PREFIX = ENCODER_PREFIX + "transformer_"
CHAR_SEQ_ENCODER_PREFIX = ENCODER_PREFIX + "char_"

# embedding prefixes
SOURCE_EMBEDDING_PREFIX = "source_embed_"
TARGET_EMBEDDING_PREFIX = "target_embed_"
SHARED_EMBEDDING_PREFIX = "source_target_embed_"

# encoder names (arguments)
RNN_NAME = "rnn"
RNN_WITH_CONV_EMBED_NAME = "rnn-with-conv-embed"
TRANSFORMER_TYPE = "transformer"
TRANSFORMER_WITH_CONV_EMBED_TYPE = "transformer-with-conv-embed"

# available encoders
ENCODERS = [RNN_NAME, RNN_WITH_CONV_EMBED_NAME, TRANSFORMER_TYPE, TRANSFORMER_WITH_CONV_EMBED_TYPE]

# available decoder
DECODERS = [RNN_NAME, TRANSFORMER_TYPE]

# rnn types
LSTM_TYPE = 'lstm'
LNLSTM_TYPE = 'lnlstm'
LNGLSTM_TYPE = 'lnglstm'
GRU_TYPE = 'gru'
LNGRU_TYPE = 'lngru'
LNGGRU_TYPE = 'lnggru'
CELL_TYPES = [LSTM_TYPE, LNLSTM_TYPE, LNGLSTM_TYPE, GRU_TYPE, LNGRU_TYPE, LNGGRU_TYPE]

# init types
INIT_XAVIER='xavier'
INIT_UNIFORM='uniform'
INIT_TYPES=[INIT_XAVIER, INIT_UNIFORM]

# RNN init types
RNN_INIT_ORTHOGONAL = 'orthogonal'
RNN_INIT_ORTHOGONAL_STACKED = 'orthogonal_stacked'
# use the default initializer used also for all other weights
RNN_INIT_DEFAULT = 'default'

# attention types
ATT_BILINEAR = 'bilinear'
ATT_DOT = 'dot'
ATT_DOT_SCALED = 'dot_scaled'
ATT_MH_DOT = 'mhdot'
ATT_FIXED = 'fixed'
ATT_LOC = 'location'
ATT_MLP = 'mlp'
ATT_COV = "coverage"
ATT_TYPES = [ATT_BILINEAR, ATT_DOT, ATT_DOT_SCALED, ATT_MH_DOT, ATT_FIXED, ATT_LOC, ATT_MLP, ATT_COV]

# weight tying components
WEIGHT_TYING_SRC='src'
WEIGHT_TYING_TRG='trg'
WEIGHT_TYING_SOFTMAX='softmax'
# weight tying types (combinations of above components):
WEIGHT_TYING_TRG_SOFTMAX='trg_softmax'
WEIGHT_TYING_SRC_TRG='src_trg'
WEIGHT_TYING_SRC_TRG_SOFTMAX='src_trg_softmax'

# default decoder prefixes
DECODER_PREFIX = "decoder_"
TRANSFORMER_DECODER_PREFIX = DECODER_PREFIX + "transformer_"

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

DEFAULT_BEAM_SIZE = 5

VERSION_NAME = "version"
CONFIG_NAME = "config"
LOG_NAME = "log"
JSON_SUFFIX = ".json"
VOCAB_SRC_NAME = "vocab.src"
VOCAB_TRG_NAME = "vocab.trg"
PARAMS_PREFIX = "params."
PARAMS_NAME = PARAMS_PREFIX + "%04d"
PARAMS_BEST_NAME = "params.best"
DECODE_OUT_NAME = "decode.output.%04d"
DECODE_IN_NAME = "decode.source"
DECODE_REF_NAME = "decode.target"
SYMBOL_NAME = "symbol" + JSON_SUFFIX
METRICS_NAME = "metrics"
TENSORBOARD_NAME = "tensorboard"

# training resumption constants
TRAINING_STATE_DIRNAME = "training_state"
TRAINING_STATE_TEMP_DIRNAME = "tmp.training_state"
TRAINING_STATE_TEMP_DELETENAME = "delete.training_state"
MODULE_OPT_STATE_NAME = "mx_optimizer.pkl"
BUCKET_ITER_STATE_NAME = "bucket.pkl"
RNG_STATE_NAME = "rng.pkl"
MONITOR_STATE_NAME = "monitor.pkl"
TRAINING_STATE_NAME = "training.pkl"
SCHEDULER_STATE_NAME = "scheduler.pkl"
TRAINING_STATE_PARAMS_NAME = "params"
ARGS_STATE_NAME = "args.json"

# Arguments that may differ and still resume training
ARGS_MAY_DIFFER = ["overwrite_output", "use-tensorboard", "quiet",
                   "align_plot_prefix", "sure_align_threshold",
                   "keep_last_params"]

# Other argument constants
INFERENCE_ARG_INPUT_LONG = "--input"
INFERENCE_ARG_INPUT_SHORT = "-i"
INFERENCE_ARG_OUTPUT_LONG = "--output"
INFERENCE_ARG_OUTPUT_SHORT = "-o"


# data layout strings
BATCH_MAJOR = "NTC"
TIME_MAJOR = "TNC"

# Training constants
OPTIMIZER_ADAM = "adam"
OPTIMIZER_EVE = "eve"
OPTIMIZER_RMSPROP = "rmsprop"
OPTIMIZER_SGD = "sgd"
OPTIMIZERS = [OPTIMIZER_ADAM, OPTIMIZER_EVE, OPTIMIZER_RMSPROP, OPTIMIZER_SGD]

LR_SCHEDULER_FIXED_RATE_INV_SQRT_T = "fixed-rate-inv-sqrt-t"
LR_SCHEDULER_FIXED_RATE_INV_T = "fixed-rate-inv-t"
LR_SCHEDULER_FIXED_STEP = "fixed-step"
LR_SCHEDULER_PLATEAU_REDUCE = "plateau-reduce"
LR_SCHEDULERS = [LR_SCHEDULER_FIXED_RATE_INV_SQRT_T,
                 LR_SCHEDULER_FIXED_RATE_INV_T,
                 LR_SCHEDULER_FIXED_STEP,
                 LR_SCHEDULER_PLATEAU_REDUCE]

OUTPUT_HANDLER_TRANSLATION = "translation"
OUTPUT_HANDLER_TRANSLATION_WITH_ALIGNMENTS = "translation_with_alignments"
OUTPUT_HANDLER_BENCHMARK = "benchmark"
OUTPUT_HANDLER_ALIGN_PLOT = "align_plot"
OUTPUT_HANDLER_ALIGN_TEXT = "align_text"
OUTPUT_HANDLERS = [OUTPUT_HANDLER_TRANSLATION,
                   OUTPUT_HANDLER_TRANSLATION_WITH_ALIGNMENTS,
                   OUTPUT_HANDLER_BENCHMARK,
                   OUTPUT_HANDLER_ALIGN_PLOT,
                   OUTPUT_HANDLER_ALIGN_TEXT]

# metrics
ACCURACY = 'accuracy'
PERPLEXITY = 'perplexity'
BLEU = 'bleu'
METRICS = [PERPLEXITY, ACCURACY, BLEU]
METRIC_MAXIMIZE = {ACCURACY: True, BLEU: True, PERPLEXITY: False}
METRIC_WORST = {ACCURACY: 0.0, BLEU: 0.0, PERPLEXITY: np.inf}

# loss names
CROSS_ENTROPY = 'cross-entropy'
SMOOTHED_CROSS_ENTROPY = 'smoothed-cross-entropy'

TARGET_MAX_LENGTH_FACTOR = 2
DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH = 2

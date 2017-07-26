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

BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
PAD_ID = 0
TOKEN_SEPARATOR = " "
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]

# default encoder prefixes
ENCODER_PREFIX = "encoder_"
EMBEDDING_PREFIX = "embed_"
ATTENTION_PREFIX = "att_"
COVERAGE_PREFIX = "cov_"
BIDIRECTIONALRNN_PREFIX = ENCODER_PREFIX + "birnn_"
STACKEDRNN_PREFIX = ENCODER_PREFIX + "rnn_"
FORWARD_PREFIX = "forward_"
REVERSE_PREFIX = "reverse_"
CHAR_SEQ_ENCODER_PREFIX = ENCODER_PREFIX + "char_"

# embedding prefixes
SOURCE_EMBEDDING_PREFIX = "source_embed_"
TARGET_EMBEDDING_PREFIX = "target_embed_"

# encoder names (arguments)
RNN_NAME = "rnn"
RNN_WITH_CONV_EMBED_NAME = "rnn-with-conv-embed"
# available encoders
ENCODERS = [RNN_NAME, RNN_WITH_CONV_EMBED_NAME]

# rnn types
LSTM_TYPE = 'lstm'
LNLSTM_TYPE = 'lnlstm'
LNGLSTM_TYPE = 'lnglstm'
GRU_TYPE = 'gru'
LNGRU_TYPE = 'lngru'
LNGGRU_TYPE = 'lnggru'
CELL_TYPES = [LSTM_TYPE, LNLSTM_TYPE, LNGLSTM_TYPE, GRU_TYPE, LNGRU_TYPE, LNGGRU_TYPE]

# init types
RNN_INIT_ORTHOGONAL = 'orthogonal'
RNN_INIT_ORTHOGONAL_STACKED = 'orthogonal_stacked'

# attention types
ATT_BILINEAR = 'bilinear'
ATT_DOT = 'dot'
ATT_DOT_SCALED = 'dot_scaled'
ATT_FIXED = 'fixed'
ATT_LOC = 'location'
ATT_MLP = 'mlp'
ATT_COV = "coverage"
ATT_TYPES = [ATT_BILINEAR, ATT_DOT, ATT_DOT_SCALED, ATT_FIXED, ATT_LOC, ATT_MLP, ATT_COV]

# default decoder prefixes
DECODER_PREFIX = "decoder_"

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

# metric names
ACCURACY = 'accuracy'
PERPLEXITY = 'perplexity'
BLEU = 'bleu'

# loss names
CROSS_ENTROPY = 'cross-entropy'
SMOOTHED_CROSS_ENTROPY = 'smoothed-cross-entropy'

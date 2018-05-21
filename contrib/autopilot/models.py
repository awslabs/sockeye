# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# Names model types
MODEL_NONE = "none"
MODEL_TRANSFORMER = "transformer"

# Named decoding settings
DECODE_STANDARD = "standard"

# Model configurations (architecture, training recipe, etc.)
MODELS = {

    MODEL_NONE: [],

    MODEL_TRANSFORMER: [
        "--encoder=transformer",
        "--decoder=transformer",
        "--num-layers=6:6",
        "--transformer-model-size=512",
        "--transformer-attention-heads=8",
        "--transformer-feed-forward-num-hidden=2048",
        "--transformer-positional-embedding-type=fixed",
        "--transformer-preprocess=n",
        "--transformer-postprocess=dr",
        "--transformer-dropout-attention=0.1",
        "--transformer-dropout-act=0.1",
        "--transformer-dropout-prepost=0.1",
        "--weight-tying",
        "--weight-tying-type=src_trg_softmax",
        "--weight-init=xavier",
        "--weight-init-scale=3.0",
        "--weight-init-xavier-factor-type=avg",
        "--num-embed=512:512",
        "--optimizer=adam",
        "--optimized-metric=perplexity",
        "--label-smoothing=0.1",
        "--gradient-clipping-threshold=-1",
        "--initial-learning-rate=0.0002",
        "--learning-rate-reduce-num-not-improved=8",
        "--learning-rate-reduce-factor=0.9",
        "--learning-rate-scheduler-type=plateau-reduce",
        "--learning-rate-decay-optimizer-states-reset=best",
        "--learning-rate-decay-param-reset",
        "--max-num-checkpoint-not-improved=32",
        "--batch-type=word",
        "--batch-size=4096",
        "--checkpoint-frequency=2000",
        "--decode-and-evaluate=500",
        "--keep-last-params=60",
    ],
}  # type: Dict[str, List[str]]

# Decoding configurations
DECODE_ARGS = {
    DECODE_STANDARD: [
        "--beam-size=5",
        "--batch-size=32",
        "--chunk-size=1000",
        "--length-penalty-alpha=0.1",
        "--length-penalty-beta=0.0",
        "--max-output-length-num-stds=2",
        "--bucket-width=10",
    ]
}

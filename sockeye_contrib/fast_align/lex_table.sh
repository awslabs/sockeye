#!/usr/bin/env bash

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

if [ $(docker images -q fast_align |wc -l) -lt 1 ]; then
    echo "Please install Docker and run build.sh to create the fast_align image." >&2
    exit 2
fi

if [ $# -lt 3 ]; then
    echo "Create lex table with fast_align" >&2
    echo "Usage: ${0} train.src train.trg lex.out" >&2
    exit 2
fi

# Bitext format: source ||| target
# Plus a few tricks for macOS compatibility
TAB=$'\t'
LEX_FILE=$(cd "$(dirname "$3")" && pwd -P)/$(basename "$3")
paste <(zcat -f <"$1") <(zcat -f <"$2") |sed -e "s/${TAB}/ ||| /g" > "$LEX_FILE".tmp

# Run fast_align with recommended settings, write lex table but not alignments
touch "$LEX_FILE"
docker run --rm -i -v "$LEX_FILE".tmp:/input -v "$LEX_FILE":/lexicon fast_align fast_align -i /input -v -d -o -p /lexicon -t -1000000 >/dev/null

# Cleanup
rm "$LEX_FILE".tmp

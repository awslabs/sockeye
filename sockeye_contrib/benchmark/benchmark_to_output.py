#!/usr/bin/env python

# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sys
from typing import Dict, Iterator


def read_benchmark_handler_output(stream: str) -> Iterator[Dict[str, str]]:
    for line in stream:
        fields = line.strip().split('\t')
        entry = dict(field.split('=', 1) for field in fields)
        yield entry


def get_output_from_benchmark_output(input_stream) -> Iterator[str]:
    for entry in read_benchmark_handler_output(input_stream):
        yield entry['output']


def main():
    for output in get_output_from_benchmark_output(sys.stdin):
        print(output)


if __name__ == '__main__':
    main()

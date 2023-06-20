
# Copyright 2017--2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import subprocess as sp

for line in open('embedding_files.txt'):
    f = line.strip()
    sp.check_call(f'wget {f}', shell=True)
    f2 = f.split('/')[-1]
    sp.check_call(f'gunzip {f2}', shell=True)



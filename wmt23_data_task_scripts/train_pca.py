
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


#pip3 install faiss-cpu==1.7.4

import numpy as np
import faiss
from glob import glob

arrays = []
N = 1_000_000
for filex in ('sentences-et-exclude-wrong_lang.tsv.gz.part0',
              'sentences-et-include-correct_lang.tsv.gz.part0',
              'sentences-lt-exclude-wrong_lang.tsv.gz.part0',
              'sentences-lt-include-correct_lang.tsv.gz.part0'):
    arrays.append(np.fromfile(filex, dtype=np.float32).reshape(-1, 1024)[:N, :])

sample = np.concatenate(arrays)

pca_ = faiss.PCAMatrix(1024, 128)
pca_.train(sample)

faiss.write_VectorTransform(pca_, "PCA.pca")

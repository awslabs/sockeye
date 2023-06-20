
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



import pathlib
pathlib.Path("output").mkdir(parents=True, exist_ok=True)

import numpy as np
import faiss
from glob import glob

pca = faiss.read_VectorTransform("PCA.pca")

for filex in glob('*tsv.gz.part*'):
    x = np.fromfile(filex, dtype=np.float32).reshape(-1, 1024)
    x2 = pca.apply(x).astype(np.float16)
    print(x2.shape, x2.dtype)
    np.save('output/'+filex+'.pca128_fp16.npz', x2, allow_pickle=False, fix_imports=True)

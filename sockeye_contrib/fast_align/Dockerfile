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

FROM amazonlinux:2017.09.0.20170930

RUN yum install -y \
    cmake \
    git \
    gcc \
    gcc-c++ \
    gperftools-devel \
    && yum clean all

ENV FAST_ALIGN_COMMIT 7c2bbca3d5d61ba4b0f634f098c4fcf63c1373e1
RUN cd /opt \
    && git clone https://github.com/clab/fast_align.git \
    && cd fast_align \
    && git reset --hard ${FAST_ALIGN_COMMIT} \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make
ENV PATH ${PATH}:/opt/fast_align/build

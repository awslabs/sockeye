FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV CUDNN_VERSION=7.6.0.64-1+cuda10.0
ENV NCCL_VERSION=2.4.7-1+cuda10.0

ENV PYTHON_VERSION=3.6

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install MXNet
ENV MXNET_VERSION=1.5.0b20190710
#RUN pip install mxnet-cu100mkl==${MXNET_VERSION}
RUN pip install mxnet-mkl==${MXNET_VERSION}

# Install Horovod, temporarily using CUDA stubs
#    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_MXNET=1 \
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_WITH_MXNET=1 \
         pip install --no-cache-dir horovod && \
    ldconfig

# Install Sockeye dependencies
RUN pip install \
    pyyaml>=5.1 \
    numpy>=1.14 \
    typing \
    portalocker \
    sacrebleu==1.3.5 \
    gluonnlp

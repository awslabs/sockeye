# Sockeye 3 Benchmark Scripts

## Host Setup

We recommend starting with a clean Python environment by installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python 3.8 Linux 64-bit).

GPU benchmarks require [CUDA 11](https://developer.nvidia.com/cuda-toolkit).

Creating lexical shortlists for Sockeye requires [Docker](https://docs.docker.com/engine/install).

## General Dependencies

### PyTorch

```bash
pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout ed94d0bb6fda5b57d7e7637bc372a47af1ec3e07
sed -ie 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/#check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/g' setup.py
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
```

### Subword-NMT

```bash
pip3 install subword-nmt
```

## NMT Toolkits

### Sockeye

<https://github.com/awslabs/sockeye>

```bash
git clone https://github.com/awslabs/sockeye.git
cd sockeye
git checkout 6905c787325e087537ce523c9d977ee4de188606
pip3 install --editable ./
cd ..
```

Optional step (faster inference): Build fast_align Docker image to train lexical shortlists:

```bash
cd sockeye/sockeye_contrib/fast_align
bash build.sh
export PATH=$PWD:/$PATH
cd ..
```

### Fairseq

<https://github.com/pytorch/fairseq>

```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout a54021305d6b3c4c5959ac9395135f63202db8f1
pip3 install --editable ./
pip3 install tensorboardX
cd ..
```

### OpenNMT

<https://github.com/OpenNMT/OpenNMT-py>

```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
git checkout 4815f07fcd482af9a1fe1d3b620d144197178bc5
pip3 install --editable ./
cd ..
```

## Running Benchmarks

### Data

WMT17 English-German data set:

```bash
bash scripts/data/download_prepare_wmt17_en_de.sh
```

WMT17 Russian-English data set:

```bash
bash scripts/data/download_prepare_wmt17_en_de.sh
```

### Sockeye

Sockeye supports the standard big transformer model and the big transformer+SSRU model.
The training script prompts for a model type and maximum number of updates (see help message).
Training runs on 8 local GPUs.

```bash
bash scripts/sockeye/prepare_data.sh
bash scripts/sockeye/train.sh
```

Optional step (higher quality): Average the 8 best checkpoints.

```bash
mv sockeye.big/params.best sockeye.big/params.best.single
sockeye-average -n 8 --output sockeye.big/params.best sockeye.big
```

Optional step (faster inference): Create a lexical shortlist.

```bash
lex_table.sh train.src.bpe train.trg.bpe lex.out
bash scripts/sockeye/shortlist.sh sockeye.big lex.out
```

Translate test sets with various settings on local GPU and CPUs.

```bash
bash scripts/sockeye/translate.sh sockeye.big gpu 64 sockeye.big.gpu.64 sockeye.shortlist
bash scripts/sockeye/translate.sh sockeye.big gpu 1 sockeye.big.gpu.64 sockeye.shortlist
bash scripts/sockeye/translate.sh sockeye.big cpu 1 sockeye.big.cpu.1 sockeye.shortlist
```

### Fairseq

Fairseq supports the standard big transformer model.
The training script prompts for a maximum number of updates (see help message).

```bash
bash scripts/fairseq/prepare_data.sh
bash scripts/fairseq/train.sh
```

Optional step (higher quality): Average the 8 best checkpoints.

```bash
mv fairseq.big/checkpoint_best.pt fairseq.big/checkpoint_best_single.pt
python3 fairseq/scripts/average_checkpoints.py --inputs $(python3 scripts/fairseq/find_best_checkpoints.py 8 <fairseq.big.log) --output fairseq.big/checkpoint_best.pt
```

Translate test sets with various settings on local GPU and CPUs.

```bash
bash scripts/fairseq/translate.sh fairseq.big gpu 64 fairseq.big.gpu.64
bash scripts/fairseq/translate.sh fairseq.big cpu 1 fairseq.big.cpu.1
```

### OpenNMT

OpenNMT supports the standard big transformer model.
The training script prompts for a maximum number of updates (see help message).

```bash
bash scripts/onmt/train.sh
```

Required step: Average the 8 best checkpoints.

```bash
onmt_average_models -models $(python3 scripts/onmt/find_best_checkpoints.py 8 <onmt/train.log) -output onmt/big_average.pt
```

Translate test sets with various settings on local GPU and CPUs.

```bash
bash scripts/onmt/translate.sh onmt gpu 64 onmt.big.gpu.64
bash scripts/onmt/translate.sh onmt cpu 1 onmt.big.cpu.1
```

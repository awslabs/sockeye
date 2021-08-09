# Sockeye Docker Images

Run the build script to produce optimized Sockeye Docker images for GPUs or CPUs:

```bash
python3 sockeye_contrib/docker/build.py gpu
python3 sockeye_contrib/docker/build.py cpu
```

## Example: Distributed Training with Horovod and Automatic Mixed Precision

Prepare the training data:

```bash
nvidia-docker run --rm -i -v $PWD:/work -w /work sockeye-gpu \
    python3 -m sockeye.prepare_data \
        --source train.src \
        --target train.src \
        --output prepared_train
```

Start multi-process mixed-precision training with `horovodrun`:

```bash
nvidia-docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm -i -v $PWD:/work -w /work sockeye-gpu \
        horovodrun -np 8 python3 -m sockeye.train \
            --prepared-data prepared_train \
            --validation-source dev.src \
            --validation-target dev.trg \
            --output model \
            --amp \
            --horovod
```

Add options to `sockeye.prepare_data` and `sockeye.train` to specify a model and training recipe.
See Domhan et al. ([2020](https://www.aclweb.org/anthology/2020.amta-research.10/)) for recommended settings.

## Example: GPU Inference

Sockeye models can be quantized to FP16 at runtime for faster GPU inference.

```bash
nvidia-docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm -i -v $PWD:/work -w /work -e MXNET_SAFE_ACCUMULATION=1 sockeye-gpu \
        python3 -m sockeye.translate \
            --dtype float16 \
            --batch-size 64 \
            --chunk-size 1000 \
            --models model \
            --input test.src \
            --output test.out
```

## Example: CPU Inference

Sockeye models can be quantized to INT8 at runtime for faster CPU inference.
A top-K lexicon can be used for vocabulary restriction/shortlisting to further improve speed.
See the [fast_align documentation](sockeye_contrib/fast_align) and `sockeye.lexicon create`; k=200 works well in practice.
Setting the number of OMP threads to the number of physical CPU cores (`NCPUS` in the following) can also improve speed.

```bash
docker run --rm -i -v $PWD:/work -w /work sockeye-cpu \
    python3 -m sockeye.translate \
        --use-cpu \
        --omp-num-threads NCPUS \
        --dtype int8 \
        --batch-size 1 \
        --models model \
        --restrict-lexicon LEXICON \
        --input test.src \
        --output test.out
```

# Sockeye Docker Image

Run the build script to produce a nvidia-docker compatible image with the current revision of Sockeye, including full CPU/GPU support and Horovod/OpenMPI.

```bash
python3 sockeye_contrib/docker/build.py
```

To update the image, run `git pull` and/or make your own code changes, then rerun the build script.

## Example: Distributed Training with Horovod

Using the Docker image greatly simplifies distributed training.

### Host Setup

See the Horovod instructions for setting up hosts:

- [Performance improvements for GPU hosts](https://github.com/horovod/horovod/blob/master/docs/gpus.rst)
- [Passwordless SSH for running on multiple hosts](https://github.com/horovod/horovod/blob/master/docs/docker.rst#running-on-multiple-machines)

### Running

This is an example running on CPUs across 2 hosts.

- `COMMIT` is the Sockeye commit
- `HOST2` is the address of the secondary host
- `/mnt/share/ssh` is a SSH directory set up following the Horovod instructions above.
- `/mnt/share` is a general shared directory that all workers will access to read training data and write model files.

#### Secondary Host(s)

On each secondary host, start a Docker container running sshd.
Horovod/OpenMPI will connect to these hosts to launch workers.

```bash
docker run --rm -i --network=host -v /mnt/share/ssh:/home/ec2-user/.ssh -v /mnt/share:/mnt/share sockeye:COMMIT \
    bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```

#### Primary Host

On the primary host, prepare the training data.

```bash
docker run --rm -i -v /mnt/share:/mnt/share --user ec2-user:ec2-user sockeye:COMMIT \
    python3 -m sockeye.prepare_data \
        --source /mnt/share/data/train.src \
        --target /mnt/share/data/train.src \
        --output /mnt/share/data/prepared_train
```

Start Sockeye training with `horovodrun`.

```bash
docker run --rm -i --network=host -v /mnt/share/ssh:/home/ec2-user/.ssh -v /mnt/share:/mnt/share --user ec2-user:ec2-user sockeye:COMMIT \
    horovodrun -np 2 -H localhost:1,HOST2:1 -p 12345 python3 -m sockeye.train \
        --prepared-data /mnt/share/data/prepared_train \
        --validation-source /mnt/share/data/dev.src \
        --validation-target /mnt/share/data/dev.trg \
        --output /mnt/share/data/model \
        --lock-dir /mnt/share/lock \
        --use-cpu \
        --horovod
```

## Experimental CPU-Optimized Image

To build a Docker image with the latest CPU-optimized version of Sockeye, run the following script:

```bash
python3 sockeye_contrib/docker/build_cpu_optimized.py
```

This produces an image called `sockeye-cpu:latest` that uses the latest versions of the following:

- [kpuatamazon/incubator-mxnet](https://github.com/kpuatamazon/incubator-mxnet): The MXNet fork that supports [intgemm](https://github.com/kpu/intgemm) and makes full use of Intel MKL (versus just DNNL).
- [kpuatamazon/sockeye](https://github.com/kpuatamazon/sockeye): The Sockeye fork that supports int8 quantization for inference.

This image can then be used with existing Sockeye models, which can be quantized to int8 at load time.
In the following example, `LEXICON` is a top-k lexicon (see the [fast_align documentation](sockeye_contrib/fast_align) and `sockeye.lexicon create`; k=200 works well in practice) and `NCPUS` is the number of physical CPU cores on the host running Sockeye.

```bash
docker run --rm -i -v $PWD:/work -w /work sockeye-cpu:latest python3 -m sockeye.translate --use-cpu --omp-num-threads NCPUS --dtype int8 --input test.src --restrict-lexicon LEXICON --models model --output test.out
```

# Sockeye Docker Image

Run the Docker build script from the Sockeye root directory:

```bash
bash sockeye_contrib/docker/build.sh
```

The script produces a nvidia-docker compatible image with this version Sockeye, including full CPU/GPU support and Horovod/OpenMPI.

## Example: Distributed Training with Horovod

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
    python -m sockeye.prepare_data \
        --source /mnt/share/data/train.src \
        --target /mnt/share/data/train.src \
        --output /mnt/share/data/prepared_train
```

Start Sockeye training with `horovodrun`.

```bash
docker run --rm -i --network=host -v /mnt/share/ssh:/home/ec2-user/.ssh -v /mnt/share:/mnt/share --user ec2-user:ec2-user sockeye:COMMIT \
    horovodrun -np 2 -H localhost:1,HOST2:1 -p 12345 python -m sockeye.train \
        --prepared-data /mnt/share/data/prepared_train \
        --validation-source /mnt/share/data/dev.src \
        --validation-target /mnt/share/data/dev.trg \
        --output /mnt/share/data/model \
        --lock-dir /mnt/share/lock \
        --use-cpu \
        --horovod
```

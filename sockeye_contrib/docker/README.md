# Docker Image

Run the Docker build script from the Sockeye root directory:

```bash
bash sockeye_contrib/docker/build.sh
```

The script produces a nvidia-docker compatible image with this version Sockeye, including full CPU/GPU support and Horovod/OpenMPI.  The Dockerfile uses logic from the [official Horovod Dockerfile](https://github.com/horovod/horovod/blob/958695e7343ce470ad3b0d9df1967b5af3bd6ec3/Dockerfile), also released under the Apache 2.0 license.

## Example: Distributed Training with Horovod

### Host Setup

See the Horovod instructions for setting up hosts:

- [Performance improvements for GPU hosts](https://github.com/horovod/horovod/blob/master/docs/gpus.rst)
- [Passwordless SSH for running on multiple hosts](https://github.com/horovod/horovod/blob/master/docs/docker.rst#running-on-multiple-machines)

### Running

This is a test system running on CPUs across 2 hosts.

#### Secondary Host(s)

On each secondary host, start a Docker container running sshd.  Horovod will connect to these hosts to launch workers.

- `/mnt/share/ssh` is a SSH directory set up following the Horovod instructions above.
- `/mnt/share` is a general shared directory that all workers will access to read training data and write model files.

```bash
docker run --rm -i --network=host -v /mnt/share/ssh:/home/ec2-user/.ssh -v /mnt/share:/mnt/share sockeye:2-horovod \
    bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```

#### Primary Host

On the primary host, start Sockeye/Horovod training in a Docker container.

- "HOST2" is the address of the secondary host

```bash
docker run --rm -i --network=host -v /mnt/share/ssh:/home/ec2-user/.ssh -v /mnt/share:/mnt/share --user ec2-user:ec2-user sockeye:2-horovod \
    horovodrun -np 2 -H localhost:1,HOST2:1 -p 12345 python -m sockeye.train \
        -s /mnt/share/data/train.src \
        -t /mnt/share/data/train.trg \
        -vs /mnt/share/data/dev.src \
        -vt /mnt/share/data/dev.trg \
        -o /mnt/share/data/model \
        --lock-dir /mnt/share/lock \
        --use-cpu \
        --horovod
```

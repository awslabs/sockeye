# Setup & Installation

## Dependencies

Sockeye requires:
- **Python3**
- [MXNet 1.4.1](https://github.com/apache/incubator-mxnet/tree/1.4.1)
- numpy

## Installation

There are several options for installing Sockeye and its dependencies.
Below we list several alternatives and the corresponding instructions.

### → via pip...

The easiest way to install is with [pip](https://pypi.org):

```bash
> pip install sockeye
```

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet Incubating contains the GPU bindings.
Depending on your version of CUDA, you can do this by running the following:

```bash
wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements/requirements.gpu-cu${CUDA_VERSION}.txt
pip install sockeye --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
rm requirements.gpu-cu${CUDA_VERSION}.txt
```
where `${CUDA_VERSION}` can be `80` (8.0), `90` (9.0), `92` (9.2), or `100` (10.0).

### → via source...

If you want to just use sockeye without extending it, simply install it via
```bash
> pip install -r requirements/requirements.txt
> pip install .
```
after cloning the repository from git.

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by
running the following:

```bash
> pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```
where `${CUDA_VERSION}` can be `80` (8.0), `90` (9.0), `92` (9.2), or `100` (10.0).

Developers will be better served by pointing `$PYTHONPATH` to the root of the git-cloned source.

### → on AWS...

[AWS DeepLearning AMI](https://aws.amazon.com/amazon-ai/amis/) users only need to run the following line to install sockeye:

```bash
> sudo pip3 install sockeye --no-deps
```

For other environments, you can choose between installing via pip or directly from source. Note that for the
remaining instructions to work you will need to use `python3` instead of `python` and `pip3` instead of `pip`.

### Optional dependencies
In order to write training statistics to a Tensorboard event file for visualization, you can optionally install mxboard
 (````pip install mxboard````). To visualize these, run the Tensorboard tool (`pip install tensorboard tensorflow`) with
 the logging directory pointed to the training output folder: `tensorboard --logdir <model>`

If you want to create alignment plots you will need to install matplotlib (````pip install matplotlib````).

In general you can install all optional dependencies from the Sockeye source folder using:
```bash
> pip install '.[optional]'
```

### Running sockeye

After installation, command line tools such as *sockeye-train, sockeye-translate, sockeye-average* and *sockeye-embeddings* are available.
For example:

```bash
> sockeye-train <args>
```

Equivalently, if the sockeye directory is on your `$PYTHONPATH`, you can run the modules directly:

```bash
> python -m sockeye.train <args>
```

# Setup & Installation

## Dependencies

Sockeye requires:
- **Python3.7** or above
- [MXNet 1.8.0](https://github.com/apache/incubator-mxnet/tree/1.8.0)
- Numpy

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
> wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install sockeye --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
> rm requirements.gpu-cu${CUDA_VERSION}.txt
```
where `${CUDA_VERSION}` can be `100` (10.0), `101` (10.1), `102` (10.2), '110' (11.0), or '112' (11.2).

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
where `${CUDA_VERSION}` can be `100` (10.0), `101` (10.1), `102` (10.2), '110' (11.0), or '112' (11.2).

Developers will be better served by pointing `$PYTHONPATH` to the root of the git-cloned source.

### → in an Anaconda environment ...

In an Anaconda environment such as the one provided by the [AWS DeepLearning AMI](https://aws.amazon.com/amazon-ai/amis/) or Azure when using the [Data Science Virtual Machine](http://aka.ms/dsvm/discover) image, users only need to run the following line to install sockeye (on an instance without a GPU):

```bash
> conda create -n sockeye python=3.6
> source activate sockeye
> pip install sockeye --no-deps
```

On an instance with a GPU, the following commands will work

```bash
> conda create -n sockeye python=3.6
> source activate sockeye
> wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install sockeye --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
rm requirements.gpu-cu${CUDA_VERSION}.txt
```
where `${CUDA_VERSION}` can be `100` (10.0), `101` (10.1), `102` (10.2), '110' (11.0), or '112' (11.2).

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

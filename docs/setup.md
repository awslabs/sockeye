# Setup & Installation

## Dependencies

Sockeye requires:
- **Python 3.7** or above
- [PyTorch 1.10](https://github.com/pytorch/pytorch/releases/tag/v1.10.0)
- Numpy

## Installation

There are several options for installing Sockeye and its dependencies.
Below we list several alternatives and the corresponding instructions.

### → via pip...

The easiest way to install is with [pip](https://pypi.org):

```bash
> pip install sockeye
```

### → via source...

If you want to just use Sockeye without extending it, simply install it via
```bash
> pip install -r requirements/requirements.txt
> pip install .
```
after cloning the repository from git.

Developers will be better served by pointing `$PYTHONPATH` to the root of the git-cloned source.

### → in an Anaconda environment ...

In an Anaconda environment such as the one provided by the [AWS DeepLearning AMI](https://aws.amazon.com/amazon-ai/amis/) or Azure when using the [Data Science Virtual Machine](http://aka.ms/dsvm/discover) image, users only need to run the following line to install sockeye (on an instance without a GPU):

```bash
> conda create -n sockeye python=3.8
> source activate sockeye
> pip install sockeye --no-deps
```

### Optional dependencies
In order to write training statistics to a Tensorboard event file for visualization, you can optionally install tensorboard
 (````pip install tensorboard````). To visualize these, run the Tensorboard tool with
 the logging directory pointed to the training output folder: `tensorboard --logdir <model>`

In general you can install all optional dependencies from the Sockeye source folder using:
```bash
> pip install '.[optional]'
```

### Running Sockeye

After installation, command line tools such as *sockeye-train, sockeye-prepare-data, sockeye-translate, sockeye-average* and *sockeye-embeddings* are available.
For example:

```bash
> sockeye-train <args>
```

Equivalently, if the sockeye directory is on your `$PYTHONPATH`, you can run the modules directly:

```bash
> python -m sockeye.train <args>
```

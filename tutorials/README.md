# MT Marathon 2017 Tutorial

## Setup

For running the examples in the tutorials you can either manually install Sockeye on your laptop (/any other machine)
or use the EC2 instance we already set up.

### Either: Provided AWS EC2 instance

Everything is already set up for you. You can simply follow the tutorials below without any setup.

### Or: Your laptop/machine
The tutorial below has been adopted to the provided EC2 machine.
The exact same tutorial with additional instructions on how to install Sockeye and preprocess data
can be found [here](https://github.com/awslabs/sockeye/tree/master/tutorials).
So in case you want to use your own computer please proceed to [that tutorial](https://github.com/awslabs/sockeye/tree/master/tutorials).

## Setup
Alright, so assuming you're running on the provided EC2 instance we can get started by creating a folder
to store you models:

```bash
ssh ubuntu@$IP_ADDRESS
mkdir /home/ubuntu/efs/working_dir/YOUR_USERNAME
cd /home/ubuntu/efs/working_dir/YOUR_USERNAME
```

All (preprocessed) data that will be used can be found in `/home/ubuntu/efs/tutorials`.


## Tutorials

The tutorial is split up into two distinct tasks that each show different features of Sockeye.
It therefore makes sense to first look at the sequence copy task and then train the WMT model.

1. [Sequence copy task](seqcopy)
1. [WMT German to English news translation](wmt)

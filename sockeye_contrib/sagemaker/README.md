# SageMaker Training

This directory contains scripts for running `sockeye-train` on [Amazon SageMaker](https://aws.amazon.com/sagemaker/).
SageMaker provides a managed environment for training models on different hardware platforms without the need to manually manage EC2 instances.

NOTE 1: SageMaker is an [Amazon web service](https://aws.amazon.com/) that requires an [AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/).
The [pricing page](https://aws.amazon.com/sagemaker/pricing/) lists hourly costs for training models on different types of instances.

NOTE 2: The Sockeye SageMaker scripts launch and monitor remote training jobs.
**Stopping the local script (Control+C) does not stop the remote SageMaker job.**
**To stop a SageMaker training job, use the [AWS Management Console](https://aws.amazon.com/console/) or [AWS CLI](https://aws.amazon.com/cli/):**

```
aws sagemaker stop-training-job --training-job-name JOB_NAME
```

## Setup

To get started, install the required packages:

```
pip install -r requirements.txt
```

Installing the [AWS CLI](https://aws.amazon.com/cli/) is also recommended.

## Model and Data Configuration

To configure a new training run, create a training arguments file (`args.yaml`) or copy and modify one from an existing model directory.
This is the YAML version of `sockeye-train`'s CLI arguments.
To get more information about these arguments, you can run `sockeye-train -h`.

Inputs (prepared data, validation source and target, etc.) can be local files or S3 paths.
Local files will be automatically uploaded to S3 before launching training.
Uploading all input files to S3 ahead of time is recommended because it enables reusing them across training runs without uploading a new copy for each run:

```
aws s3 cp prepared_data s3://path/to/prepared_data
```

Alternatively, you can launch the first training run with local files, copy the S3 paths where they are uploaded, and use those paths for future training runs.

Instead of using the arguments file's `output` location, the training job will upload the output directory to S3:

```
[INFO:__main__] Trained model will be uploaded to: s3://path/to/output
```

## Launching Training

Once you have created an arguments file, you can use the launch script to start a SageMaker training job.
Specify the arguments file, an [instance type](https://aws.amazon.com/sagemaker/pricing/), and a job prefix.
On single-GPU and CPU instances, the training script will run a single `sockeye-train` process.
On multi-GPU instances, the script will run distributed training with one `sockeye-train` process per GPU.

### Example 1

Training a minimal test model on CPUs (digits sorting or similar):

```
python launch.py -c args.yaml -i ml.c5.2xlarge -j sockeye-train-${USER}-test
```

### Example 2

Training a WMT-scale model on 8 GPUs:

```
python launch.py -c args.yaml -i ml.p3.16xlarge -j sockeye-train-${USER}-wmt
```

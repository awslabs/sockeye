# Image Captioning

This module extends Sockeye to perform image captioning. It follows the same logic of sequence-to-sequence frameworks, which consist of encoder-decoder models.
In this case the encoder takes an image instead of a sentence and encodes it in a feature representation.
This is decoded with attention (optionally) using exactly the same models of Sockeye (RNNs, transformers, or CNNs).


## Installation

Follow the instructions to install Sockeye, and install further dependencies:

```bash
> sudo pip3 install Pillow
```

Optionally you can also install matplotlib for visualization:
```bash
> sudo pip3 install matplotlib
```


## First Steps

### Train

In order to train your first image captioning model you will need two sets of parallel files: one for training
and one for validation. The latter will be used for computing various metrics during training.
Each set should consist of two files: one with source images and one with target sentences (captions).
Both files should have the same number of lines, each line containing the relative path of the image and a single
sentence, respectively. Each sentence should be a whitespace delimited list of tokens.

First, you need to obtain the mxnet image models from the model gallery: https://github.com/dmlc/mxnet-model-gallery

Then, we can extract features from them:
```bash
> python -m sockeye.image_captioning.extract_features \
        --image-root /path/to/image/dataset/folder/ \
        --input training_set.images \
        --output-root /path/to/feature/cache/folder/ \
        --output training_set.features \
        --device-id 0 \
        --batch-size 128 \
        --source-image-size 3 224 224 \
        --image-encoder-model-path /path/to/mxnet/model/filename_prefix \
        --image-encoder-layer stage4_unit3_conv3

> python -m sockeye.image_captioning.extract_features \
        --image-root /path/to/image/dataset/folder/ \
        --input validation_set.images \
        --output-root /path/to/feature/cache/folder/ \
        --output validation_set.features \
        --device-id 0 \
        --batch-size 128 \
        --source-image-size 3 224 224 \
        --image-encoder-model-path /path/to/mxnet/model/filename_prefix \
        --image-encoder-layer stage4_unit3_conv3
```
In the option `--image-encoder-model-path`, `filename_prefix` should be the prefix of the MXNet model without `-symbol.json` or `-0000.params`.

The script above will generate the features stored in `/path/to/feature/cache/` and a file `training_set.features` which contains the path to the features relative to `/path/to/feature/cache/`.
Note that finetuning of the image model is not supported yet.


Now we can train an one-layer LSTM with attention for image captioning model as follows:
```bash
> python -m sockeye.image_captioning.train \
        --source-root /path/to/feature/cache/folder/ \
        --source training_set.features \
        --target training_set.captions \
        --validation-source-root /path/to/feature/cache/folder/ \
        --validation-source validation_set.features \
        --validation-target validation_set.captions \
        --batch-size 64 \
        --initial-learning-rate 0.0003 \
        --gradient-clipping-threshold 1.0 \
        --bucket-width 5 \
        --max-seq-len 1:60 \
        --fill-up replicate \
        --output models/ \
        --encoder image-pretrain-cnn \
        --rnn-num-hidden 512 \
        --rnn-decoder-state-init zero \
        --checkpoint-frequency 200 \
        --weight-normalization
```
Use the option `--load-all-features-to-memory` to load all the features to memory. This is possible depending on the size of the dataset/features and amount of available CPU memory.
There is an initial overhead to load the feature (training does not start immediately), but with the big advantage that training is 15X-20X faster.

You can add the options `--decode-and-evaluate 200 --max-output-length 60` to perform captioning of the part of the validation set (200 samples in this case) during training.

### Image to Text

Assuming that features were pre-extracted, you can do image captioning as follows:

```bash
> python -m sockeye.image_captioning.captioner \
        --models models/ \
        --input validation_set.features \
        --source-root /path/to/feature/cache/folder/ \
        --max-output-length 60 \
        --batch-size 1024 \
        --chunk-size 2048 \
        --beam-size 3 > validation_set.predictions
```

This will take the best set of parameters found during training and then load the image provided in the STDIN and
write the caption to STDOUT, which is redirected using `>` to the file `validation_set.predictions` overwriting its content if it exists already.

You can also caption directly from image with the option `--extract-image-features` as follows:

```bash
> python -m sockeye.image_captioning.captioner \
        --extract-image-features \
        --source-image-size 3 224 224 \
        --image-encoder-model-path /path/to/mxnet/model/filename_prefix \
        --models models/ \
        --input validation_set.images \
        --source-root  /path/to/image/dataset/folder/ \
        --max-output-length 60 \
        --batch-size 512 \
        --chunk-size 1024 \
        --beam-size 3 > validation_set.predictions
```

### Visualization

You can now visualize the results in a nice format as follows:

```bash
> python -m sockeye.image_captioning.visualize \
        --image-root /path/to/image/dataset/folder/ \
        --source validation_set.images \
        --prediction validation_set.predictions \
        --ground-truth validation_set.captions \
        --save-to-folder validation_set/
````
This will save to disk plots containing images, predicted captions (white background) and optionally (mutiple) ground-truth captions (green background).
It is possible to remove `--save-to-folder` and the plots will be visualized on screen.
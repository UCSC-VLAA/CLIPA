# An Inverse Scaling Law for CLIP Training
This is a PyTorch/PyTorch-XLA implementation of the paper  [An Inverse Scaling Law for CLIP Training]().
It can run on both GPU or TPU devices (see [TRAIN](https://github.com/UCSC-VLAA/RobustCNN/blob/main/TRAIN.md)/[TEST](https://github.com/UCSC-VLAA/RobustCNN/blob/main/TEST.md) instructions).
Our implementation is heavily based on the [OpenCLIP](https://github.com/mlfoundations/open_clip).




## Installation
A simple 
```
pip install -r requirements.txt
```
is enough.

Note that this repo is compatible with both GPU and TPU. If you want to run the code on Google Cloud TPU, here are some documents you may find helpful:
[Google Cloud User's Guide](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm) and [TIMM bits README](https://github.com/rwightman/pytorch-image-models/blob/bits_and_tpu/timm/bits/README.md)


## Data preparation

### LAION400M
You can download the LAION400M dataset using the handy [img2dataset](https://github.com/rom1504/img2dataset) tool. 
It supports both `webdataset` and `tfrecord` format. 

To run this code on Google Cloud, we strongly recommend `tfrecord` over `webdataset`.
Since LAION400M is not a TFDS officially supported dataset, we provide some custom scripts for post-processing the downloaded tfrecord files [here]().

### ImageNet-1K
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the [standard layout](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) for the torchvision, and the training and validation data is expected to be in the `train` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

We also support reading ImageNet-1K in the `tfrecord` format. 
Check this [official doc](https://www.tensorflow.org/datasets/cli) for how to prepare tfds dataset.


## Acknowledgment

This repo is mainly built on [OpenCLIP](https://github.com/mlfoundations/open_clip). 
And this work is supported by a gift from Open Philanthropy, TPU Research Cloud (TRC) program, and Google Cloud Research Credits program.
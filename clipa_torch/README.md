# An Inverse Scaling Law for CLIP Training
This is a PyTorch/PyTorch-XLA implementation of the paper  [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017).
This repo is heavily based on [OpenCLIP](https://github.com/mlfoundations/open_clip), 
with a few changes including pytorch-xla compatible implementation, uint8 data transfer, and some minor modifications.

<p align="center">
  <img src="figs/inverse_scaling_law.png" width="1080">
Overview of the Inverse Scaling Law: larger image/text encoders
enable training with fewer image/text tokens while maintaining competitive performance
</p>

## Installation
A simple 
```
pip install -r requirements.txt
```
is enough.

Note that this repo is compatible with both GPU and TPU. If you want to run the code on Google Cloud TPU, here are some documents you may find helpful:
[Google Cloud User's Guide](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm) and [TIMM bits README](https://github.com/rwightman/pytorch-image-models/blob/bits_and_tpu/timm/bits/README.md)


## Data preparation
### LAION-400M
You can download the LAION-400M dataset using the handy [img2dataset](https://github.com/rom1504/img2dataset) tool. 
It supports both `webdataset` and `tfrecord` format. 

To run this code on Google Cloud, we strongly recommend `tfrecord` over `webdataset`.
Since LAION-400M is not a TFDS officially supported dataset, for your convenience, we provide some self-implemented scripts (sorry, we know they are crudely written) for post-processing the downloaded tfrecord files [here](../data/laion400m/README.md)..

### ImageNet-1K
Download and extract ImageNet data from http://image-net.org/.
The directory structure is the [standard layout](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and note that we only need the validation set:

```
/path/to/imagenet/
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

We also support reading ImageNet-1K in the `tfrecord` format. 
Check the [official doc](https://www.tensorflow.org/datasets/cli) for how to prepare the tfds dataset.

## Usage

### Training Instructions
We provide example scripts to reproduce our CLIPA results on an A100 eight-GPU machine.

For instance, to reproduce the CLIPA-L16(I37,T8) results, first run the pre-training script
```
bash scripts/exp/gpu/vit_l16/i37_t8_pretrain.sh
```
and fine-tune the pre-trained checkpoint with
```
bash scripts/exp/gpu/vit_l16/i37_t8_finetune.sh
```
- Remember to change the path to dataset and checkpoint to your own path. You have to use the imagenet validation path `/path/to/imagenet/val` if you are reading from disk! 
- The training time is ~2 days for pre-training and ~1 day for fine-tuning on an A100 eight-GPU machine.
- Note that to ensure proper shuffling diversity, each worker maintains a TFDS shuffling buffer and prefetch buffer. 
This significantly increase cpu memory burden. If you observe cpu out-of-memory issue, try tune down the values of `TFDS_PREFETCH_SIZE` and `TFDS_SHUFFLE_SIZE`.

### Testing Instructions
This repo was only used for evaluating zero-shot Top-1 accuracy on ImageNet-1k. 
For evaluation on more datasets, check the amazing [clip_benchmark](https://github.com/LAION-AI/CLIP_benchmark).
The checkpoints from this repo should be readily applicable.

### Model Weights
Here are CLIPA trained weights on LAION-400M with academic resources. 
All models are pre-trained for 6 epochs with reduced input token lengths and subsequently fine-tuned for 0.36 epoch with full input token lengths.

|                     |                                          Pretrained Model                                           | ImageNet |
|---------------------|:---------------------------------------------------------------------------------------------------:|:--------:|
| CLIPA-B/16(I50,T16) | [download link](https://drive.google.com/file/d/1fURK0K_a3-83jVEI4PVEbnEJb_V6UbGv/view?usp=sharing) |   63.2   |
| CLIPA-L/16(I17,T16) | [download link](https://drive.google.com/file/d/18qqZGOTGOgb3I3JWONuat6qObsgLq7sR/view?usp=sharing) |   67.8   |
| CLIPA_L/16(I37,T8)  | [download link](https://drive.google.com/file/d/1lV7pLORUK04T9QKKx9TpYtMws-AZrib0/view?usp=sharing) |   69.3   |
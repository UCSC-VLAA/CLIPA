
This repo contains official JAX implementation of **CLIPA** in our paper: [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017) 

## Data preparation

### LAION-400M
You can download the LAION-400M dataset using the handy [img2dataset](https://github.com/rom1504/img2dataset) tool. 

We have only tested this jax implementation on Google Cloud TPU. To run this code, you have to choose `tfrecord` format, instead of more common `webdataset` format.
Since LAION-400M is not a TFDS officially supported dataset, for your convenience, we provide some self-implemented scripts (sorry, we know they are crudely written) for post-processing the downloaded tfrecord files [here](../data/laion400m/README.md).

### ImageNet-1K
Download and extract ImageNet data from http://image-net.org/. 
This jax implementation only support reading dataset in tfrecord format. 
Check the [official doc](https://www.tensorflow.org/datasets/cli) for how to prepare the tfds dataset.


## Usage
First, you need to prepare the dataset and create the TPU VM instance. Refer to [TPU_USAGE](../TPU_USAGE.md) for details.

The [configs](configs/) folder contains the detailed configuration of our model and training details.

First 
```
cd clipa_jax
```

To begin with, navigate to the 'scripts/' directory and locate the three scripts [set_up_env.sh](scripts/set_up_env.sh), [pre_training.sh](scripts/pre_training.sh) and [fine_tuning.sh](scripts/fine_tuning.sh) provided. Before executing the scripts, ensure that you specify the TPU VM instance information and dataset path at the top of each file.


Next, you can upload all necessary files to your TPU VM instance and set up the required environment by running the following command:
```
bash scripts/set_up_env.sh
```

Then pre-training can be done by running:
```
bash scripts/pre_training.sh
```

After pre-training, you can fine-tune the model by running:
```
bash scripts/fine_tuning.sh
```

## Results
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">data</th>
<th valign="bottom">sampled</th>
<th valign="bottom">zero-shot IN-1K</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">LAION-400M</td>
<td align="center">2.56B</td>
<td align="center">59.9</td>
</tbody></table>



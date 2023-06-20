
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
<th valign="bottom">Schedule</th>
<th valign="bottom">GPU Hours</th>
<th valign="bottom">Estimated Cost</th>
<th valign="bottom">zero-shot IN-1K</th>
<th valign="bottom">model weight</th>
<!-- TABLE BODY -->
<tr><td align="left">H/14</td>
<td align="center">LAION-2B</td>
<td align="center">12.8B@84 + 512M@224</td>
<td align="center">7776</td>
<td align="center">$12247</td>
<td align="center">78.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1H0ZNNvySDrZ2hmBrc9PrjdQP8YftwGFh/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">LAION-2B</td>
<td align="center">12.8B@84 + 512M@224 + 128M@336</td>
<td align="center">8580</td>
<td align="center">$13616</td>
<td align="center">79.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1IQ0BgWGy0Tsui9iK_wdVifFKc3NfC0BD/view?usp=sharing">download</td>
<tr><td align="left">L/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224</td>
<td align="center">4008</td>
<td align="center">$6318</td>
<td align="center">79.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1oh9IFuX9pD0nd-m4Apl-Z9irX3N-G2_h/view?usp=sharing">download</td>
<tr><td align="left">L/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224 + 128M@336</td>
<td align="center">4520</td>
<td align="center">$7124</td>
<td align="center">80.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1yJD8p27HdZGK2DZ8x64uR6idl_QNkL7w/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@70 + 512M@224</td>
<td align="center">5920</td>
<td align="center">$9324</td>
<td align="center">81.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1f55nRM5DQu0lnLbmCp1GttuTFrJg2UDP/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224</td>
<td align="center">7776</td>
<td align="center">$12247</td>
<td align="center">81.5</td>
<td align="center"><a href="https://drive.google.com/file/d/1Rpd157eay3t8_qsrnSHi_nWeGmhQGTvN/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224 + 128M@336</td>
<td align="center">8580</td>
<td align="center">$13616</td>
<td align="center">81.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1t0k_5m3VVLRyThUjc4JQ_-z29hABuFBV/view?usp=sharing">download</td>
</tbody></table>

Our CLIPA-v2’s GPU hour is estimated using an 8-A100 80GB GPU machine on Google Cloud. 
The corresponding training cost is estimated based on 80GB A100’s cloud pricing.

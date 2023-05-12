## An Inverse Scaling Law for CLIP Training

This repo contains official JAX implementation of **CLIPA** in our paper: [An Inverse Scaling Law for CLIP Training]() 


<p align="center">
  <img src="figs/inverse_scaling_law.png" width="1080">
Overview of the Inverse Scaling Law: larger image/text encoders
enable training with fewer image/text tokens while maintaining competitive performance
</p>


### Usage
First, you need to prepare the dataset and create the TPU VM instance.

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

### Results
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



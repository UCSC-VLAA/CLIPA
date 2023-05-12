## An Inverse Scaling Law for CLIP Training

This repo contains official JAX implementation of **CLIPA** in our paper: [An Inverse Scaling Law for CLIP Training]() 


<p align="center">
  <img src="figs/inverse_scaling_law.png" width="1080">
Overview of the Inverse Scaling Law: larger image/text encoders
enable training with fewer image/text tokens while maintaining competitive performance
</p>


### Usage
After clone this repo and create your own dataset and TPU VM instance

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


### License
This project is under the  Apache 2.0 License.


### Acknowledgement

This project is built on [big vision](https://github.com/google-research/big_vision). Thanks for their contribution and delicate works!

### Contact
If you have any question, please feel free to raise an issue or contact us directily: 
Xianhang Li: xli421@ucsc.edu ;
Zeyu Wang:  zwang615@ucsc.edu

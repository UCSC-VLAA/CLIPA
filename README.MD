# An Inverse Scaling Law for CLIP Training

This repo contains official Pytorch and JAX implementation of **CLIPA** in our paper: [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017) 




<p align="center">
  <img src="clipa_jax/figs/inverse_scaling_law.png" width="1080">
Overview of the Inverse Scaling Law: larger image/text encoders
enable training with fewer image/text tokens while maintaining competitive performance
</p>

## Introduction
CLIP, the first foundation model that connects images and text, has enabled many recent breakthroughs in computer vision. 
However, its associated training cost is prohibitively high, imposing a significant barrier to its widespread exploration. 
In this paper, we present a surprising finding that there exists an _inverse_ scaling law for CLIP training, 
whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. 
Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law.

As a result of this finding, we are able to successfully train CLIP even by using academic resources. 
For example, on an A100 eight-GPU server, our CLIP models achieve zero-shot top-1 ImageNet accuracies of **63.2%** in about **2 days**, 
**67.8%** in about **3 days**, and **69.3%** in about **4 days**. 
By reducing the computation barrier associated with CLIP, we hope to inspire more research in this field, particularly from academics.

<p align="center">
  <img src="clipa_jax/figs/acc_compute.png" width="1080">
</p>


## License
This project is under the  Apache 2.0 License.


## Acknowledgement

The jax repo is built on [big vision](https://github.com/google-research/big_vision), and the pytorch repo is built on [OpenCLIP](https://github.com/mlfoundations/open_clip). 
We've also borrowed some code from [TIMM](https://github.com/huggingface/pytorch-image-models) and [MAE](https://github.com/facebookresearch/mae).
Many thanks to the awesome works from the open-source community!

We are also very grateful that this work is supported by a gift from Open Philanthropy, TPU Research Cloud (TRC) program, and Google Cloud Research Credits program.

## Citation

```
@article{li2023inverse,
      title={An Inverse Scaling Law for CLIP Training}, 
      author={Xianhang Li and Zeyu Wang and Cihang Xie},
      journal={arXiv preprint arXiv:2305.07017},
      year={2023},
}
```
## Contact
If you have any question, please feel free to raise an issue or contact us directily: 
Xianhang Li: xli421@ucsc.edu;
Zeyu Wang:  zwang615@ucsc.edu

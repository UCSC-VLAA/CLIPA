# An Inverse Scaling Law for CLIP Training

This repo contains official Pytorch and JAX implementation of **CLIPA** in our paper: [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017) 

<p align="center">
  <img src="clipa_jax/figs/inverse_scaling_law.png" width="1080">
Overview of the Inverse Scaling Law: larger image/text encoders
enable training with fewer image/text tokens while maintaining competitive performance
</p>




## 📰 News


**[2023.10.4]** We have achieved a successful scale-up of our model to **G/14**, attaining an impressive **83.0%** zero-shot top-1 accuracy on the ImageNet-1K dataset.
For the detailed training configuration, please refer to the [t5x branch](https://github.com/UCSC-VLAA/CLIPA/tree/t5x/clipa_jax/configs/model_bigG). Additionally, you can access the pre-trained and fine-tuned weights for the JAX version in the [Google Drive](https://drive.google.com/drive/folders/1nu5i8mLbOvYq4kovoLP85DjjV8I3N46L?usp=sharing).
<br>

**[2023.9.21]** [Our paper](https://arxiv.org/abs/2305.07017) is accepted by **NeurIPS 2023**!


**[2023.6.16]**  We release [CLIPA-v2](https://arxiv.org/abs/2306.15658). Compared to the prior best publicly available CLIP model, our CLIPA-v2 can be trained significantly faster and yields stronger performance. Our best model is H/14@336x336 on DataComp-1B with an accuracy of 81.8, and its estimated training cost is <$15k! 
<br>

**[Note]** All of our [CLIPA-v2](https://arxiv.org/abs/2306.15658) models were trained on TPU using our [JAX codebase](https://github.com/UCSC-VLAA/CLIPA/tree/master/clipa_jax). We followed the same pre-training process as CLIPA-v1, but with a more efficient fine-tuning strategy. In order to replicate our results, we have provided the **training configuration** (e.g. the H-14 model in this folder [here](https://github.com/UCSC-VLAA/CLIPA/tree/master/clipa_jax/configs/model_h)), along with the **pre-trained weights, configuration, and logs**, which can be found [here](https://drive.google.com/drive/folders/1ArW1-JL7DL43ifg47cIJ4CBeMFIpPkrR?usp=sharing).  
<br>

<p align="center">
  <img src="clipa_jax/figs/CLIPA_v2_teaser.png" width="1080">
</p>


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
<td align="center">12.8B@84 + 512M@224 + 128M@336</td>
<td align="center">8640</td>
<td align="center">$13613</td>
<td align="center">79.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1EiQpLvL51AXEFzJ33e6z58N0dQ83CSux/view?usp=sharing">PyTorch / <a href="https://drive.google.com/file/d/1IQ0BgWGy0Tsui9iK_wdVifFKc3NfC0BD/view?usp=sharing">JAX</td>
<tr><td align="left">L/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224 +128M@336 </td>
<td align="center">4520</td>
<td align="center">$7124</td>
<td align="center">80.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1Vpon6Dn0E3xDfyCIuOW1SPo9haKYvFiD/view?usp=sharing">PyTorch / <a href="https://drive.google.com/file/d/1yJD8p27HdZGK2DZ8x64uR6idl_QNkL7w/view?usp=sharing">JAX</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224 + +128M@336</td>
<td align="center">8640</td>
<td align="center">$13613</td>
<td align="center">81.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1oOACMg3MKXUpG-xn-UrqDWFVEIvenA-F/view?usp=sharing">PyTorch / <a href="https://drive.google.com/file/d/1t0k_5m3VVLRyThUjc4JQ_-z29hABuFBV/view?usp=sharing">JAX</td>
</tbody></table>

Our CLIPA-v2’s GPU hour is estimated using an 8-A100 80GB GPU machine on Google Cloud. 
The corresponding training cost is estimated based on 80GB A100’s cloud pricing.


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

## TPU Usage
Our experiments are conducted on both GPUs and TPUs. Both the JAX and PyTorch implementation enable TPU training.
But how to gain access and setup TPU machines? Check this [brief doc](TPU_USAGE.md).
In a nutshell, you can access TPU machines on Google Cloud **for free**!


## License
This project is under the  Apache 2.0 License.


## Acknowledgement

The jax repo is built on [big vision](https://github.com/google-research/big_vision), and the pytorch repo is built on [OpenCLIP](https://github.com/mlfoundations/open_clip). 
We've also borrowed some code from [TIMM](https://github.com/huggingface/pytorch-image-models) and [MAE](https://github.com/facebookresearch/mae).
Many thanks to the awesome works from the open-source community!

We are also very grateful that this work is supported by a gift from Open Philanthropy, TPU Research Cloud (TRC) program, and Google Cloud Research Credits program.

## Citation

```
@article{li2023clipa,
      title={An Inverse Scaling Law for CLIP Training}, 
      author={Xianhang Li and Zeyu Wang and Cihang Xie},
      journal={arXiv preprint arXiv:2305.07017},
      year={2023},
}
@article{li2023clipav2,
      title={CLIPA-v2: Scaling CLIP Training with 81.1% Zero-shot ImageNet Accuracy within a $10,000 Budget; An Extra $4,000 Unlocks 81.8% Accuracy}, 
      author={Xianhang Li and Zeyu Wang and Cihang Xie},
      journal={arXiv preprint arXiv:2306.15658},
      year={2023},
}
```
## Contact
If you have any question, please feel free to raise an issue or contact us directily: 
Xianhang Li: xli421@ucsc.edu;
Zeyu Wang:  zwang615@ucsc.edu

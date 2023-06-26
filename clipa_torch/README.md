This is a PyTorch/PyTorch-XLA implementation of the paper  [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017).
This repo is heavily based on [OpenCLIP](https://github.com/mlfoundations/open_clip), 
with a few changes including pytorch-xla compatible implementation, uint8 data transfer, and some minor modifications.


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
```
cd CLIPA/clipa_torch
pip install -v .
```

```
import torch
import torch.nn.functional as F
from PIL import Image
from clipa_torch.open_clip import create_model_and_transforms, get_tokenizer

model, _, preprocess = create_model_and_transforms('ViT-H-14-CL32-GAP-BigVision', 
                                                                pretrained='/path/to/ckpt', 
                                                                force_image_size=336,
                                                                image_mean=[0.485, 0.456, 0.406],
                                                                image_std=[0.229, 0.224, 0.225],
                                                                interpolation='bilinear',
                                                                square_resize_only=True,
                                                                )
tokenizer = get_tokenizer('ViT-H-14-CL32-GAP-BigVision')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

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
The CLIPA-v1 checkpoints from this repo should be readily applicable.
As for CLIPA-v2 checkpoints, we have provided some [example testing scripts](scripts/test) for your convenience. 
If you wish to use them in clip_benchmark, check our example testing scripts carefully and follow their instructions to add other CLIP models.

### Model Weights
Here are CLIPA-v1 trained weights on LAION-400M with academic resources. 
All models are pre-trained for 6 epochs with reduced input token lengths and subsequently fine-tuned for 0.36 epoch with full input token lengths.

|                     |                                          Pretrained Model                                           | zero-shot IN-1K |
|---------------------|:---------------------------------------------------------------------------------------------------:|:-----:|
| CLIPA-B/16(I50,T16) | [download](https://drive.google.com/file/d/1fURK0K_a3-83jVEI4PVEbnEJb_V6UbGv/view?usp=sharing) | 63.2  |
| CLIPA-L/16(I17,T16) | [download](https://drive.google.com/file/d/18qqZGOTGOgb3I3JWONuat6qObsgLq7sR/view?usp=sharing) | 67.8  |
| CLIPA_L/16(I37,T8)  | [download](https://drive.google.com/file/d/1lV7pLORUK04T9QKKx9TpYtMws-AZrib0/view?usp=sharing) | 69.3  |

Here are CLIPA-v2 trained weights on the LAION-2B or DataComp-1B dataset. These weights are trained by our jax implementation and converted into pytorch format.
Slight performance variation is possible due to framework difference. Note that these converted weights are not open_clip compatible.
Try our example testing scripts for evaluation.

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
<td align="center">$13616</td>
<td align="center">79.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1EiQpLvL51AXEFzJ33e6z58N0dQ83CSux/view?usp=sharing">download</td>
<tr><td align="left">L/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224</td>
<td align="center">4008</td>
<td align="center">$6318</td>
<td align="center">79.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1PZCZZ-mxHnye_fluCPxqHSdm5SmF9BCT/view?usp=sharing">download</td>
<tr><td align="left">L/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224 + 128M@336</td>
<td align="center">4520</td>
<td align="center">$7124</td>
<td align="center">80.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1Vpon6Dn0E3xDfyCIuOW1SPo9haKYvFiD/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@70 + 512M@224</td>
<td align="center">5920</td>
<td align="center">$9324</td>
<td align="center">81.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1ELP6A3Z_P6QvVpq15rMaywdYSlsyXdzZ/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224</td>
<td align="center">7776</td>
<td align="center">$12247</td>
<td align="center">81.5</td>
<td align="center"><a href="https://drive.google.com/file/d/1JwnpWGgMV29svZRTZR8gPm_2ieZcPAy6/view?usp=sharing">download</td>
<tr><td align="left">H/14</td>
<td align="center">DataCOMP-1B</td>
<td align="center">12.8B@84 + 512M@224 + 128M@336</td>
<td align="center">8640</td>
<td align="center">$13616</td>
<td align="center">81.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1oOACMg3MKXUpG-xn-UrqDWFVEIvenA-F/view?usp=sharing">download</td>
</tbody></table>

Our CLIPA-v2’s GPU hour is estimated using an 8-A100 80GB GPU machine on Google Cloud. 
The corresponding training cost is estimated based on 80GB A100’s cloud pricing.
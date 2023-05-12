# #Copyright @2023 Xianhang Li
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision] and jax-models https://github.com/DarshanDeshpande/jax-models.
# # Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ConvNext.

However, the names of modules are made to match the old ones for easy loading.
"""
import jax
from flax.linen.partitioning import remat
from typing import Optional, Sequence, Union, Iterable, Tuple, Callable

from absl import logging
from helpers import utils
from models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np
import scipy.ndimage

from models.common import DropPath


initializer = nn.initializers.normal(0.02)


class DepthwiseConv2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: str or Sequence[Tuple[int, int]] = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param("bias", self.bias_init,
                           (self.channel_multiplier * input.shape[-1],))

        conv = jax.lax.conv_general_dilated(
            input,
            w,
            self.stride,
            self.padding,
            (1,) * len(self.kernel_shape),
            (1,) * len(self.kernel_shape),
            ("NHWC", "HWIO", "NHWC"),
            input.shape[-1],
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv


class ConvNeXtBlock(nn.Module):
    dim: int = 256
    layer_scale_init_value: float = 0.
    drop_path: float = 0.0
    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = DepthwiseConv2D(
            (7, 7), weights_init=initializer, name="dwconv")(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer, name="pwconv2")(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x

        x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class _Model(nn.Module):
    """ViT model."""

    num_classes: Optional[int] = None
    dims: Iterable = (96, 192, 384, 768)
    depths: Iterable = (3, 3, 9, 3)
    layer_scale_init_value: float = 0.0
    head_init_scale: float = 1.0
    dropout: float = 0.0
    drop_path: float = 0.0
    head_zeroinit: bool = False
    remat_policy: str = 'none'

    @nn.compact
    def __call__(self, image, *, train=False, mask_ratio=0):
        out = {}

        dp_rates = jnp.linspace(0, self.drop_path, sum(self.depths))
        curr = 0

        # Stem
        x = nn.Conv(
            self.dims[0], (4, 4), 4, kernel_init=initializer, name="embedding"
        )(image)
        x = nn.LayerNorm(name="downsample_layers01")(x)

        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0],
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"encoderblock_{j}",
            )(x, not train)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"downsample_layers{i + 1}0")(x)
            x = nn.Conv(
                self.dims[i + 1],
                (2, 2),
                2,
                kernel_init=initializer,
                name=f"downsample_layers{i + 1}1",
            )(x)

            for j in range(self.depths[i + 1]):
                x = ConvNeXtBlock(
                    self.dims[i + 1],
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i + 1}{j}",
                )(x, not train)

            curr += self.depths[i + 1]

        if self.num_classes:

            x = nn.LayerNorm(name="norm")(jnp.mean(x, [1, 2]))
            x = out["logits"] = nn.Dense(self.num_classes, kernel_init=nn.initializers.normal(
                stddev=self.dims[-1] ** -0.5), name="head")(x)

        return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
    """Factory function, because linen really don't like what I'm doing!"""
    return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "dims": {"Ti": (96, 192, 384, 768), "S": (96, 192, 384, 768), "B": (128, 256, 512, 1024)}[v],
        "depths": {"Ti": (3, 3, 9, 3), "S": (3, 3, 27, 3), "B": (3, 3, 27, 3)}[v],
        # pylint:enable=line-too-long
    }


def resample_posemb(old, new):
    """This function implements "high-res finetuning" for transformer models."""
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    if old.shape == new.shape:
        return old

    logging.info("ViT: resize %s to %s", old.shape, new.shape)
    gs_old = int(np.sqrt(old.shape[1]))
    gs_new = int(np.sqrt(new.shape[1]))
    logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
    grid = old.reshape(gs_old, gs_old, -1)

    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    grid = scipy.ndimage.zoom(grid, zoom, order=1)
    grid = grid.reshape(1, gs_new * gs_new, -1)
    return jnp.array(grid)


def fix_old_checkpoints(params):
    """Fix small bwd incompat that can't be resolved with names in model def."""

    params = flax.core.unfreeze(
        flax.training.checkpoints.convert_pre_linen(params))

    # Original ViT paper variant had posemb in a module:
    if "posembed_input" in params["Transformer"]:
        logging.info("ViT: Loading and fixing VERY old posemb")
        posemb = params["Transformer"].pop("posembed_input")
        params["pos_embedding"] = posemb["pos_embedding"]

    # Widely used version before 2022 had posemb in Encoder:
    if "pos_embedding" in params["Transformer"]:
        logging.info("ViT: Loading and fixing old posemb")
        params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

    # Old vit.py used to first concat [cls] token, then add posemb.
    # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
    # so we changed to add posemb then concat [cls]. We can recover the old
    # checkpoint by manually summing [cls] token and its posemb entry.
    if "pos_embedding" in params:
        pe = params["pos_embedding"]
        if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
            logging.info("ViT: Loading and fixing combined cls+posemb")
            pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
            if "cls" in params:
                params["cls"] += pe_cls

    # MAP-head variants during ViT-G development had it inlined:
    if "probe" in params:
        params["MAPHead_0"] = {
            k: params.pop(k) for k in [
                "probe",
                "MlpBlock_0",
                "MultiHeadDotProductAttention_0",
                "LayerNorm_0"]}

    return params


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
    """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
    del model_cfg

    #init_file = VANITY_NAMES.get(init_file, init_file)
    restored_params = utils.load_params(None, init_file)

    #restored_params = fix_old_checkpoints(restored_params)

    # possibly use the random init for some of the params (such as, the head).
    restored_params = common.merge_params(
        restored_params, init_params, dont_load)

    # resample posemb if needed.
    if init_params and "pos_embedding" in init_params:
        restored_params["pos_embedding"] = resample_posemb(
            old=restored_params["pos_embedding"],
            new=init_params["pos_embedding"])

    if 'pos_embedding' in dont_load:
        logging.info(
            'fixed pos_embedding cannot be stored, re-intialized needed')
        _, l, c = init_params["pos_embedding"].shape
        h, w = (l - 1)**.5, (l - 1)**.5
        #restored_params['pos_embedding'] = get_2d_sincos_pos_embed(c, h, cls_token=True)
        restored_params['pos_embedding'] = posemb_sincos_2d(
            h, w, c, cls_token=True)

    from helpers.utils import recover_dtype
    restored_params = jax.tree_map(recover_dtype, restored_params)
    return restored_params

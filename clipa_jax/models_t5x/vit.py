# #Copyright @2023 Xianhang Li
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
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

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

from typing import (Any, Callable, Optional, Tuple)

import jax
from flax.linen.partitioning import remat
from typing import Optional, Sequence, Union

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
import t5x.layers


def posemb_sincos_2d(
        h,
        w,
        width,
        temperature=10_000.,
        dtype=jnp.float32,
        cls_token=False):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate(
        [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    if cls_token:
        pe = jnp.concatenate([jnp.zeros([1, width]), pe], axis=0)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(
        self,
        typ,
        seqshape,
        width,
        name,
        dtype=jnp.float32,
        cls_token=False):
    if typ == "learn":
        num_token = 1 if cls_token else 0
        return t5x.layers.param_with_axes(name,
                          nn.initializers.normal(stddev=width ** -0.5),
                          (1, np.prod(seqshape) + num_token, width), dtype, axes=("_null0",))
    elif typ == "sincos2d":
        return posemb_sincos_2d(
            *seqshape,
            width,
            dtype=dtype,
            cls_token=cls_token)

    else:
        raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout: float = 0.0
    fc_init: Callable = nn.initializers.xavier_uniform()
    proj_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Applies Transformer MlpBlock module."""
        inits = dict(
            kernel_init=self.fc_init,
            bias_init=nn.initializers.zeros,
        )

        n, l, d = x.shape  # pylint: disable=unused-variable

        x = t5x.layers.Dense(features=self.mlp_dim or 4 * d, **inits, kernel_axes=("embed", "mlp"),)(x)
       # x = nn.gelu(x, approximate=False)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.partitioning.with_sharding_constraint(x, ("batch", "length", "mlp"))
        x =  t5x.layers.Dense(
            features=d,
            kernel_init=self.proj_init,
            bias_init=nn.initializers.zeros,
            kernel_axes=("mlp", "embed"),)(x)
        return x



class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    depth: int = 12

    @nn.compact
    def __call__(self, x, deterministic=True):
        width = x.shape[-1]
        init_std = {
            'proj': (width ** -0.5) * ((2 * self.depth) ** -0.5),
            'attn': width ** -0.5,
            'fc': (2 * width) ** -0.5
        }
        out = {}
        x = nn.partitioning.with_sharding_constraint(x, ("batch", "length", "embed"))
        y = t5x.layers.LayerNorm(axes=("embed",))(x)
        y = nn.partitioning.with_sharding_constraint(y, ("batch", "length", "embed"))
        y = out["sa"] = t5x.layers.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_kernel_init=nn.initializers.normal(stddev=init_std['attn']),
            out_kernel_init=nn.initializers.normal(stddev=init_std['proj']),
            bias_init=nn.initializers.zeros,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = out["+sa"] = x + y
        x = nn.partitioning.with_sharding_constraint(x, ("batch", "length", "embed"))
        y = t5x.layers.LayerNorm(axes=("embed",))(x)
        y = nn.partitioning.with_sharding_constraint(y, ("batch", "length", "embed"))
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim, dropout=self.dropout,
            fc_init=nn.initializers.normal(stddev=init_std['fc']),
            proj_init=nn.initializers.normal(stddev=init_std['proj']),
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = out["+mlp"] = x + y
        x = nn.partitioning.with_sharding_constraint(x, ("batch", "length", "embed"))
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    depth: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    remat_policy: str = "none"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        dpr = [
            float(x) for x in np.linspace(
                0,
                self.drop_path,
                self.depth)]  # drop path decay
        # Input Encoder
        BlockLayer = Encoder1DBlock
        if self.remat_policy not in (None, "none"):
            logging.info(f"remat policy: {self.remat_policy}")
            if self.remat_policy == "minimal":
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            else:
                policy = None
            logging.info(f"activation checkpointing {self.remat_policy}")
            BlockLayer = remat(  # pylint: disable=invalid-name
                Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
            )  # "deterministic" is a static argument in Encoder1DBlock

        for lyr in range(self.depth):
            x, out[f"block{lyr:02d}"] = BlockLayer(
                name=f"encoderblock_{lyr}",
                mlp_dim=self.mlp_dim, num_heads=self.num_heads, dropout=self.dropout, drop_path=dpr[lyr])(x, deterministic)
         # x, out[f"block{lyr:02d}"] = block(x, deterministic)
        # Alias for last block, but without the number in it.
        out["pre_ln"] = x

        return x, out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # TODO
        n, l, d = x.shape  # pylint: disable=unused-variable
        probe = self.param("probe", nn.initializers.xavier_uniform(),
                           (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform())(probe, x)

        # TODO: dropout on head?
        y = nn.LayerNorm()(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
        return x[:, 0]


class _Model(nn.Module):
    """ViT model."""

    num_classes: Optional[int] = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: Union[int, bool] = False
    dropout: float = 0.0
    drop_path: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = False
    patch_embeding: str = 'conv'
    remat_policy: str = 'none'
    partitioner: Any = None

    @nn.compact
    def __call__(self, image, *, train=False, mask_ratio=0):
        out = {}

        if self.patch_embeding == 'conv':
            # Patch extraction
            x = out["stem"] = t5x.layers.Conv(
                features=self.width,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                use_bias=False,
                padding="VALID",
                name="embedding",
                kernel_axes=("_null0", "_null1", "_null2", "embed"),)(image)

            n, h, w, c = x.shape
            x = jnp.reshape(x, [n, h * w, c])
        else:
            p = self.patch_size[0]
            h = w = image.shape[2] // p
            x = image.reshape((image.shape[0], h, p, w, p, 3))
            x = jnp.einsum('nhpwqc->nhwpqc', x)
            x = x.reshape((image.shape[0], h * w, p ** 2 * 3))
            x = out["stem"] = nn.Dense(self.width, name="embedding")(x)
            n, l, c = x.shape

        # if self.pool_type == "tok":
        cls = t5x.layers.param_with_axes("cls", nn.initializers.zeros, (1, 1, c), x.dtype,  axes=("_null0", "_null1", "embed"))

        # # timm initial
        x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        x = out["with_posemb"] = x + get_posemb(
            self, self.posemb, (h, w), c, "pos_embedding", x.dtype, cls_token=True)
        n, l, c = x.shape  # pylint: disable=unused-variable
        x = nn.Dropout(rate=self.dropout)(x, not train)

        if mask_ratio > 0:
            cls_token = x[:, :1]
            rng_mask = self.make_rng('random_mask')
            x, _, _ = self.random_masking(
                x[:, 1:], mask_ratio=mask_ratio, rng_mask=rng_mask)
            x = jnp.concatenate([cls_token, x], axis=1)
        encoder_blocks = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            drop_path=self.drop_path,
            remat_policy=self.remat_policy,
            name="Transformer")

        x, out["encoder"] = encoder_blocks(
            x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
        elif self.pool_type == "gap":
            x = jnp.mean(x[:, 1:], axis=1)
            x = out["head_input"] = t5x.layers.LayerNorm(name="encoder_norm", axes=("embed",))(x)
            encoded = encoded[:, 1:]
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = t5x.layers.LayerNorm(name="encoder_norm", axes=("embed",))(x)
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        if self.num_classes:
            head = t5x.layers.Dense(
                features=self.num_classes,
                name="head",
                kernel_init=nn.initializers.normal(
                    stddev=self.width ** -0.5),
                use_bias=False,
                kernel_axes=("_null0", "_null1"),
               # kernel_axes=("_null0", "embed"),
            )
            x = out["logits"] = head(x)

        return x, out

    def random_masking(self, x, mask_ratio, rng_mask=None):

        x = nn.partitioning.with_sharding_constraint(
            x, ("batch", "length", "embed")
        )

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))


        noise = jax.random.uniform(rng_mask, (N, L))

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #x_masked = batched_gather(x, ids_keep)

        x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], 1)

        x_masked = nn.partitioning.with_sharding_constraint(
            x_masked, ("batch", "length", "embed")
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones((N, L))
        mask = mask.at[:, :len_keep].set(0)

        #mask = batched_gather(mask, ids_restore)
        mask = jnp.take_along_axis(mask, ids_restore, 1)
        return x_masked, mask, ids_restore


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
        "width": {"Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "H": 1280, "g": 1408, "G": 1664, "e": 1792}[v],
        "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "H": 32, "g": 40, "G": 48, "e": 56}[v],
        "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "H": 5120, "g": 6144, "G": 8192, "e": 15360}[v],
        "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "H": 16, "g": 16, "G": 16, "e": 16}[v],
        # pylint:enable=line-too-long
        **patch
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

    init_file = VANITY_NAMES.get(init_file, init_file)
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


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",
    # pylint: disable=line-too-long
    # pylint: enable=line-too-long
}

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


from flax.linen.partitioning import remat
import jax
from typing import Optional, Sequence, Union

from absl import logging
from helpers import utils
from models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np

from models.common import DropPath
import functools
from typing import (Any, Callable, Optional, Tuple)
from flax.linen.linear import DenseGeneral

from flax.linen.module import compact
from flax.linen.module import merge_param

import t5x.layers

Array = Any


def posemb_sincos_1d(
        max_len,
        width,
        min_scale=1.,
        max_scale=10_000.,
        dtype=jnp.float32,
        cls_token=False):
    """Follows the MoCo v3 logic."""
    d_feature = width
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe, dtype=dtype)


def get_posemb(
        self,
        typ,
        max_len,
        width,
        name,
        dtype=jnp.float32,
        cls_token=False):
    if typ == "learn":
        num_token = 1 if cls_token else 0
        return  t5x.layers.param_with_axes(name,
                          # nn.initializers.variance_scaling(scale=0.3072,
                          # distribution="truncated_normal", mode='fan_out'), #
                          # timm trunc
                          nn.initializers.normal(stddev=0.01),
                          (1, max_len, width), dtype, axes=("_null0",))
    elif typ == "sincos1d":
        return posemb_sincos_1d(
            max_len,
            width,
            dtype=dtype,
            cls_token=cls_token)
        # return get_2d_sincos_pos_embed(width, seqshape[0], dtype=dtype,
        # cls_token=cls_token)
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
      #  x = t5x.layers.with_sharding_constraint(x, ("batch", "length", "embed"))
        n, l, d = x.shape  # pylint: disable=unused-variable
        x = t5x.layers.Dense(features=self.mlp_dim or 4 * d, **inits, kernel_axes=("embed", "mlp"),)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.partitioning.with_sharding_constraint(x, ("batch", "length", "mlp"))
        x =  t5x.layers.Dense(
            features=d,
            kernel_init=self.proj_init,
            bias_init=nn.initializers.zeros,
            kernel_axes=("mlp", "embed"),)(x)
      #  x = t5x.layers.with_sharding_constraint(x, ("batch", "length", "embed"))
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
        y =t5x.layers.LayerNorm(axes=("embed",))(x)
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

from flax.linen.partitioning import remat

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


class _Model(nn.Module):
    """ViT model."""

    num_classes: Optional[int] = None
    width: int = 512
    depth: int = 12
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    posemb: str = "learn"  # Can also be "sincos2d"
    pool_type: str = "last"  # Can also be "map" or "tok"
    vocab_size: int = 32000
    head_zeroinit: bool = False
    drop_path: float = 0.0
    remat_policy: str = 'none'

    @nn.compact
    def __call__(self, text, *, train=False, mask_ratio=0):
        out = {}

        embedding = t5x.layers.Embed(
            num_embeddings=self.vocab_size,
            features=self.width,
            embedding_init=nn.initializers.normal(
                stddev=0.02),
            axes=["classes", "embed"],  # do not use 'vocab'
        )
        x = out['embedded'] = embedding(text)

        n, l, d = x.shape  # pylint: disable=unused-variable
        # Add posemb before adding extra token.

        x = x + get_posemb(
            self, self.posemb, l, d, "pos_embedding", x.dtype, cls_token=True)

        x = nn.Dropout(rate=self.dropout)(x, not train)
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

        x = out["norm"] = t5x.layers.LayerNorm(name="encoder_norm", axes=("embed",))(x)

        if self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x[:, 1:], axis=1)
        elif self.pool_type == "last":
            x = out["head_input"] = x[:, -1, :]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        if self.num_classes:

            head = t5x.layers.Dense(
                features=self.num_classes,
                name="head",
                use_bias=False,
                kernel_init=nn.initializers.normal(
                    stddev=self.width ** -0.5),
                kernel_axes=("_null0", "_null1"),)
              #  kernel_axes = ("_null0", "embed"),)
            x = out["logits"] = head(x)

        return x, out

    def random_masking(self, x, mask_ratio, rng_mask=None):

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
        "width": {"Ti": 192, "S": 384, "M": 512, "B": 512, "L": 768, "H": 1024, "g": 1280, "G": 1280, "e": 1792}[v],
        "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 12, "H": 24, "g": 32, "G": 32, "e": 56}[v],
        "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 2048, "L": 3072, "H": 3072, "g": 5120, "G": 5120, "e": 15360}[v],
        "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 8, "L": 12, "H": 16, "g": 20, "G": 20, "e": 16}[v],
        # pylint:enable=line-too-long

    }

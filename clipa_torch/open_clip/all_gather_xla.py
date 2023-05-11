""" All-gather operation that supports differentiation.

Reference: https://github.com/pytorch/xla

Hacked together by / Copyright 2023 Zeyu Wang
"""
import torch
import torch_xla.core.xla_model as xm
import torch_xla
from torch_xla.core.functions import all_gather as xlaf_all_gather
import torch_xla.distributed.xla_multiprocessing as xmp



class _AlltoAll(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, split_dim, concat_dim, split_count, groups, pin_layout):
    ctx.split_dim = split_dim
    ctx.concat_dim = concat_dim
    ctx.split_count = split_count
    ctx.groups = groups
    ctx.pin_layout = pin_layout
    ctx.ordinal = xm.get_ordinal()
    ctx.world_size = xm.xrt_world_size()
    return xm.all_to_all(input, split_dim, concat_dim, split_count, groups=groups, pin_layout=pin_layout)

  @staticmethod
  def backward(ctx, grad_output):
    return xm.all_to_all(grad_output, ctx.split_dim, ctx.concat_dim,
                         ctx.split_count, groups=ctx.groups, pin_layout=ctx.pin_layout) \
           + (None, None, None, None, None)


class _AllGather(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, dim):
    ctx.dim = dim
    ctx.ordinal = xm.get_ordinal()
    ctx.world_size = xm.xrt_world_size()
    return xm.all_gather(input, dim=dim)

  @staticmethod
  def backward(ctx, grad_output):
    gxs = _AlltoAll.apply(grad_output, ctx.dim, ctx.dim, ctx.world_size, None, True)
    assert gxs.shape[0] % ctx.world_size == 0, 'shape is divisible by world size!'
    gxs = torch.split(gxs, gxs.shape[0] // ctx.world_size)
    gx = torch.sum(torch.stack(gxs), dim=ctx.dim)
    return gx, None


def all_gather(value, dim=0):
  """Performs an all-gather operation along a given dimension.

  This is the same as `xm.all_gather()` but supports autograd differentiation.

  Args:
    value (torch.Tensor): The input tensor.
    dim (int): The gather dimension.
      Default: 0
  Returns:
    A tensor which has, in the ``dim`` dimension, all the values from the
    participating replicas.
  """
  return _AllGather.apply(value, dim)
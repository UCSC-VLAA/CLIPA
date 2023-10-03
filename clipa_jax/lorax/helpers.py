import functools

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map_with_path, DictKey, SequenceKey

from .constants import LORA_FREEZE, LORA_FULL
from .transform import EmptyNode, LoraNode, custom_tree_map

def init_lora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1., is_leaf=None):
    def freeze_getter(param, spec_val):
        if spec_val == LORA_FULL:
            return EmptyNode
        return param


    def tune_getter(path, param, spec_val):
        if spec_val == LORA_FREEZE:
            return EmptyNode
        if spec_val == LORA_FULL:
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')
        if len(param.shape) == 2:
            b_dim, a_dim = param.shape
           # print(f'b_dim: {b_dim}, a_dim: {a_dim}, spec_val: {spec_val}')
            b = jnp.zeros((b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(rng, (spec_val, a_dim), dtype=dtype) * stddev
            b = jax.device_put(b, device=jax.devices("cpu")[0])
            a = jax.device_put(a, device=jax.devices("cpu")[0])
            return LoraNode(a, b, alpha=alpha)

        if len(param.shape) == 3 :
              s_x, s_y, s_z = param.shape
              if s_z > s_x:
                   b_dim, a_dim = s_x*s_y, s_z

              else:
                    b_dim, a_dim = s_x, s_y*s_z

              b = jnp.zeros((b_dim, spec_val), dtype=dtype)
              a = jax.random.normal(rng, (spec_val, a_dim), dtype=dtype) * stddev
              b = jax.device_put(b, device=jax.devices("cpu")[0])
              a = jax.device_put(a, device=jax.devices("cpu")[0])
              return LoraNode(a, b, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        return LoraNode(a, b, alpha=alpha)

    return (
        jax.tree_map(freeze_getter, param_tree, spec, is_leaf=is_leaf),
        jax.tree_util.tree_map_with_path(tune_getter, param_tree, spec, is_leaf=is_leaf)
    )

def simple_spec(params, decision_fn=None, tune_vectors=False, is_leaf=None):
    """
    Create a simple lora spec for a pytree
    Args:
        params: pytree of parameters
        tune_vectors: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function that takes a string marking the position in the tree and the parameter,
            and returns the spec value for that parameter
            The position strings are of the form key1/key2/key3 etc. For lists or tuples the key is
            the index
    """
    if decision_fn is None:
        def decision_fn(*args):
            return LORA_FREEZE

    def full_fn(path, arr):
        if len(arr.shape) < 2:
            return LORA_FULL if tune_vectors else LORA_FREEZE

        path_str = '/'.join(
            str(node.key
                if isinstance(node, DictKey) else
                node.idx
                if isinstance(node, SequenceKey)
                else
                str(node)
            ) for node in path
        )
        value = decision_fn(path_str, arr)
        return value

    return tree_map_with_path(full_fn, params, is_leaf=is_leaf)

def merge_params(frozen_params, tunable_params, destructive=True, use_scaling=True):
    """Re-merge LoRA parameters.
    Arguments:
        destructive: If True, the buffers in frozen_params may be freed to save memory.
        use_scaling: Whether to multiply LoRA params by alpha/r
    """

    def merge(frozen, tunable):
        if tunable is EmptyNode:
            return frozen
        if frozen is EmptyNode:
            return tunable
        tuned_para = tunable.evaluate(rescale=use_scaling)
        tuned_para = jnp.reshape(tuned_para, frozen.shape)
        new_param = frozen + tuned_para
        if destructive:
            frozen.device_buffer.delete()
        return new_param
    return custom_tree_map(
        merge,
        frozen_params,
        tunable_params,
    )

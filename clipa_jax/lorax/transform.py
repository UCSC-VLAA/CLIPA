from functools import partial, wraps
from itertools import chain, repeat
import logging
import warnings

import jax
from jax import api_util
import jax.experimental.pjit
import jax.linear_util as lu
import jax.numpy as jnp
from jax._src.lax.lax import _convert_element_type
from  jax.util import safe_map
import numpy as np

from .constants import LORA_FREEZE, LORA_FULL

LORA_IMPLS = {}

class LoraNode:
    def __init__(self, a, b, alpha=1.):
        self.a = a
        self.b = b
        self.alpha = alpha
        #for parameter  profiling only
        if isinstance(a, int):
            self.shape = 1
        elif isinstance(a, np.ndarray):
            self.shape = a.shape
        else:
            pass

       # self.shape = a.shape if not isinstance(a, int) else 1

    def __str__(self):
        return f'{type(self).__name__}(a={self.a}, b={self.b}, alpha={self.alpha})'

    def __repr__(self):
        return str(self)

    def evaluate(self, rescale=True):
        result = self.b @ self.a
        if rescale:
            result *= self.get_scale()
        return result

    def get_scale(self):
        return self.alpha / self.b.shape[1]

    def mean(self):
        return np.mean(np.asarray(self.a))

    def std(self):
        return  np.std(self.a)



class EmptyNodeCls:
    def __init__(self):
        self.shape = 0

    def mean(self):
        return 0

    def std(self):
        return 0

jax.tree_util.register_pytree_node(
    LoraNode,
    lambda node: ((node.a, node.b), node.alpha),
    lambda alpha, xs: LoraNode(*xs, alpha=alpha)
)

EmptyNode = EmptyNodeCls()
jax.tree_util.register_pytree_node(EmptyNodeCls, lambda _: ((), None), lambda _, x: EmptyNode)

def leaf_pred(x):
    return x is EmptyNode or isinstance(x, LoraNode)

custom_tree_map = partial(jax.tree_util.tree_map, is_leaf=leaf_pred)
custom_tree_leaves = partial(jax.tree_util.tree_leaves, is_leaf=leaf_pred)

def lora_to_orig(freeze_param, tune_param):
    if freeze_param is EmptyNode:
        return tune_param
    return freeze_param

def lora(f, argnums=0, use_scaling=True):
    if isinstance(argnums, int):
        argnums = (argnums,)

    @wraps(f)
    def wrapper(*args, **kwargs):
        orig_args = [*args]
        for argnum in argnums:
            orig_args[argnum] = custom_tree_map(lora_to_orig, *args[argnum])
            assert not any(node is EmptyNode for node in custom_tree_leaves(orig_args[argnum]))

        shape_args, shape_kwargs = jax.tree_map(
            lambda x: jax.core.get_aval(x) if isinstance(x, jax.core.Tracer) else x, 
            (orig_args, kwargs)
        )
        closed_jaxpr = jax.make_jaxpr(f)(*shape_args, **shape_kwargs)
        out_shape = jax.eval_shape(f, *shape_args, **shape_kwargs)
        out_structure = jax.tree_util.tree_structure(out_shape)

        paired_args = []
        for i, arg in enumerate(args):
            if i in argnums:
                frozen_leaves, lora_leaves = (custom_tree_leaves(a) for a in arg)
            else:
                frozen_leaves = repeat(EmptyNode)
                lora_leaves = jax.tree_util.tree_leaves(arg)

            for frozen_leaf, lora_leaf in zip(frozen_leaves, lora_leaves):
                paired_args.append((frozen_leaf, lora_leaf))

        paired_args.extend((EmptyNode, leaf) for leaf in jax.tree_util.tree_leaves(kwargs))

        jaxpr = closed_jaxpr.jaxpr

        result = lora_interpreter(jaxpr, closed_jaxpr.literals, *paired_args, use_scaling=use_scaling)
        unflattened_result = jax.tree_util.tree_unflatten(out_structure, result)

        return unflattened_result
    return wrapper

def materialize_val(val, use_scaling=True):
    if not isinstance(val, tuple):
        return val

    freeze, tune = val
    if freeze is EmptyNode:
        return tune
    if tune is EmptyNode:
        return freeze
    # reshape the 2d paras into 3d since I use 2d matmul in lora
    tuned_para = tune.evaluate(rescale=use_scaling)
    tuned_para = jnp.reshape(tuned_para, freeze.shape)
    full = freeze + tuned_para
    warnings.warn(f'LoRA matrix of shape {full.shape} was materialized')
    return full


def is_lora_tuple(val):
    return isinstance(val, tuple) and isinstance(val[1], LoraNode)

def lora_interpreter(jaxpr, literals, *args, use_scaling=True):
    env = {}

    def read(var):
        if isinstance(var, jax.core.Literal):
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.constvars, literals)
    safe_map(write, jaxpr.invars, args)

    for eqn in jaxpr.eqns:
        # TODO: run inside other interpreters in a smarter way
        use_default_eval = True
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        args = safe_map(read, eqn.invars)
        if eqn.primitive.name == 'pjit' and eqn.params['name'] == '_einsum':
            params = dict(eqn.params)
            pjit_jaxpr = params.pop('jaxpr')
            literals = pjit_jaxpr.literals
            subjaxpr = pjit_jaxpr.jaxpr
            ans = jax.experimental.pjit.pjit(
                partial(lora_interpreter, subjaxpr),
            )(literals, *args)
            use_default_eval = False
        elif eqn.primitive.name == 'remat2':
            subjaxpr = eqn.params['jaxpr']
            ans = jax.remat(
                partial(lora_interpreter, subjaxpr)
            )([], *args)
            use_default_eval = False
        elif eqn.primitive.name in LORA_IMPLS:
            if any(safe_map(is_lora_tuple, args)):
                ans = LORA_IMPLS[eqn.primitive.name](eqn, *args, use_scaling=use_scaling)
                use_default_eval = ans is None

        if use_default_eval:
            materialized_args = safe_map(partial(materialize_val, use_scaling=use_scaling), args)
            ans = eqn.primitive.bind(*subfuns, *materialized_args, **bind_params)
        if not eqn.primitive.multiple_results:
            ans = [ans]
        safe_map(write, eqn.outvars, ans)

    return safe_map(read, jaxpr.outvars)

def _reversed_dot(lhs, rhs, dimension_numbers):
    return jax.lax.dot_general(
        rhs,
        lhs,
        dimension_numbers

    )

def eval_lora_dot(eqn, lhs, rhs, use_scaling=True):
    dimension_numbers = eqn.params['dimension_numbers']
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        warnings.warn('Lorax does not support batched matmuls')
        return None
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        warnings.warn('Lorax only supports matmul')
        return None

    lhs_contract, = lhs_contract
    rhs_contract, = rhs_contract

    use_lhs = is_lora_tuple(lhs)

    use_rhs = is_lora_tuple(rhs)
    if use_lhs and use_rhs:
        warnings.warn('Product of two LoRA matrices is not implemented so RHS will be materialized')
        use_rhs = False

    if use_lhs:
        a_first = lhs_contract == 1
        fixed_arg = materialize_val(rhs, use_scaling=use_scaling)
        frozen, lora = lhs
        fn = jax.lax.dot_general

        second_dimension_numbers = (
            ((lhs_contract,), (0,)),
            dimension_numbers[1]
        )

    elif use_rhs:
        a_first = rhs_contract == 1
        fixed_arg = materialize_val(lhs, use_scaling=use_scaling)
        frozen, lora = rhs
        fn = _reversed_dot

        final_lhs_dim = len(fixed_arg.shape) - 1

        second_dimension_numbers = (
            ((final_lhs_dim,), (rhs_contract,)),
            dimension_numbers[1]
        )
    else:
        raise ValueError('No lora node')

    orig_product = fn(
        frozen,
        fixed_arg,
        dimension_numbers=dimension_numbers
    )

    first, second = (lora.a, lora.b) if a_first else (lora.b, lora.a)
    lora_product = fn(
            first,
            fixed_arg,
            dimension_numbers=dimension_numbers
        )

    lora_product = fn(
            second,
            lora_product,
            dimension_numbers=second_dimension_numbers
        )
    if use_scaling:
        lora_product *= lora.get_scale()
    # reshape the 2d paras into 3d since I use 2d matmul in lora
    lora_product = jnp.reshape(lora_product, orig_product.shape)
    return orig_product + lora_product

def eval_lora_conv(eqn, inp, kernel, use_scaling=True):
    if not is_lora_tuple(kernel):
        return None

    if is_lora_tuple(inp):
        warnings.warn('Lorax only supports convolutions with the a LoRA kernel, so the input will be materialized')

    inp = materialize_val(inp)

    dimension_numbers = eqn.params['dimension_numbers']
    if not dimension_numbers.rhs_spec[:1] != (
        len(dimension_numbers.rhs_spec) - 1,
        len(dimension_numbers.rhs_spec) - 2,
    ):
        raise ValueError('Lorax only supports convolutions with shape (..., in_features, out_features)')

    frozen, lora = kernel
    orig = jax.lax.conv_general_dilated(
        inp,
        frozen,
        **eqn.params
    )

    kwargs = eqn.params.copy()
    lora_product = jax.lax.conv_general_dilated(
        inp,
        lora.b,
        **kwargs
    )

    kwargs['window_strides'] = (1,) * (len(dimension_numbers.rhs_spec) - 2)
    kwargs['padding'] = 'VALID'
    lora_product = jax.lax.conv_general_dilated(
        lora_product,
        lora.a,
        **kwargs
    )

    if use_scaling:
        lora_product *= lora.get_scale()
    return orig + lora_product

def eval_lora_gather(eqn, arr, indices, use_scaling=True):
    if not is_lora_tuple(arr):
        return None

    indices = materialize_val(indices)

    dimension_numbers = eqn.params['dimension_numbers']
    if dimension_numbers.offset_dims != (len(indices.shape) - 1,):
        return None

    frozen, lora = arr
    constraint_dim = lora.b.shape[-1]

    slice_sizes = eqn.params['slice_sizes']

    if slice_sizes != (1, lora.a.shape[1]):
        return None

    new_params = eqn.params.copy()
    new_params['slice_sizes'] = (1, constraint_dim)


    orig = jax.lax.gather(frozen, indices, **eqn.params)

    lora_product = jax.lax.gather(lora.b, indices, **new_params)

    lora_product = lora_product @ lora.a
    if use_scaling:
        lora_product *= lora.get_scale()

    return orig + lora_product

def eval_lora_transpose(eqn, arg, **kwargs):
    if not len(arg[0].shape) == 2 and eqn.params['permutation'] == (1, 0):
        return None
    frozen, lora = arg

    frozen_T = frozen.T
    lora_T = LoraNode(lora.b.T, lora.a.T, alpha=lora.alpha)
    return frozen_T, lora_T

def eval_lora_convert_element_type(eqn, arg, **kwargs):
    lora_node = arg[1]
    frozen = _convert_element_type(arg[0], **eqn.params)

    a = _convert_element_type(lora_node.a, **eqn.params)
    b = _convert_element_type(lora_node.b, **eqn.params)

    return frozen, LoraNode(a, b, alpha=lora_node.alpha)


LORA_IMPLS.update({
    'dot_general': eval_lora_dot,
    'conv_general_dilated': eval_lora_conv,
    'gather': eval_lora_gather,
    'transpose': eval_lora_transpose,
    'convert_element_type': eval_lora_convert_element_type
})

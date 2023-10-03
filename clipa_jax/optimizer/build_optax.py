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

"""Gradient transformations and other optax utilities."""

import operator
import jax
import jax.numpy as jnp
import optax
import helpers.utils as u


def steps(prefix, config, data_size=None, batch_size=None, total_steps=None,
          default=ValueError):
    """Gets duration named `prefix` out of `config` and converts it to steps.

    Using this function to access a configuration value that denotes some kind
    of duration (eg training time, warmup, checkpoint frequency, ...) allows the
    duration to be specified in terms of steps, epochs, examples, or percent of
    training time, and converts any of these into steps, such that the training
    code only deals with steps.
    If the result is not an integer step number, it is rounded to the nearest one.

    Args:
      prefix: The name of the duration to query. The actual config fields can
        then be one of `prefix_steps`, `prefix_examples`, or `prefix_epochs`.
      config: The dictionary (config) from which to read the duration.
      data_size: The total number of training examples in one epoch.
      batch_size: The number of examples processed per step.
      total_steps: The total number of training steps to run.
      default: The default value to return when no duration of the name `prefix`
        is found in the `config`. Set to `ValueError` (the default) to raise an
        error instead of returning a default value.

    Returns:
      The number of steps from the config, or the default value.

    Raises:
      ValueError if there is no such duration in the config and no default is set.
    """
    # Be helpful and make sure only match one of the following suffixes.
    suffixes = {"steps", "examples", "epochs", "percent"}
    matches = {f"{prefix}_{s}" for s in suffixes if f"{prefix}_{s}" in config}
    # Note that steps=0 is also a valid value (e.g. to only run evaluators).
    assert len(matches) <= 1, f"Only one of '{matches}' should be defined."

    if f"{prefix}_steps" in config:
        return config[f"{prefix}_steps"]

    if batch_size and f"{prefix}_examples" in config:
        return max(round(config[f"{prefix}_examples"] / batch_size), 1)

    if batch_size and data_size and f"{prefix}_epochs" in config:
        steps_per_epoch = data_size / batch_size
        return max(round(config[f"{prefix}_epochs"] * steps_per_epoch), 1)

    if total_steps and f"{prefix}_percent" in config:
        pct = config[f"{prefix}_percent"]
        assert 0.0 <= pct <= 1.0, (  # Be helpful, since it's not obvious.
            f"Percents should lie in [0.0, 1.0], but {prefix}_percent is {pct}")
        return max(round(pct * total_steps), 1)

    if default is ValueError:
        raise ValueError(
            f"Cannot convert {prefix} to steps, due to missing batch_size "
            f"({batch_size}), data_size ({data_size}), total_steps ({total_steps})"
            ", or corresponding entry in config:\n" + "\n".join(config.keys()))

    return default


def create_learning_rate_schedule(
        total_steps, batch_size=None, data_size=None,
        base=1.0, decay_type="stair",
        scale_with_batchsize=False, **kw):
    """Creates learning rate schedule, see (internal link).

    Args:
      total_steps: The total number of steps to run.
      batch_size: The global batch-size optionally used for scaling.
      data_size: Number of examples in the training data (for epoch conversion).
      base: The starting learning-rate (without warmup).
      decay_type: 'linear' or 'cosine', 'rsqrt', 'stair'.
      scale_with_batchsize: Whether or not to scale lr automatically.
      **kw: extra arguments specific to individual decay_types. Also contains
        declaration of `{warmup,cooldown}_{steps,epochs,examples}` that applies
        on top of any/all decay_type.

    Returns:
      A function learning_rate(step): float -> {"learning_rate": float}.
    """

    warmup_steps = steps(
        "warmup", kw, data_size, batch_size, total_steps, default=0)
    cooldown_steps = steps(
        "cooldown", kw, data_size, batch_size, total_steps, default=0)

    # Early catch hard to backtrack errors due to warmup_steps >= total_steps,
    # but let it run for 0 and 1 steps used to eval and debug runs.
    assert (total_steps <= 1) or (warmup_steps < total_steps), (
        "warmup_steps is >= total_steps")

    def step_fn(step):
        """Step to learning rate function."""
        lr = base

        # This implements the linear scaling rule following
        # Goyal et al. at arxiv.org/abs/1706.02677.
        # The reference batch size in literature is 256, so we scale the lr to
        # adjust to the literature lr when bach_size changes.
        if scale_with_batchsize:
            lr = lr * batch_size / 256.0

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        if decay_type in ("linear", "polynomial"):
            power = kw.get("power", 1)
            zero = kw.get("end", kw.get("linear_end", 0))
            lr = zero + (lr - zero) * (1.0 - progress) ** power
        elif decay_type == "cosine":
            if kw.get('min_lr'):
                min_lr_ratio = kw.get('min_lr') / kw.get('max_lr')
                lr = min_lr_ratio + (lr - min_lr_ratio) * \
                    0.5 * (1. + jnp.cos(jnp.pi * progress))
            else:
                lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
        elif decay_type == "rsqrt":
            timescale = kw.get("timescale", 10_000)
            shift = timescale - warmup_steps
            lr = jnp.where(warmup_steps < step, lr /
                           jnp.sqrt((step + shift) / timescale), lr)
        elif decay_type == "stair":
            i = jnp.searchsorted(jnp.array(kw.get("steps", [])), step + 1)
            lr = lr * jnp.take(jnp.array([1.0] + list(kw.get("mults", []))), i)
        else:
            raise ValueError(f"Unknown lr type {decay_type}")

        if warmup_steps:
            lr = lr * jnp.minimum(1., step / warmup_steps)
        if cooldown_steps:
            lr = lr * jnp.minimum(1., (total_steps - step) / cooldown_steps)

        return jnp.asarray(lr, dtype=jnp.float32)

    return step_fn


def find_states(opt_state, cls):
    leaves = jax.tree_util.tree_leaves(
        opt_state, is_leaf=lambda node: isinstance(node, cls))
    return [leaf for leaf in leaves if isinstance(leaf, cls)]


def get_count(opt_state):
    """Returns `ScaleByScheduleState.count` from `opt_state` as an integer."""
    counts = {
        int(state.count)
        for state in find_states(opt_state, optax.ScaleByScheduleState)
    }
    assert len(
        counts) == 1, f"Expected exactly 1 ScaleByScheduleState: {counts}"
    return next(iter(counts))


def replace_frozen(schedule, pytree, replacement, log=None):
    """Replaces values matching frozen params in `pytree` with `replacement`."""
    if not isinstance(schedule, (list, tuple)):
        return pytree
    masks, scheds = _make_mask_trees(pytree, schedule, log=log)
    frozen_mask, _, _ = _split_frozen(masks, scheds)
    return jax.tree_map(
        lambda v, f: replacement if f else v, pytree, frozen_mask)


def make(config, params, *, sched_kw):
    """Returns gradient transform and learning rate functions."""

    # Global schedule. No schedule means frozen.
    schedule = config.schedule
    if not isinstance(schedule, (tuple, list)):
        schedule = [(".*", schedule)]
    masks, scheds = _make_mask_trees(params, schedule, "config.schedule")
    frozen_mask, masks, scheds = _split_frozen(masks, scheds)
    not_frozen_mask = jax.tree_map(operator.not_, frozen_mask)

    def create_schedule(mult=1.0, **kw):
        assert "base" not in kw, kw
        return create_learning_rate_schedule(base=mult, **kw)
    schedule_fns = [create_schedule(**sched_kw, **sched) for sched in scheds]
    schedule_txs = [
        optax.masked(optax.scale_by_schedule(schedule_fn), mask)
        for schedule_fn, mask in zip(schedule_fns, masks)
    ] + [
        # Removes weight decay updates. Note that weight decay already has an
        # independent mask (which cannot be combined easily with a second mask),
        # so instead we multiply updates for frozen params with zero.
        optax.masked(optax.set_to_zero(), frozen_mask)
    ]

    # Gradient clipping.
    grad_clip_norm_tx = (
        optax.masked(optax.clip_by_global_norm(config.grad_clip_norm),
                     not_frozen_mask)
        if config.get("grad_clip_norm") else optax.identity())

    # Optimizer updates.
    tx_func = operator.attrgetter(config.optax_name)(optax)
    opt_txs = [optax.masked(
        tx_func(**config.get("optax", {})), not_frozen_mask)]
    assert "optimizer" not in config, "Deprecated option, use config.optax."

    # Learning rate multipliers. Defaults to 1.0.
    lr_mult_txs = [optax.scale(config.lr)]
    if config.get("lr_mults"):
        masks, mults = _make_mask_trees(
            params, config.lr_mults, "config.lr_mults")
        assert all(mult > 0 for mult in mults), (
            f"Use schedule=None for parameter freezing instead of lr_mults={mults}")
        lr_mult_txs += [
            optax.masked(optax.scale(mult), mask)
            for mult, mask in zip(mults, masks)
        ]

    if config.get('lwd'):
        from models.vit import decode_variant
        num_layer = decode_variant(config.model.image.variant)['depth']
        lwd_mults = [("img/.*encoderblock_" + str(i) + "/.*",
                      config.lwd**(num_layer - i)) for i in range(num_layer)]
        lwd_mults.append(("head.*", 1.0))
        lwd_mults.append(("encoder_norm.*", 1.0))
        lwd_mults.append(("embedding.*", config.lwd**(num_layer + 1)))
        lwd_mults.append(("pos_embedding.*", config.lwd ** (num_layer + 1)))
        lwd_mults.append(("cls.*", config.lwd ** (num_layer + 1)))

        masks, mults = _make_mask_trees(params, lwd_mults, "config.lwd_mults")
        lr_mult_txs += [
            optax.masked(optax.scale(mult), mask)
            for mult, mask in zip(mults, masks)
        ]

    # Weight decay. Defaults to 0.0.
    # Weight decay is not gradient-based but insted uses "params side-input".
    # Hence, weight decay is additive and independent of previous gradient-based
    # updates.
    assert "weight_decay" not in config, "Deprecated option. Use wd and schedule."
    assert config.get("weight_decay_decouple", True), (
        "Coupled weight decay not supported anymore.")
    if config.get("wd"):
        wd_mults = config.get("wd_mults", [(".*/kernel$", 1.0)])
        masks, mults = _make_mask_trees(params, wd_mults, "config.wd_mults")
        weight_decay_txs = [
            optax.add_decayed_weights(config.wd * mult, mask)
            for mult, mask in zip(mults, masks)
        ]
    else:
        weight_decay_txs = []

    # Combine gradient updates and learning rate schedules.
    return optax.chain(
        grad_clip_norm_tx,
        *opt_txs,
        *weight_decay_txs,  # exchange the order of learning rate to match pytorch implementation
        *lr_mult_txs,
        *schedule_txs,
        optax.scale(-1.0)), schedule_fns


def _make_mask_trees(params, patterns_values, log):
    patterns, values = zip(*patterns_values)
    masks = u.make_mask_trees(params, patterns, log=log)
    return masks, values


def _split_frozen(masks, scheds):
    """Computes `frozen_mask` and updates `masks` and `scheds`."""
    # Specifying `None` as a scheduler freezes params.
    all_false = jax.tree_map(lambda *bools: not any(bools), *masks)
    assert not any(jax.tree_flatten(all_false)[0]), (
        f"All params must be covered (use `None` for freezing): {all_false}")
    frozen_masks = [
        mask for mask, sched in zip(masks, scheds) if sched is None]
    frozen_mask = jax.tree_map(
        lambda *bools: any(bools), *frozen_masks,
        all_false)  # `all_false` is required when `frozen_masks==[]`.
    masks, scheds = zip(*(
        (mask, sched) for mask, sched in zip(masks, scheds) if sched is not None))
    return frozen_mask, masks, scheds

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


"""Utils very specific to this project, not generic."""

import collections
import contextlib
import dataclasses
import io
import json
import multiprocessing
import multiprocessing.pool
import os
import re
import time
from typing import Mapping

from absl import logging
from helpers import registry as pp_registry
import einops
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections as mlc
import numpy as np

import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import

Registry = pp_registry.Registry


# pylint: disable=logging-fstring-interpolation


def pad_shard_unpad(wrapped, static_argnums=(0,), static_argnames=()):
    """Wraps a function with code that pads, shards, then un-shards, un-pads.

    Args:
      wrapped: the function to be wrapped. Signature is `params, *args, *kwargs`.
      static_argnums: indices of arguments to `wrapped` that should _not_ be
        padded and sharded, but instead be forwarded as-is. The default is (0,)
        because by far the most common use-case is to pass `params` first.
      static_argnames: names of kwargs to `wrapped` that should _not_ be padded
        and sharded, but instead be forwarded as-is.

    Returns:
      A new function that pads and shards its arguments before passing them to
      the wrapped function, and un-shards and un-pads the returned pytree.

      This is useful for calling a pmap'ed function with inputs that aren't
      divisible by the number of devices. A typical use is:
        @pad_shard_unpad
        @jax.pmap
        def forward(params, x): ...

    Notes:
      The padding is done in host-memory before being passed to the function, and
      the values returned by the function are transferred back to host memory.

      The returned function is augmented with a new keyword-only argument
      `min_device_batch` that, if specified, forces padding inputs to at least
      this size per device. This can be useful to avoid recompiles for the last
      batch and reduce memory fragmentation.
    """

    def pad_shard_unpad_wrapper(*args, min_device_batch=None, **kw):
        d = jax.local_device_count()  # d = devices, b = batch
        batch_sizes = (
            {a.shape[0] for i, a in enumerate(args) if i not in static_argnums} |
            {v.shape[0] for k, v in kw.items() if k not in static_argnames})
        assert len(
            batch_sizes) == 1, f"Inconsistent batch-sizes: {batch_sizes}"
        b = batch_sizes.pop()

        def maybe_pad(x, actually_pad=True):
            if not actually_pad:
                return x  # For call-site convenience below.
            _, *shape = x.shape
            db, rest = divmod(b, d)
            if rest:
                x = np.concatenate(
                    [x, np.zeros((d - rest, *shape), x.dtype)], axis=0)
                db += 1
            if min_device_batch and db < min_device_batch:
                x = np.concatenate(
                    [x, np.zeros((d * (min_device_batch - db), *shape), x.dtype)])
                db = min_device_batch
            return x.reshape(d, db, *shape)

        args = [maybe_pad(a, i not in static_argnums)
                for i, a in enumerate(args)]
        kw = {k: maybe_pad(v, k not in static_argnames) for k, v in kw.items()}
        out = wrapped(*args, **kw)

        def unpad(x):
            # Transfer back before cutting, to reduce on-device shape
            # diversity.
            return einops.rearrange(jax.device_get(
                x), "d b ... -> (d b) ...")[:b]
        return jax.tree_util.tree_map(unpad, out)

    return pad_shard_unpad_wrapper


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    x = jax.lax.select(x, jnp.full(x.shape, on_value),
                       jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def npload(fname):
    """Loads `fname` and returns an np.ndarray or dict thereof."""
    # Load the data; use local paths directly if possible:
    if os.path.exists(fname):
        loaded = np.load(fname, allow_pickle=False)
    else:
        # For other (remote) paths go via gfile+BytesIO as np.load requires
        # seeks.
        with gfile.GFile(fname, "rb") as f:
            data = f.read()
        loaded = np.load(io.BytesIO(data), allow_pickle=False)

    # Support loading both single-array files (np.save) and zips (np.savez).
    if isinstance(loaded, np.ndarray):
        return loaded
    else:
        return dict(loaded)


def load_checkpoint(tree, npz):
    """Loads a jax pytree from a npz file.

    Args:
      tree: deprecated, use None.
        Bwd-compat for old format that only stored values: the pytree structure.
      npz: Either path to the checkpoint file (.npz), or a dict-like.

    Returns:
      A pytree that is the checkpoint.
    """
    if isinstance(npz, str):  # If not already loaded, then load.
        npz = npload(npz)
    keys, values = zip(*list(npz.items()))
    if tree:
        checkpoint = tree.unflatten(values)
    else:
        checkpoint = recover_tree(keys, values)
    return checkpoint


def load_params(tree, npz):
    """Loads a parameters from a npz checkpoint.

    Args:
      tree: deprecated, use None.
        Bwd-compat for old format that only stored values: the pytree structure.
      npz: Either path to the checkpoint file (.npz), or a dict-like.

    Returns:
      A pytree that is the checkpoint.

    Notes:
      The filename can contain an indicator like `/path/to/file.npz:keyname`, in
      which case ["opt"]["params"]["keyname"] will become ["opt"]["params"] in
      the returned checkpoint. This allows ANY model that uses this function to
      load itself from a checkpoint that contains multiple sub-models, such as
      checkpoints generated from image_text or Distillation trainers.
    """
    key = None  # Whether we want to extract only a sub-key of the model.
    if isinstance(npz, str):
        if ((":" in npz and "://" not in npz) or  # Like /path/to/file:subtree_name
                ("://" in npz and npz.count(":") == 2)):  # Like gs://path/to/file:sub
            npz, key = npz.rsplit(":", 1)
    checkpoint = load_checkpoint(tree, npz)
    if "params" in checkpoint:
        # Checkpoint with optax state (after cl/423007216).
        params = checkpoint["params"]
    elif "opt" in checkpoint:
        # Checkpoint with Flax optimizer.
        params = checkpoint["opt"]["target"]
    else:
        # When open-sourcing, we usually shared only the params directly.
        params = checkpoint
    if key is not None:
        params = tree_get(params, key)
    return params


def prefetch_scalar(it, nprefetch=1, devices=None):
    n_loc_dev = len(devices) if devices else jax.local_device_count()
    repl_iter = (np.ones(n_loc_dev) * i for i in it)
    return flax_utils.prefetch_to_device(repl_iter, nprefetch, devices)


def itstime(step, every_n_steps, total_steps, host=None, last=True, first=True,
            drop_close_to_last=0.25):
    """Returns True if it's time to execute an action.

    Args:
      step: the current step representing "now".
      every_n_steps: the action should run every this many steps.
      total_steps: the step number of the last step of training.
      host: host number. If provided, only run if we are this process.
      last: whether to run on the last step or not.
      first: whether to run on the first step or not.
      drop_close_to_last: if a step would run, but is this close (in terms of
        fraction of every_n_step) to the last one, skip.

    Returns:
      True if the action should be executed, False if not.
    """

    # This logic avoids running `itstime` "a few" steps before the last step.
    # Canonical example: don't save checkpoint 2 steps before the last, and then
    # at the last again; it's pointless and checkpoint timing will time out.
    close_to_last = False
    if drop_close_to_last and every_n_steps:
        close_to_last = abs(
            step - total_steps) < drop_close_to_last * every_n_steps

    is_host = host is None or jax.process_index() == host
    is_step = every_n_steps and (step %
                                 every_n_steps == 0) and not close_to_last
    is_last = every_n_steps and step == total_steps
    is_first = every_n_steps and step == 1
    return is_host and (is_step or (last and is_last) or (first and is_first))


def checkpointing_timeout(writer, timeout):
    # Make sure checkpoint writing is not a bottleneck
    if writer is not None:
        try:
            writer.get(timeout=timeout)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError(
                "Checkpoint writing seems to be a bottleneck. Make sure you do "
                "not do something wrong, like writing checkpoints to a distant "
                "cell. In a case you are OK with checkpoint writing being a "
                "bottleneck, you can configure `ckpt_timeout` parameter") from e


def hms(s):
    """Format time in hours/minutes/seconds."""
    if s < 60:
        return f"{s:.0f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m:.0f}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{h:.0f}h{m:.0f}m"  # Seconds intentionally omitted.


class Chrono:
    """Measures time and reports progress, hyper-specific to our train loops.

    Some concepts:
    1. This differentiates between three "types" of time:
      - training time: the time spent on actual training (fprop/bprop/update)
      - program time: overall time the program runs, including all overheads
      - pause time: the chronometer can be paused (eg during evals).
    2. This handles a "warmup": the first step is skipped for training time
        purposes, as it includes significant compilation overheads, which distort
        estimates.
    3. `accum`ulates (i.e. integrates) timings, and save/load them across
        restarts.
    """

    def __init__(self):
        self._timing_history = collections.defaultdict(list)
        self._measure = None
        self._write_note = None

        self.program_start_time = time.monotonic()
        self.train_start_time = None
        self.train_start_step = None  # When we started timing (after warmup)

        self.prev_time = None
        self.prev_step = None

        self.pause_start = None
        self.paused_time = 0

        self.total_steps = None
        self.global_bs = None
        self.steps_per_epoch = None

        self.warmup = 2  # How many calls to `tick` to skip.
        self.load()  # Inits accum integrators.
        self.note = "Chrono n/a"

    def inform(self, *, first_step=None, total_steps=None, global_bs=None,
               steps_per_epoch=None, measure=None, write_note=None):
        """Provide some extra info that's only known later in the program."""
        # The pattern of `self.x = x or self.x` allows one to call `inform` various
        # times with various subset of information (args), as they become available.
        # Except for `first_step` which can be 0 so is a bit more verbose.
        self.prev_step = first_step if first_step is not None else self.prev_step
        self.total_steps = total_steps or self.total_steps
        self.steps_per_epoch = steps_per_epoch or self.steps_per_epoch
        self.global_bs = global_bs or self.global_bs
        self._measure = measure or self._measure
        self._write_note = write_note or self._write_note
        if self.total_steps and self.prev_step is not None:
            self.note = (f"Steps:{self.prev_step}/{self.total_steps} "
                         f"[{self.prev_step/self.total_steps:.1%}]")

    def tick(self, step, measure=None, write_note=None):
        """A chronometer tick."""
        if step == self.prev_step:
            return  # Can happen from evals for example.

        measure = measure or self._measure
        write_note = write_note or self._write_note

        now = time.monotonic()
        measure("uptime", now - self.program_start_time)
        self.flush_timings()

        # We do always count examples, regardless of the timing-related warmup that
        # happens a few lines below.
        ds = step - self.prev_step  # Steps between ticks
        self.prev_step = step
        self.accum_examples_seen += ds * self.global_bs
        measure("examples_seen", self.accum_examples_seen)
        measure("progress", step / self.total_steps)
        if self.steps_per_epoch:
            measure("epoch", step / self.steps_per_epoch)

        # We take the start as the second time `tick` is called, so we avoid
        # measuring the overhead of compilation and don't include it in time
        # estimates.
        if self.warmup > 1:
            self.warmup -= 1
            write_note(self.note)  # This can help debugging.
            return
        if self.warmup == 1:
            self.train_start_time = self.prev_time = now
            self.train_start_step = step
            self.accum_program_time += now - self.program_start_time
            # Drop pauses that happened before timing starts.
            self.paused_time = 0
            self.warmup = 0
            write_note(self.note)  # This can help debugging.
            return

        # Measurement with micro-timings of current training steps speed.
        # Time between ticks (ignoring pause)
        dt = now - self.prev_time - self.paused_time
        ncores = jax.device_count()  # Global device count
        measure("img/sec/core", self.global_bs * ds / dt / ncores)

        # Accumulate (integrate) times, good for plots.
        self.accum_train_time += dt
        self.accum_pause_time += self.paused_time
        self.accum_program_time += dt + self.paused_time

        # Convert to, and log as, core hours.
        core_hours = self.accum_train_time * ncores / 60 / 60
        devtype = jax.devices()[0].device_kind
        measure(f"core_hours_{devtype}", core_hours)
        # For convenience as x-axis in sweeps.
        measure("core_hours", core_hours)

        # Progress note with "global" full-program average timings
        # (eg in program-time minus warmup)
        dt = now - self.train_start_time  # Time elapsed since end of warmup.
        steps_timed = step - self.train_start_step
        steps_todo = self.total_steps - step
        self.note = f"Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]"
        self.note += f"\nWalltime:{hms(self.accum_program_time)}"
        self.note += f" ({hms(self.accum_pause_time)} eval)"
        self.note += f"\nETA:{hms(dt / steps_timed * steps_todo)}"
        self.note += f"\nTotal train time:{hms(dt / steps_timed * self.total_steps)}"
        write_note(self.note)

        self.prev_time = now
        self.paused_time = 0

    def pause(self, wait_for=()):
        assert self.pause_start is None, "Don't pause twice."
        jax.block_until_ready(wait_for)
        self.pause_start = time.monotonic()

    def resume(self):
        self.paused_time += time.monotonic() - self.pause_start
        self.pause_start = None

    def save(self):
        return dict(
            accum_program_time=self.accum_program_time,
            accum_train_time=self.accum_train_time,
            accum_pause_time=self.accum_pause_time,
            accum_examples_seen=self.accum_examples_seen,
        )

    def load(self, ckpt={}):  # pylint: disable=dangerous-default-value
        self.accum_program_time = ckpt.get("accum_program_time", 0.0)
        self.accum_train_time = ckpt.get("accum_train_time", 0.0)
        self.accum_pause_time = ckpt.get("accum_pause_time", 0.0)
        self.accum_examples_seen = ckpt.get("accum_examples_seen", 0)

    @contextlib.contextmanager
    def log_timing(self, name, *, noop=False):
        """Use this when you time sth once per step and want instant flushing."""
        t0 = time.monotonic()
        yield
        dt = time.monotonic() - t0
        if not noop:
            self._measure(name, dt)
            logging.info("TIMING[%s]: %s", name, dt)
            logging.flush()

    @contextlib.contextmanager
    def log_timing_avg(self, name, *, noop=False):
        """Use this when you time sth multiple times per step (eg in a loop)."""
        t0 = time.monotonic()
        yield
        dt = time.monotonic() - t0
        if not noop:
            self._timing_history[name].append(dt)
            logging.info("TIMING[%s]: avg %s current %s",
                         name, np.mean(self._timing_history[name]), dt)
            logging.flush()

    def flush_timings(self):
        for name, times in self._timing_history.items():
            self._measure(name, np.mean(times))
        self._timing_history.clear()


# Singleton to use from everywhere. https://stackoverflow.com/a/6760726/2366315
chrono = Chrono()


def _traverse_with_names(tree, with_inner_nodes=False):
    """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
    if dataclasses.is_dataclass(tree):
        tree = flax.serialization.to_state_dict(tree)
    # Don't output the non-leaf nodes. If the optimizer doesn't have a state
    # the tree leaves can be Nones which was interpreted as a leaf by this
    # function but not by the other functions (like jax.tree_util.tree_map).
    if tree is None:
        return
    elif isinstance(tree, Mapping):
        keys = sorted(tree.keys())
        for key in keys:
            for path, v in _traverse_with_names(tree[key], with_inner_nodes):
                yield (key + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    elif isinstance(tree, (list, tuple)):
        for idx in range(len(tree)):
            for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
                yield (str(idx) + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    else:
        yield "", tree


def tree_flatten_with_names(tree):
    """Populates tree_flatten with leaf names.

    This function populates output of tree_flatten with leaf names, using a
    custom traversal that produces names is provided. The custom traversal does
    NOT have to traverse tree in the same order as jax, as we take care of
    automatically aligning jax' and custom traversals.

    Args:
      tree: python tree.

    Returns:
      A list of values with names: [(name, value), ...]
    """
    vals, tree_def = jax.tree_flatten(tree)

    # "Fake" token tree that is use to track jax internal tree traversal and
    # adjust our custom tree traversal to be compatible with it.
    tokens = range(len(vals))
    token_tree = tree_def.unflatten(tokens)
    val_names, perm = zip(*_traverse_with_names(token_tree))
    inv_perm = np.argsort(perm)

    # Custom traverasal should visit the same number of leaves.
    assert len(val_names) == len(vals)

    return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_unflatten(names_and_vals):
    """Reverses `tree_flatten_with_names(tree)[0]`."""
    return recover_tree(*zip(*names_and_vals))


def tree_map_with_names(f, tree, *rest):
    """Like jax.tree_util.tree_map but with a filter on the leaf path name.

    Args:
      f: A function with first parameter `name` (path-like "a/b/c") and remaining
        parameters values of `tree` and `*rest` corresponding to the given `name`
        Should return a new value for parameter `name`.
      tree: The tree of parameters `f` should be applied to.
      *rest: more trees of the exact same structure.

    Returns:
      A tree identical in structure to `tree` and `*rest` but with the leaves the
      result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
    """
    names_and_vals, tree_def = tree_flatten_with_names(tree)
    names, vals = zip(*names_and_vals)
    rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
    vals = [f(*name_and_vals)
            for name_and_vals in zip(names, vals, *rest_vals)]
    return tree_def.unflatten(vals)


def tree_map_with_regex(f, tree, regex_rules, not_f=lambda x: x, name=None):
    """Apply jax-style tree_map based on regex rules.

    Args:
      f: a function that is being applied to every variable.
      tree: jax tree of arrays.
      regex_rules: a list of tuples `(pattern, args)`, where `pattern` is a regex
        which used for variable matching and `args` are positional arguments
        passed to `f`. If some variable is not matched, we apply `not_f` transform
        which is id by default. If multiple patterns match, then only the first
        rule is applied.
      not_f: optional function which is applied to variables that do not match any
        pattern.
      name: a name of transform for logging purposes.

    Returns:
      a tree, transformed by `f` according to the given rules.
    """
    def _f(vname, v):
        for pattern, arg in regex_rules:
            if re.fullmatch(pattern, vname):
                if name and jax.process_index() == 0:
                    logging.info("Applying %s to %s with %s due to `%s`",
                                 name, vname, arg, pattern)
                return f(v, arg)
        return not_f(v)
    return tree_map_with_names(_f, tree)


def tree_get(tree, name):
    """Get an entry of pytree by flattened key name, eg a/b/c, with nice error.

    Args:
      tree: the pytree to be queried.
      name: the path to extract from the tree, see below for examples.

    Returns:
      A few examples:
        tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
        tree_get(tree, 'a') == 1
        tree_get(tree, 'b/c') == 2
        tree_get(tree, 'b') == {'c': 2, 'd': 3}
    """
    flattened = dict(_traverse_with_names(tree, with_inner_nodes=True))
    try:
        return flattened[name]
    except KeyError as e:
        class Msg(str):  # Reason: https://stackoverflow.com/a/70114007/2366315
            def __repr__(self):
                return str(self)
        msg = "\n".join([name, "Available keys:", *flattened, ""])
        # Turn into configdict to use its "did you mean?" error message!
        msg = mlc.ConfigDict(flattened)._generate_did_you_mean_message(
            name, msg)  # pylint: disable=protected-access
        raise KeyError(Msg(msg)) from e


def tree_replace(tree, replacements):
    """Renames/removes (nested) keys.

    Example usage:

      tree = {'a': {'b': 2, 'c': 3}, 'c': 4}
      replacements = {
          'a/b': 'a/b/x',  # replaces 'a/b' with 'a/b/x'
          '.*c': 'C',      # replaces 'c' with 'C' ('a/c' is removed)
          'C': 'D',        # replaces 'C' (which was 'c') with 'D'
          '.*/c': None,    # removes 'a/c'
      }
      tree2 = rename_remove(tree, replacements)
      assert tree2 == {'D': 4, 'a': {'b': {'x': 2}}}

    Args:
      tree: A nested dictionary.
      replacements: Rules specifying `regex` as keys and `replacement` as values
        to be used with `m = re.match(regex, key)` and `m.expand(replacement)`
        for every `key` independently.

        Note that:
        1. If any rule matches with `replacement=None`, then the key is removed.
        2. The rules are applied in order. It's possible to have multiple
           transformations on a single key.

    Returns:
      Updated `tree` according to rules defined in `replacements`.
    """
    replacements = {
        re.compile(kk): vv for kk, vv in replacements.items()
    }

    def rename(k):
        for kk, vv in replacements.items():
            m = kk.match(k)
            if m:
                k = k[:m.start()] + m.expand(vv) + k[m.end():]
        return k

    def should_remove(k):
        return any(vv is None and kk.match(k)
                   for kk, vv in replacements.items())

    names_and_vals, _ = tree_flatten_with_names(tree)
    names_and_vals = [
        (rename(k), v) for k, v in names_and_vals if not should_remove(k)
    ]
    return tree_unflatten(names_and_vals)


def tree_compare(tree1, tree2):
    """Returns `(tree1_only, tree2_only, dtype_shape_mismatch)`."""
    tree1 = flax.traverse_util.flatten_dict(tree1, sep="/")
    tree2 = flax.traverse_util.flatten_dict(tree2, sep="/")
    return set(tree1) - set(tree2), set(tree2) - set(tree1), {
        k: [(v.dtype, v.shape), (tree2[k].dtype, tree2[k].shape)]
        for k, v in tree1.items()
        if k in tree2 and (v.dtype != tree2[k].dtype or v.shape != tree2[k].shape)
    }


def recover_dtype(a):
    """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
    if hasattr(a, "dtype") and a.dtype.type is np.void:
        assert a.itemsize == 2, "Unknown dtype!"
        return a.view(jax.numpy.bfloat16)
    else:
        return a


# Checkpoint names encode tree structure, you can check out this colab for an
# example of how to recover tree structure from names:
# (internal link)
def save_checkpoint(checkpoint, path, step_copy=None, compressed=False):
    """Util for checkpointing: saves jax pytree objects to the disk.

    Args:
      checkpoint: arbitrary jax pytree to be saved.
      path: a path to save the checkpoint.
      step_copy: creates a copy of the checkpoint with `path-{step_copy}` name.
      compressed: whether to use np.savez or np.savez_compressed, useful if saving
        large buffers that are easily compressed (e.g. repeated or integers).
    """
    names_and_vals, _ = tree_flatten_with_names(checkpoint)
    io_buffer = io.BytesIO()

    if compressed:
        np.savez_compressed(io_buffer, **{k: v for k, v in names_and_vals})
    else:
        np.savez(io_buffer, **{k: v for k, v in names_and_vals})

    # In order to be robust to interruptions we first save checkpoint to the
    # temporal file and then move to actual path name.
    path_tmp = path + "-TEMPORARY"
    with gfile.GFile(path_tmp, "wb") as f:
        f.write(io_buffer.getvalue())
    gfile.rename(path_tmp, path, overwrite=True)

    if step_copy is not None:
        gfile.copy(path, f"{path}-{step_copy:09d}", overwrite=True)


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are saved by our programs
    without need to access the exact source code of the experiment. In particular,
    it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
    subtree of parameters.

    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.

    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def _sync(x):
    return jax.lax.psum(x, "i")


def sync():
    """Syncs hosts and empties async computation queue."""
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(_sync, "i")(x))
    assert x[0] == jax.device_count()


def check_and_compile_patterns(patterns):
    """Validates and compiles a list of param-patterns.

    The validation consists of checking for common mistakes, currently only that
    the pattern does not start with a slash, because unlike FLAX, our parameter
    names don't start with a slash.

    Args:
      patterns: a single (string) pattern (regex), or a list of patterns.

    Returns:
      A list of compiled and verified regexes.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    assert isinstance(patterns, (list, tuple)), patterns

    def check_and_compile(pattern):
        assert not pattern.startswith("/"), (
            f"Big vision parameter names never start with '/': '{pattern}")
        return re.compile(pattern)

    return list(map(check_and_compile, patterns))


def make_mask_trees(tree, patterns, *, log=None):
    """Returns a boolean mask tree for every pattern (only first match)."""
    compiled_patterns = check_and_compile_patterns(patterns)

    def matchfirst(name, _):
        matches = []
        for pattern in compiled_patterns:
            matches.append(not any(matches) and bool(pattern.fullmatch(name)))
        if log is not None and True in matches and jax.process_index() == 0:
            logging.info("%s: %s - matched by %s", log, name,
                         patterns[matches.index(True)])
        return np.array(matches)

    multimask = tree_map_with_names(matchfirst, tree)
    return [
        jax.tree_util.tree_map(lambda matches, i=idx: matches[i], multimask)
        for idx in range(len(patterns))
    ]


@contextlib.contextmanager
def profile(name, ttl=3 * 365 * 24 * 3600, noop=False):
    if not noop:
        sess = startstop_prof_at_steps(None, name=name, ttl=ttl)
    yield
    if not noop:
        startstop_prof_at_steps(sess, name=name, ttl=ttl)


def startstop_prof(sess, step=None, first_step=0,
                   log_steps=1, surround=20, **kw):
    """Runs the profiler for `surround` steps around the next `log_steps`."""
    first_log = first_step + log_steps - (first_step % log_steps)
    # don't start before first!
    start = max(first_log - surround // 2, first_step + 1)
    return startstop_prof_at_steps(sess, step, start, start + surround, **kw)


def startstop_prof_at_steps(
        sess, step=None, first_step=None, last_step=None,
        name="steps", ttl=3 * 365 * 24 * 3600):
    del sess, step, first_step, last_step, name, ttl
    pass  # TODO: implement using `jax.profiler` API. Needs workdir.


# This is a very minimal variant for open-sourcing. Our internal code makes use
# of multiple internal logging tools instead.
class BigVisionMetricWriter:
    """A class for logging metrics."""

    def __init__(self, xid=-1, wid=-1, workdir=None, config=None):
        self.step_start(0)
        if jax.process_index() != 0:
            return  # Only one host shall write stuff.

        self.pool = multiprocessing.pool.ThreadPool(1)  # 1 is important here.
        self.fname = None
        if workdir:
            if xid != -1 and wid != -1:
                self.fname = os.path.join(
                    workdir, f"big_vision_{xid}_{wid}_metrics.txt")
            else:
                self.fname = os.path.join(workdir, "big_vision_metrics.txt")
            if config:
                with gfile.GFile(os.path.join(workdir, "config.json"), "w") as f:
                    f.write(config.to_json())

    def step_start(self, step):
        self.step = step
        self.step_metrics = {}

    def measure(self, name, value):
        """Logs the metric value."""
        if jax.process_index() != 0:
            return  # Only one host shall write stuff.

        # Convenience for accepting scalar np/DeviceArrays, as well as N-d single
        # scalars, like [[[123]]] or similar, avoiding silly mistakes.
        value = np.array(value).squeeze()

        # If the value is a scalar, we keep it in mind to append a line to the logs.
        # If it has any structure, we instead just log its shape.
        value = float(value) if value.ndim == 0 else value.shape

        logging.info(f"\u001b[35m[{self.step}]\u001b[0m {name} = {value}")
        logging.flush()
        self.step_metrics[name] = value

        return value  # Just for convenience

    def step_end(self):
        """Ends a training step, write its full row."""
        if not self.step_metrics:
            return

        def write(metrics):
            with gfile.GFile(self.fname, "a") as f:
                f.write(json.dumps({"step": self.step, **metrics}) + "\n")

        if self.fname:
            self.pool.apply(lambda: None)  # Potentially wait for past writes.
            self.pool.apply_async(write, (self.step_metrics,))

    def close(self):
        self.step_end()
        if jax.process_index() == 0:
            self.pool.close()
            self.pool.join()


def maybe_cleanup_workdir(workdir, cleanup, info):
    """Potentially removes workdirs at end of run for cleanup."""
    if not workdir:
        return

    if not cleanup:
        info("Logs/checkpoints are in %s", workdir)
    elif jax.process_index() == 0:
        gfile.rmtree(workdir)
        try:  # Only need this on the last work-unit, if already empty.
            gfile.remove(os.path.join(workdir, "../.."))
        except tf.errors.OpError:
            pass


mae_weights = {
    'pretrained': {
        'B/16': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        'L/16': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        'H/14': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth'},
    'finetuned': {
        'B/16': 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth',
        'L/16': 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth',
        'H/14': 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth'}
}


def load_mae_weights(jax_vit_paras, model_variant='B/16', pretrained=False):
    if pretrained:
        url = mae_weights['pretrained'][model_variant]
    else:
        url = mae_weights['finetuned'][model_variant]
    weight_name = url.split('/')[-1]
    from pathlib import Path
    import torch
    mae_eval_path = Path('../mae_checkpoints/')
    Path.mkdir(mae_eval_path, exist_ok=True)
    if jax.process_index() == 0:
        logging.info(f"Initialize model from MAE " + url)

    if os.path.isfile(os.path.join(mae_eval_path, weight_name)):
        checkpoint = torch.load(
            os.path.join(
                mae_eval_path,
                weight_name),
            map_location='cpu')
    else:
        import torch.utils.model_zoo
        checkpoint = torch.utils.model_zoo.load_url(
            url, model_dir=str(mae_eval_path))
    para_dict = checkpoint['model']

    def create_jax_tree(torch_state_dict, params_cpu):
        import copy
        params = copy.deepcopy(params_cpu)
        for k, v in torch_state_dict.items():
            if k == 'cls_token':
                params['cls'] = jnp.array(v.numpy())
            if k == 'pos_embed':
                params['pos_embedding'] = jnp.array(v.numpy())
            if k == 'head.weight':
                params['head']['kernel'] = jnp.array(v.numpy()).transpose(1, 0)
            if k == 'head.bias':
                params['head']['bias'] = jnp.array(v.numpy())
            if k == 'patch_embed.proj.weight':
                #shape = params['embedding']['kernel'].shape
                params['embedding']['kernel'] = jnp.array(
                    v.numpy()).transpose(
                    2, 3, 1, 0)  # .reshape(shape)
            if k == 'patch_embed.proj.bias':
                params['embedding']['bias'] = jnp.array(v.numpy())
            if k == 'fc_norm.weight':
                params['encoder_norm']['scale'] = jnp.array(v.numpy())
            if k == 'fc_norm.bias':
                params['encoder_norm']['bias'] = jnp.array(v.numpy())
            if 'blocks' in k:
                block_num = k.split('.')[1]
                new_block_name = 'encoderblock_' + block_num
                if 'norm1.weight' in k:
                    params['Transformer'][new_block_name]['LayerNorm_0']['scale'] = jnp.array(
                        v.numpy())
                if 'norm1.bias' in k:
                    params['Transformer'][new_block_name]['LayerNorm_0']['bias'] = jnp.array(
                        v.numpy())
                if 'norm2.weight' in k:
                    params['Transformer'][new_block_name]['LayerNorm_1']['scale'] = jnp.array(
                        v.numpy())
                if 'norm2.bias' in k:
                    params['Transformer'][new_block_name]['LayerNorm_1']['bias'] = jnp.array(
                        v.numpy())
                if 'mlp' in k:
                    fc_num = int(k.split('.')[3][2]) - 1
                    mlp_block_name = 'MlpBlock_0'
                    fc_name = 'Dense_' + str(fc_num)
                    if 'bias' in k:
                        params['Transformer'][new_block_name][mlp_block_name][fc_name]['bias'] = jnp.array(
                            v.numpy())
                    else:
                        #shape = params['Transformer'][new_block_name][mlp_block_name][fc_name]['kernel'].shape
                        params['Transformer'][new_block_name][mlp_block_name][fc_name]['kernel'] = jnp.array(
                            v.numpy()).transpose(1, 0)
                if 'attn' in k:
                    attn_name = 'MultiHeadDotProductAttention_0'
                    if 'proj' in k:
                        if 'bias' in k:
                            params['Transformer'][new_block_name][attn_name]['out']['bias'] = jnp.array(
                                v.numpy())
                        else:
                            shape = params['Transformer'][new_block_name][attn_name]['out']['kernel'].shape
                            params['Transformer'][new_block_name][attn_name]['out']['kernel'] = jnp.array(
                                v.numpy()).transpose(1, 0).reshape(shape)
                    else:
                        qkv_list = ['query', 'key', 'value']
                        if 'bias' in k:
                            # pass
                            q_bias, k_bias, v_bias = torch.chunk(v, 3, dim=0)
                            shape = params['Transformer'][new_block_name][attn_name]['query']['bias'].shape
                            params['Transformer'][new_block_name][attn_name][qkv_list[0]]['bias'] = jnp.array(
                                q_bias.numpy()).reshape(shape)
                            params['Transformer'][new_block_name][attn_name][qkv_list[1]]['bias'] = jnp.array(
                                k_bias.numpy()).reshape(shape)
                            params['Transformer'][new_block_name][attn_name][qkv_list[2]]['bias'] = jnp.array(
                                v_bias.numpy()).reshape(shape)
                        else:
                            q_w, k_w, v_w = torch.chunk(v, 3, dim=0)
                            shape = params['Transformer'][new_block_name][attn_name]['query']['kernel'].shape
                            params['Transformer'][new_block_name][attn_name][qkv_list[0]]['kernel'] = jnp.array(
                                q_w.numpy()).transpose(1, 0).reshape(shape)
                            params['Transformer'][new_block_name][attn_name][qkv_list[1]]['kernel'] = jnp.array(
                                k_w.numpy()).transpose(1, 0).reshape(shape)
                            params['Transformer'][new_block_name][attn_name][qkv_list[2]]['kernel'] = jnp.array(
                                v_w.numpy()).transpose(1, 0).reshape(shape)
            # params[k] = jnp.array(v.numpy())
        return jax.tree_util.tree_map(lambda x: x, params)

    params_mae = create_jax_tree(para_dict, jax_vit_paras)
    from models.common import merge_params
    params_mae = merge_params(params_mae, jax_vit_paras)
    params_out = jax.tree_map(recover_dtype, params_mae)
    return params_out


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


def load_yutong_ckpt(jax_vit_paras, model='large'):
    from pathlib import Path
    mae_eval_path = Path('/home/lixianhang/yutong_ckpt/' + model)
    Path.mkdir(mae_eval_path, exist_ok=True)
    if model == 'huge':
        weight_name = 'ckpt-499200'
        width = 1280
    else:
        weight_name = 'ckpt-499200 (1)'
        width = 1024
    if os.path.isfile(
        os.path.join(
            mae_eval_path,
            weight_name +
            '.data-00000-of-00001')):
        # reader = ckpt.read(os.path.join(mae_eval_path, weight_name))
        import tensorflow as tf
        reader = tf.compat.v1.train.NewCheckpointReader(
            os.path.join(mae_eval_path, weight_name))
        var_to_shape_map = reader.get_variable_to_shape_map()
        model_para = {}
        # Print tensor name and values
        for key in var_to_shape_map:
            if 'optimizer' not in key and 'clip' not in key and 'decoder' not in key:
                model_para[key] = reader.get_tensor(key)
                # print("tensor_name: ", key)
                # print(reader.get_tensor(key))
                if 'layers' not in key:
                    print(key)
                if '_encoder_layers/9' in key:
                    print(key)
    else:
        print('The checkpoint path does not exists')
        return
    import copy
    params = copy.deepcopy(jax_vit_paras)

    params['pos_embedding'] = posemb_sincos_2d(
        16, 16, width=width, cls_token=True)
    for k, v in model_para.items():
        if 'token_cls' in k:
            params['cls'] = jnp.array(v)

        if 'patch_to_embed' in k and 'kernel' in k:
            params['embedding']['kernel'] = jnp.array(v)

        if 'patch_to_embed' in k and 'bias' in k:
            params['embedding']['bias'] = jnp.array(v)

        if '_encoder_layers' in k:
            block_num = k.split('/')[3]
            new_block_name = 'encoderblock_' + block_num
            if '_attention_layer_norm' in k and 'gamma' in k:
                params['Transformer'][new_block_name]['LayerNorm_0']['scale'] = jnp.array(
                    v)
            if '_attention_layer_norm' in k and 'beta' in k:
                params['Transformer'][new_block_name]['LayerNorm_0']['bias'] = jnp.array(
                    v)
            if '_output_layer_norm' in k and 'gamma' in k:
                params['Transformer'][new_block_name]['LayerNorm_1']['scale'] = jnp.array(
                    v)
            if '_output_layer_norm' in k and 'beta' in k:
                params['Transformer'][new_block_name]['LayerNorm_1']['bias'] = jnp.array(
                    v)
            if 'dense' in k and '_attention_layer' not in k:
                mlp_block_name = 'MlpBlock_0'
                if '_intermediate_dense' in k:
                    fc_name = 'Dense_' + str(0)
                    if 'bias' in k:
                        params['Transformer'][new_block_name][mlp_block_name][fc_name]['bias'] = jnp.array(
                            v)
                    else:
                        # shape = params['Transformer'][new_block_name][mlp_block_name][fc_name]['kernel'].shape
                        params['Transformer'][new_block_name][mlp_block_name][fc_name]['kernel'] = jnp.array(
                            v)
                if '_output_dense' in k:
                    fc_name = 'Dense_' + str(1)
                    if 'bias' in k:
                        params['Transformer'][new_block_name][mlp_block_name][fc_name]['bias'] = jnp.array(
                            v)
                    else:
                        params['Transformer'][new_block_name][mlp_block_name][fc_name]['kernel'] = jnp.array(
                            v)
            if '_attention_layer' in k:
                attn_name = 'MultiHeadDotProductAttention_0'
                if '_output_dense' in k:
                    if 'bias' in k:
                        params['Transformer'][new_block_name][attn_name]['out']['bias'] = jnp.array(
                            v)
                    else:

                        params['Transformer'][new_block_name][attn_name]['out']['kernel'] = jnp.array(
                            v)
                if '_query_dense' in k:
                    if 'bias' in k:
                        params['Transformer'][new_block_name][attn_name]['query']['bias'] = jnp.array(
                            v)
                    else:
                        params['Transformer'][new_block_name][attn_name]['query']['kernel'] = jnp.array(
                            v)

                if '_key_dense' in k:
                    if 'bias' in k:
                        params['Transformer'][new_block_name][attn_name]['key']['bias'] = jnp.array(
                            v)
                    else:
                        params['Transformer'][new_block_name][attn_name]['key']['kernel'] = jnp.array(
                            v)

                if '_value_dense' in k:
                    if 'bias' in k:
                        params['Transformer'][new_block_name][attn_name]['value']['bias'] = jnp.array(
                            v)
                    else:
                        params['Transformer'][new_block_name][attn_name]['value']['kernel'] = jnp.array(
                            v)

    from models.common import merge_params
    params = merge_params(params, jax_vit_paras)
    params = jax.tree_map(recover_dtype, params)
    return params

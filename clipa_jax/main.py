# Copyright @2023 Xianhang Li

# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer
# vision and for their open-source contributions to this project.

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

"""Contrastive training loop.

For models Like
- LiT (https://arxiv.org/abs/2111.07991)
- CLIP (https://arxiv.org/abs/2103.00020)
"""
# pylint: disable=consider-using-from-import
from transforms.mixup import mixup, cutmix
from helpers.utils import *
from datasets.input_pipeline import shard_and_put
from datasets import input_pipeline
import optim
import losses
from tensorflow.io import gfile
from optim import replace_frozen, steps
from models.common import merge_params
import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
import evaluators.common as eval_common

from clu import parameter_overview
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# prepare config


try:
    import wandb
    has_wandb = True
   # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
except ImportError:
    has_wandb = False
    print('please install wandb')

# pylint: disable=logging-fstring-interpolation


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def all_gather(z):
    """All gather and flatten first two dims."""
    def gather_flat(x): return jnp.concatenate(
        jax.lax.all_gather(x, "batch"), 0)
    return jax.tree_map(gather_flat, z)


def main(argv):
    del argv
    tf.config.experimental.set_visible_devices([], "GPU")

    config = flags.FLAGS.config
    workdir = flags.FLAGS.workdir
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"\u001b[33mHello from process {jax.process_index()} holding "
        f"{jax.local_device_count()}/{jax.device_count()} devices and "
        f"writing to workdir {workdir}.\u001b[0m")

    save_ckpt_path = None
    if workdir:  # Always create if requested, even if we may not write into it.
        gfile.makedirs(workdir)
        save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

    # The pool is used to perform misc operations such as logging in async way.
    pool = multiprocessing.pool.ThreadPool()

    # Here we register preprocessing ops from modules listed on `pp_modules`.
    for m in config.get(
            "pp_modules", [
            "ops_general", "ops_image", "ops_text"]):
        importlib.import_module(f"transforms.{m}")

    # This seed makes the Jax part of things (like model init) deterministic.
    # However, full training still won't be deterministic, for example due to the
    # tf.data pipeline not being deterministic even if we would set TF seed.
    # See (internal link) for a fun read on what it takes.
    rng = jax.random.PRNGKey(config.get("seed", 0))

    # These functions do more stuff internally, for OSS release we mock them by
    # trivial alternatives in order to minize disruptions in the code.
    xid, wid = -1, -1

    def info(s, *a):
        logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    write_note("Initializing...")

    batch_size = config.input.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must "
            f"be divisible by device number ({jax.device_count()})")
    info(
        "Global batch size %d on %d hosts results in %d local batch size. With "
        "%d dev per host (%d dev total), that's a %d per-device batch size.",
        batch_size,
        jax.process_count(),
        batch_size // jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        batch_size // jax.device_count())

    if config.wandb.log_wandb:
        if has_wandb and jax.process_index() == 0:
            if config.wandb.wandb_offline:
                os.environ["WANDB_MODE"] = 'offline'
            else:
                wandb.init(
                    project=str(
                        config.wandb.project), name=str(
                        config.wandb.experiment), entity=str(
                        config.wandb.entity), resume=config.wandb.resume)
                wandb.config.update(dict(config))
        else:
            logging.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # First thing after above sanity checks, so we can log "start" ticks.
    metric = BigVisionMetricWriter(xid, wid, workdir, config)

    write_note("Initializing train dataset...")
    train_ds, ntrain_img = input_pipeline.training(config.input)

    # Start prefetching already.
    n_prefetch = config.get("prefetch_to_device", 1)

    train_iter = input_pipeline.start_input_pipeline(
        train_ds,
        n_prefetch,
        shard=False if config.wandb.debug_data else True,
        config=config)

    total_steps = steps("total", config, ntrain_img, batch_size)

    def get_steps(name, default=ValueError, cfg=config):
        return steps(name, cfg, ntrain_img, batch_size, total_steps, default)

    chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size,
                  measure=metric.measure, write_note=write_note)

    info("Running for %d steps, that means %f epochs",
         total_steps, total_steps * batch_size / ntrain_img)

    if config.wandb.debug_data:
        for step, batch in zip(range(0 + 1, total_steps + 1), train_iter):
            metric.step_start(step)

            print(step, tf.shape(batch['labels']))
            if config.wandb.log_wandb:
                wandb_image = [
                    wandb.Image(
                        batch['image'][i]) for i in range(
                        0, 1024, 100)]
                wandb.log({'demo': wandb_image})

            if step == 10900:
                exit()

        exit()
    write_note(f"Initializing {config.model_name} model...")
    model_mod = importlib.import_module(f"models.{config.model_name}")
    model = model_mod.Model(**config.get("model", {}))

    # We want all parameters to be created in host RAM, not on any device, they'll
    # be sent there later as needed, otherwise we already encountered two
    # situations where we allocate them twice.
    @functools.partial(jax.jit, backend="cpu", static_argnums=1)
    def init(rng, text_size):
        bs = batch_size // jax.device_count()
        image_size = tuple(train_ds.element_spec["image"].shape[1:])
        no_image = jnp.zeros((bs,) + image_size, jnp.float32)
        no_text = jnp.zeros((bs,) + text_size, jnp.int32)
        params = flax.core.unfreeze(
            model.init(rng, no_image, no_text))["params"]
        return params
    if config.get('noun_sample', False):
        text_size = (config.text_length, )
    else:
        text_size = tuple(train_ds.element_spec["labels"].shape[1:])
    rng, rng_init = jax.random.split(rng)
    with chrono.log_timing("z/secs/init"):
        params_cpu = init(rng_init, text_size)

    if jax.process_index() == 0:
        num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
        parameter_overview.log_parameter_overview(
            params_cpu, msg="init params")
        metric.measure("num_params", num_params)

    write_note(f"Initializing {config.optax_name} optimizer...")
    tx, sched_fns = optim.make(config, params_cpu, sched_kw=dict(
        total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

    # We jit this, such that the arrays are created on the CPU, not device[0].
    opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
    sched_fns_cpu = [jax.jit(sched_fn, backend="cpu")
                     for sched_fn in sched_fns]

    @functools.partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
    def update_fn(params, opt, rng, batch):
        """Update step."""
        #assert "mixup" not in config, "We still have to figure out mixup."

        images = batch["image"]
        labels = batch["labels"]

        if config.get("cpu_unit8", True):
            mean = jnp.asarray(
                [0.485 * 255, 0.456 * 255, 0.406 * 255])[None, None, None, :]
            std = jnp.asarray(
                [0.229 * 255, 0.224 * 255, 0.225 * 255])[None, None, None, :]
            images = (jnp.asarray(images, dtype=jnp.float32) - mean) / std


        # Get device-specific loss rng.
        rng, rng_model = jax.random.split(rng, 2)
        rng_model_local = jax.random.fold_in(
            rng_model, jax.lax.axis_index("batch"))

        def loss_fn(params, images, labels):

            zimg, ztxt, extras = model.apply({"params": params}, images, labels, train=True, mask_ratio=config.mask_ratio, rngs={
                "dropout": rng_model_local, 'drop_path': rng_model_local, 'random_mask': rng_model_local})


            if config.get("local_loss", False):
                local_img, local_txt = zimg, ztxt
            else:
                local_img, local_txt = None, None

            # Gather representations across cores for larger batch size for
            # loss.
            if config.get("loss_use_global_batch", False):
                zimg, ztxt = all_gather((zimg, ztxt))

            l, l_extras = losses.bidirectional_contrastive_loss(
                zimg, ztxt, extras["t"], reduction=True, local_loss=config.local_loss, local_img_logits=local_img, local_txt_logits=local_txt)


            return l, {
                "t": extras["t"],
                "t/parameter": extras["t/parameter"],
                "nimg": jnp.mean(extras["img/norm"]),
                "ntxt": jnp.mean(extras["txt/norm"]),
                **l_extras,
            }

        (l, measurements), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, images, labels)
        l, measurements, grads = jax.lax.pmean((l, measurements, grads),
                                               axis_name="batch")
        updates, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, updates)

        def record_grads(grads, measurements):
            img_grads = grads['img']

            if 'embedding' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['embedding'],
                        0.))
                measurements["l2_grad_embeding"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))

            if 'embedding1' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['embedding1'],
                        0.))
                measurements["l2_grad_embeding1"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))
            if 'embedding2' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['embedding2'],
                        0.))
                measurements["l2_grad_embeding2"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))

            if 'embedding3' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['embedding3'],
                        0.))
                measurements["l2_grad_embedding3"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))

            if 'embedding4' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['embedding4'],
                        0.))
                measurements["l2_grad_embedding4"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))
            if 'cls' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['cls'],
                        0.))
               # g = jax.tree_leaves(img_grads['cls'])
                measurements["l2_grad_cls"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))
            if 'head' in img_grads.keys():
                gs = jax.tree_leaves(
                    optim.replace_frozen(
                        config.schedule,
                        img_grads['head'],
                        0.))
                #g = jax.tree_leaves(img_grads['head'])
                measurements["l2_grad_head"] = jnp.sqrt(
                    sum([jnp.vdot(g, g) for g in gs]))

            if 'Transformer' in img_grads.keys():
                for i in range(len(img_grads['Transformer'].keys())):
                    block_name = 'encoderblock_' + str(i)
                    gs = jax.tree_leaves(
                        optim.replace_frozen(
                            config.schedule,
                            img_grads['Transformer'][block_name]['MlpBlock_0']['Dense_1']['kernel'],
                            0.))
                    #g =  jax.tree_leaves(img_grads['Transformer'][block_name]['MlpBlock_0']['Dense_1']['kernel'])
                    measurements["l2_grad_" +
                                 block_name] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
            return measurements

        measurements = record_grads(grads, measurements)
        gs = jax.tree_leaves(optim.replace_frozen(config.schedule, grads, 0.))
        measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
        ps = jax.tree_leaves(params)
        measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
        us = jax.tree_leaves(updates)
        measurements["l2_updates"] = jnp.sqrt(
            sum([jnp.vdot(u, u) for u in us]))

        return params, opt, rng, l, measurements

    # We require hashable function reference for evaluator.
    # We do not jit/pmap this function, because it is passed to evaluator that
    # does it later. We output as many intermediate tensors as possible for
    # maximal flexibility. Later `jit` will prune out things that are not
    # needed.
    def predict_fn(params, image=None, text=None, **unused_kwargs):
        del unused_kwargs  # `unused_kwargs` is to be compatible with few-shot
        zimg, ztxt, out = model.apply({"params": params}, image, text)
        return zimg, ztxt, out

    # Only initialize evaluators when they are first needed.
    @functools.lru_cache(maxsize=None)
    def evaluators():
        return eval_common.from_config(
            config, {"predict": predict_fn},
            lambda s: write_note(f"Init evaluator: {s}…\n{chrono.note}"),
            lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        )

    # Decide how to initialize training. The order is important.
    # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
    # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
    # 3. Initialize model from something, e,g, start a fine-tuning job.
    # 4. Train from scratch.
    resume_ckpt_path = None
    if save_ckpt_path and gfile.exists(save_ckpt_path):
        resume_ckpt_path = save_ckpt_path
    elif config.get("resume"):
        resume_ckpt_path = config.resume.format(wid=xm_wu.id)
    if resume_ckpt_path:
        write_note("Resume training from checkpoint...")
        checkpoint = {
            "params": params_cpu,
            "opt": opt_cpu,
            "chrono": chrono.save(),
        }
        checkpoint_tree = jax.tree_structure(checkpoint)
        loaded = load_checkpoint(checkpoint_tree, resume_ckpt_path)
        # bfloat16 type gets lost when data is saved to disk, so we recover it.
        checkpoint = jax.tree_map(recover_dtype, loaded)
        params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]
        chrono.load(checkpoint["chrono"])
    elif config.get("model_init"):
        write_note(f"Initialize model from {config.model_init}...")
        params_cpu = model_mod.load(
            params_cpu, config.model_init, config.get("model"),
            **config.get("model_load", {}))
        if jax.process_index() == 0:
            parameter_overview.log_parameter_overview(
                params_cpu, msg="restored params")

    elif config.get('masked_init'):

        pretrained_params_cpu = load_params(None, config.masked_init)

        params_cpu = merge_params(pretrained_params_cpu,
                                  params_cpu,
                                  **config.get("masked_no_load",
                                               {'dont_load': []}))

    write_note("Kicking off misc stuff...")
    first_step = optim.get_count(opt_cpu)
    chrono.inform(first_step=first_step)
    prof = None  # Keeps track of start/stop of profiler state.

    write_note(f"Replicating...\n{chrono.note}")
    params_repl = flax.jax_utils.replicate(params_cpu)
    opt_repl = flax.jax_utils.replicate(opt_cpu)

    rng, rng_loop = jax.random.split(rng, 2)
    rngs_loop = flax.jax_utils.replicate(rng_loop)
    ckpt_writer = None

    write_note(f"First step compilations...\n{chrono.note}")
    if config.get('eval_only', False):
        step = 0
        for (name, evaluator, log_steps, prefix) in evaluators():
            if itstime(
                    step,
                    log_steps,
                    total_steps,
                    first=log_steps < total_steps,
                    last=False):
                chrono.pause(wait_for=params_repl)
                # Record things like epoch number, core hours etc.
                chrono.tick(step)
                write_note(f"{name} evaluation...\n{chrono.note}")
                with chrono.log_timing(f"z/secs/eval/{name}"):
                    for key, value in evaluator.run(params_repl):
                        metric.measure(f"{prefix}{key}", value)
                chrono.resume()
        metric.step_end()
        exit()
    # Using a python integer for step here, because opt.state.step is allocated
    # on TPU during replication.
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
        metric.step_start(step)

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            with chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
                params_repl, opt_repl, rngs_loop, loss_value, measurements = update_fn(
                    params_repl, opt_repl, rngs_loop, batch)

        # On the first host, let's always profile a handful of early steps.
        if jax.process_index() == 0:
            prof = startstop_prof(
                prof, step, first_step, get_steps("log_training"))

        # Report training progress
        if (itstime(step, get_steps("log_training"), total_steps, host=0)
                or chrono.warmup and jax.process_index() == 0):
            for i, sched_fn_cpu in enumerate(sched_fns_cpu):
                metric.measure(
                    f"global_schedule{i if i else ''}",
                    sched_fn_cpu(
                        step - 1))
            l = metric.measure("training_loss", loss_value[0])
            for name, value in measurements.items():
                metric.measure(name, value[0])
            chrono.tick(step)
            if not np.isfinite(l):
                raise RuntimeError(
                    f"The loss became nan or inf somewhere within steps "
                    f"[{step - get_steps('log_training')}, {step}]")

        # Checkpoint saving
        if (save_ckpt_path and
                (itstime(step, get_steps("ckpt", None), total_steps, host=0) or
                 itstime(step, get_steps("keep_ckpt", None), total_steps, host=0))):
            chrono.pause(wait_for=(params_repl, opt_repl))
            checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
            # We need to transfer the weights over now or else we risk keeping them
            # alive while they'll be updated in a future step, creating hard to debug
            # memory errors (see (internal link)). Also, takes device 0's
            # params only.
            params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)
            opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)

            # Check whether we want to keep a copy of the current checkpoint.
            copy_step = None
            if itstime(step, get_steps("keep_ckpt", None), total_steps):
                copy_step = step

            ckpt = {
                "params": params_cpu,
                "opt": opt_cpu,
                "chrono": chrono.save()}
            ckpt_writer = pool.apply_async(
                save_checkpoint, (ckpt, save_ckpt_path, copy_step))
            chrono.resume()

        for (name, evaluator, log_steps, prefix) in evaluators():
            if itstime(
                    step,
                    log_steps,
                    total_steps,
                    first=log_steps < total_steps,
                    last=False):
                chrono.pause(wait_for=params_repl)
                # Record things like epoch number, core hours etc.
                chrono.tick(step)
                write_note(f"{name} evaluation...\n{chrono.note}")
                with chrono.log_timing(f"z/secs/eval/{name}"):
                    for key, value in evaluator.run(params_repl):
                        metric.measure(f"{prefix}{key}", value)
                chrono.resume()
        metric.step_end()
        if has_wandb and jax.process_index() == 0:
            if config.wandb.log_wandb:
                wandb.log(metric.step_metrics, step=step)

    # Run evals after done with training. Running them here guarantees evals
    # will run if job is restarted after writting the last checkpoint and
    # also supports eval only runs (when total_steps or num_epochs is 0).
    metric.step_start(total_steps)
    for (name, evaluator, _, prefix) in evaluators():
        write_note(f"{name} evaluation...\n{chrono.note}")
        with chrono.log_timing(f"z/secs/eval/{name}"):
            for key, value in evaluator.run(params_repl):
                metric.measure(f"{prefix}{key}", value)
    if has_wandb and jax.process_index() == 0:
        if config.wandb.log_wandb:
            wandb.log(metric.step_metrics, step=step)
    # Always give a chance to stop the profiler, no matter how things ended.
    # TODO: can we also do this when dying of an exception like OOM?
    if jax.process_index() == 0 and prof is not None:
        startstop_prof(prof)

    # Last note needs to happen before the pool's closed =)
    write_note(f"Done!\n{chrono.note}")

    pool.close()
    pool.join()
    metric.close()

    # Make sure all hosts stay up until the end of main.
    sync()

    maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
    app.run(main)

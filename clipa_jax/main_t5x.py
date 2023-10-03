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
import flax.traverse_util

from transforms.mixup import mixup, cutmix
from helpers.utils import *
from datasets.input_pipeline import shard_and_put
from datasets import input_pipeline
import optimizer
import losses
from tensorflow.io import gfile
from optimizer import replace_frozen, steps, _make_mask_trees, create_learning_rate_schedule
from models_t5x.common import merge_params
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
from helpers.utils import  save_device_memory_profile_to_gcs

import t5x.checkpoints_update
import t5x.model_info
import t5x.rng
import t5x.partitioning
import t5x.optimizers
import  t5x.layers
import t5x.train_state as train_state_lib

from helpers import checkpoint_util as ckp

import jax.profiler
jax.profiler.start_server(9999)
os.makedirs('memory_profile', exist_ok=True)



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

    # initialize pjit ppartitioner
    partitioner = t5x.partitioning.PjitPartitioner(**config.partitioning)
    partitioner._logical_axis_rules += (("_null0", None),)
    partitioner._logical_axis_rules += (("_null1", None),)
    partitioner._logical_axis_rules += (("_null2", None),)
    partitioner._logical_axis_rules += (("classes", None),)


    data_layout = partitioner.get_data_layout(batch_size)

    write_note("Initializing train dataset...")
    train_ds, ntrain_img = input_pipeline.training(config.input, data_layout)

    # Start prefetching already.
    n_prefetch = config.get("prefetch_to_device", 1)

    train_iter = input_pipeline.start_input_pipeline(
        train_ds,
        n_prefetch,
        shard=False if config.partitioning.partition_states else True,
        partitioner=partitioner,
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
            # if config.wandb.log_wandb:
            #     wandb_image = [
            #         wandb.Image(
            #             batch['image'][i]) for i in range(
            #             0, 1024, 100)]
            #     wandb.log({'demo': wandb_image})

            if step == 100:
                exit()

        exit()
    write_note(f"Initializing {config.model_name} model...")
    model_mod = importlib.import_module(f"models_t5x.{config.model_name}")
    model = model_mod.Model(**config.get("model", {}), partitioner=partitioner)

    # We want all parameters to be created in host RAM, not on any device, they'll
    # be sent there later as needed, otherwise we already encountered two
    # situations where we allocate them twice.

    def init(rng, text_size):
        bs = batch_size // jax.device_count()
        image_size = tuple(train_ds.element_spec["image"].shape[1:])
        no_image = jnp.zeros((2,) + image_size, jnp.float32)
        no_text = jnp.zeros((2,) + text_size, jnp.int32)
        params = model.init({"params": rng, "dropout": jax.random.PRNGKey(0),
                             'drop_path': jax.random.PRNGKey(0),  "random_mask": jax.random.PRNGKey(0)},
                            {"image":no_image, "labels":no_text},
                            train=False, mask_ratio=config.mask_ratio,
                            labels=no_text, config=config)
        return params

    if config.get('noun_sample', False):
        text_size = (config.text_length, )
    else:
        text_size = tuple(train_ds.element_spec["labels"].shape[1:])
    rng, rng_init = jax.random.split(rng)

    write_note(f"Initializing scheduler...")

    sched_fns = create_learning_rate_schedule(base=config.lr, total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img,
                                                   **config.schedule[0][1]
                                                   )
    sched_fns_cpu = [sched_fns]

    with chrono.log_timing("z/secs/init"):
        opt_cpu = None
        init_fn = functools.partial(init, text_size=text_size)
        params_shape = jax.eval_shape(init_fn, (rng_init))


        # if config.get("wd"):
        #     wd_mults = config.get("wd_mults", [(".*/kernel$", 1.0)])
        #     mask, _ = _make_mask_trees(params_shape["params"], wd_mults, "config.wd_mults")
        from helpers.opt_util import filter_parameters, filter_bias_and_norm, filter_posembed, filter_t
        mask = jax.tree_util.tree_map(
            lambda x, y, z: bool(x and y and z),
            filter_parameters(params_shape["params"], filter_bias_and_norm),
            filter_parameters(
                params_shape["params"], filter_posembed
            ),  filter_parameters(
                params_shape["params"], filter_t
            ) # Note: we must exclude posembed wd in adamw
        )
        opt_cpu = t5x.optimizers.wrap_optax_optimizer(optax.adamw)

        #opt_cpu = t5x.optimizers.wrap_optax_optimizer(optax.lion)
        opt_cpu = opt_cpu(
            learning_rate=sched_fns,
            **config.optax,
            weight_decay=config.wd,
            mask=mask
        )

        def initialize_train_state(rng_init):
            # split rng for init and for state
            initial_variables = init(rng_init, text_size)
            if opt_cpu:
                return train_state_lib.FlaxOptimTrainState.create(opt_cpu, initial_variables)
            return train_state_lib.InferenceState.create(initial_variables)

        # prepare p_init_fn not actually
        train_state_shape = jax.eval_shape(initialize_train_state, rng_init=rng_init)
        train_state_axes = partitioner.get_mesh_axes(train_state_shape)
        p_init_fn = partitioner.partition(
            initialize_train_state,
            in_axis_resources=None,
            out_axis_resources=train_state_axes,
        )
        t5x.model_info.log_model_info(None, train_state_shape, partitioner)




    def update_fn(params, batch, rng, learning_rate_fn=None):
        """Update step."""
        #assert "mixup" not in config, "We still have to figure out mixup."

        # Get device-specific loss rng.
        rng, rng_model = jax.random.split(rng, 2)

        rng_step_model = jax.random.fold_in(rng_model, params.step)
        def loss_fn(params):
            zimg, ztxt, extras, l = model.apply({"params": params},
                                             batch,
                                             train=True,
                                             mask_ratio=config.mask_ratio,
                                             config=config,
                                             rngs={"dropout": rng_step_model, 'drop_path': rng_step_model, 'random_mask': rng_step_model},
                                             )

            return l, {
                "lr": 0,
                "t": extras["t"],
                "t/parameter": extras["t/parameter"],
                "nimg": jnp.mean(extras["img/norm"]),
                "ntxt": jnp.mean(extras["txt/norm"]),

            }

        (l, measurements), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params.params)
        new_state = params.apply_gradient(
        grads, learning_rate= learning_rate_fn(params.step) if learning_rate_fn else None
    )
        if learning_rate_fn:
            measurements['lr'] = learning_rate_fn(params.step)
        save_device_memory_profile_to_gcs(f"memory_profile/update.prof", workdir)
        return new_state, l, measurements, rng



    # We require hashable function reference for evaluator.
    # We do not jit/pmap this function, because it is passed to evaluator that
    # does it later. We output as many intermediate tensors as possible for
    # maximal flexibility. Later `jit` will prune out things that are not
    # needed.
    def predict_fn(params, batch, labels, **unused_kwargs):
        del unused_kwargs  # `unused_kwargs` is to be compatible with few-shot
        zimg, ztxt, out = model.apply({"params": params.params},  batch, labels=labels, train=False)
        return zimg, ztxt, out


    # Only initialize evaluators when they are first needed.
    @functools.lru_cache(maxsize=None)
    def evaluators():
        return eval_common.from_config(
            config, {"predict": predict_fn},
            lambda s: write_note(f"Init evaluator: {s}…\n{chrono.note}"),
            lambda key, cfg: get_steps(key, default=None, cfg=cfg),
            partitioner=partitioner,
            train_state_axes=train_state_axes,
        )

    t5x_path = os.path.join(workdir, 't5x')
    checkpointer = t5x.checkpoints_update.Checkpointer(
        train_state=train_state_shape,
        partitioner=partitioner,
        checkpoints_dir=t5x_path
    )

    resume_step = None
    if save_ckpt_path and gfile.exists(save_ckpt_path) and gfile.exists(t5x_path):
        resume_step = 190000
        params_tpu = ckp.restore_checkpoint(checkpointer, path=t5x_path, step=resume_step)
        if resume_step:
            first_step = resume_step
        else:
            first_step = t5x.checkpoints_update.latest_step(t5x_path)
        write_note("Resume training from checkpoint...")
        checkpoint = {
                "chrono": chrono.save(),
            }
        checkpoint_tree = jax.tree_structure(checkpoint)
        loaded = load_checkpoint(checkpoint_tree, save_ckpt_path)
        # bfloat16 type gets lost when data is saved to disk, so we recover it.
        checkpoint = jax.tree_map(recover_dtype, loaded)
        chrono.load(checkpoint["chrono"])

    elif config.get('masked_init', None):
        logging.info("Initializing train_state...")


       # params_tpu = p_init_fn(rng_init)
       # params_tpu = p_init_fn(rng_init)



        path = os.path.join(config.masked_init, 't5x')
        first_step = t5x.checkpoints_update.latest_step(path)

        logging.info(f"load pretrain at {first_step} steps")
        path_chkpt = (
            path if first_step is None else t5x.checkpoints_update.get_checkpoint_dir(path, first_step)
        )
        params_tpu = checkpointer.restore(
            path=path_chkpt,
            mask_init=True, #ignore shape mismatch
       #     fallback_state=params_tpu.state_dict(),
       #     state_transformation_fns=[ckp.remove_optimizer_state]
        #    state_transformation_fns = [ ckp.remove_pos_embed]
        )

        print(params_tpu._optimizer.optimizer_def.optax_optimizer)
        logging.info('re-set optimizer needed')
        params_tpu = params_tpu.replace_optimizer(opt_cpu)
        params_tpu._optimizer.optimizer_def.optax_optimizer = optax.adamw(
            learning_rate=sched_fns,
            **config.optax,
            weight_decay=config.wd,
            mask=mask
        )
        #
        # params_tpu._optimizer.optimizer_def.optax_optimizer = optax.lion(
        #     learning_rate=sched_fns,
        #     **config.optax,
        #     weight_decay=config.wd,
        #     mask=mask
        # )

        #update mismatch parameters
        params_update = flax.core.unfreeze(params_tpu.params)
        logging.info('re-load pos-embedding for text transformer')
        pos_loaded = jax.experimental.multihost_utils.process_allgather(params_update['txt']['pos_embedding'])
        pos_update =  jax.image.resize(pos_loaded, (1, text_size[0], pos_loaded.shape[-1]), method='bilinear')
        logging.info(f'updated pos-embedding has size of {pos_update.shape}')
        params_update['txt']['pos_embedding'] = pos_update
        params_update = flax.core.freeze(params_update)
        logging.info('parameter updated successfully')

       # re-init optimizer states
        states_update = flax.core.unfreeze(params_tpu.state_dict()['state'])
        params_states_update = states_update["param_states"]
        logging.info('re-initialize optimizer state') # not sure if needed since it consume some memory
        mu = flax.traverse_util.flatten_dict(params_states_update['0']['mu'])
        nu = flax.traverse_util.flatten_dict(params_states_update['0']['nu'])
        flat_mu = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), mu)
        flat_nu = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), nu)
        params_states_update['0']['mu'] = flax.traverse_util.unflatten_dict(flat_mu)
        params_states_update['0']['nu'] = flax.traverse_util.unflatten_dict(flat_nu)

        for key in ['nu', 'mu']:
            mu_pos_embedded =params_states_update['0'][key]['txt']['pos_embedding']
            mu_pos_loaded = jax.experimental.multihost_utils.process_allgather(mu_pos_embedded)
            mu_pos_update = jax.image.resize(mu_pos_loaded, (1, text_size[0], mu_pos_loaded.shape[-1]), method='bilinear')
            params_states_update['0'][key]['txt']['pos_embedding'] = mu_pos_update
        params_states_update['0'] = flax.core.freeze(params_states_update['0'])
        params_states_update['2']['count'] = 0
        params_states_update = flax.core.freeze(params_states_update)


        states_update["param_states"] = params_states_update
        states_update["step"] = 0
        states_update = flax.core.freeze(states_update)
        from flax.serialization import to_state_dict
        state_dict_updated = to_state_dict(
            {"target": to_state_dict(params_update), "state": to_state_dict(states_update)}
        )
        logging.info('optimizer state updated successfully')


        # params_tpu = params_tpu.replace_params(params=params_update)
        # params_tpu = params_tpu.replace_param_states(param_states=params_states_update)

        params_tpu = params_tpu.restore_state(state_dict_updated)
        first_step = optimizer.get_count(params_tpu._optimizer.state)
        logging.info(f"All weights and states loaded succeffuly start fine-tuning at {first_step}")
        save_device_memory_profile_to_gcs(f"memory_profile/load.prof", workdir)


    else:
        logging.info("Initializing train_state...")
        params_tpu = p_init_fn(rng_init)
        first_step = 0

    write_note("Kicking off misc stuff...")
    # first_step = optimizer.get_count(opt_cpu)

    chrono.inform(first_step=first_step)
    prof = None  # Keeps track of start/stop of profiler state.


    rng, rng_loop = jax.random.split(rng, 2)
    ckpt_writer = None

    update_fn = functools.partial(update_fn, learning_rate_fn=None)

    partitioned_train_step = partitioner.partition(
        update_fn,
        in_axis_resources=(train_state_axes, partitioner.data_partition_spec, None),
        out_axis_resources=(train_state_axes, None, None, None),
        donate_argnums=(0,),
    )


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
                chrono.pause(wait_for=params_tpu)
                # Record things like epoch number, core hours etc.
                chrono.tick(step)
                write_note(f"{name} evaluation...\n{chrono.note}")
                with chrono.log_timing(f"z/secs/eval/{name}"):
                    for key, value in evaluator.run(params_tpu):
                        metric.measure(f"{prefix}{key}", value)
                chrono.resume()
                save_device_memory_profile_to_gcs(f"memory_profile/eval.prof", workdir)
        metric.step_end()
        exit()
    # Using a python integer for step here, because opt.state.step is allocated
    # on TPU during replication.
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
        metric.step_start(step)

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            with chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
                params_tpu, loss_value, measurements, rng_loop = partitioned_train_step(
                    params_tpu, batch, rng_loop)

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
            l = metric.measure("training_loss", loss_value)
            for name, value in measurements.items():
                metric.measure(name, value)
            chrono.tick(step)
            if not np.isfinite(l):
                raise RuntimeError(
                    f"The loss became nan or inf somewhere within steps "
                    f"[{step - get_steps('log_training')}, {step}]")

        # # Checkpoint saving
        if (save_ckpt_path and
                (itstime(step, get_steps("ckpt", None), total_steps, host=0) or
                 itstime(step, get_steps("keep_ckpt", None), total_steps, host=0))):
            chrono.pause(wait_for=(params_tpu))
            checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))

            # Check whether we want to keep a copy of the current checkpoint.
            copy_step = None
            if itstime(step, get_steps("keep_ckpt", None), total_steps):
                copy_step = step

            ckpt = {
                "chrono": chrono.save()}
            ckpt_writer = pool.apply_async(
                save_checkpoint, (ckpt, save_ckpt_path, copy_step))
            logging.info("Saving checkpoint: {}".format(workdir))
            chrono.resume()
        if t5x_path and itstime(
                    step,
                    get_steps("ckpt", None),
                    total_steps) :
            chrono.pause(wait_for=(params_tpu))
            try:
                checkpointer.save(params_tpu)
            except:
                logging.info('This saving failed, contiune training')
            chrono.resume()
        for (name, evaluator, log_steps, prefix) in evaluators():
            if itstime(
                    step,
                    log_steps,
                    total_steps,
                    first=log_steps < total_steps,
                    last=False):
                chrono.pause(wait_for=params_tpu)
                # Record things like epoch number, core hours etc.
                chrono.tick(step)
                write_note(f"{name} evaluation...\n{chrono.note}")
                with chrono.log_timing(f"z/secs/eval/{name}"):
                    for key, value in evaluator.run(params_tpu):
                        metric.measure(f"{prefix}{key}", value)
                chrono.resume()
                save_device_memory_profile_to_gcs(f"memory_profile/eval.prof", workdir)
        metric.step_end()
        if has_wandb and jax.process_index() == 0:
            if config.wandb.log_wandb:
                wandb.log(metric.step_metrics, step=step)
        save_device_memory_profile_to_gcs(f"memory_profile/main.prof", workdir)
    # Run evals after done with training. Running them here guarantees evals
    # will run if job is restarted after writting the last checkpoint and
    # also supports eval only runs (when total_steps or num_epochs is 0).
    metric.step_start(total_steps)
    for (name, evaluator, _, prefix) in evaluators():
        write_note(f"{name} evaluation...\n{chrono.note}")
        with chrono.log_timing(f"z/secs/eval/{name}"):
            for key, value in evaluator.run(params_tpu):
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

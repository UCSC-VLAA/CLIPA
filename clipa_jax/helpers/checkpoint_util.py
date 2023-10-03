# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import jax.image
from absl import logging

import t5x.checkpoints_update


# --------------------------------------------
# checkpoint interfaces
# --------------------------------------------
def restore_checkpoint(checkpointer, path, step=None):
    if step is None:
        step = t5x.checkpoints_update.latest_step(
            path
        )  # try to load the latest checkpoint if not specified
    path_chkpt = (
        path if step is None else t5x.checkpoints_update.get_checkpoint_dir(path, step)
    )
    state = checkpointer.restore(path=path_chkpt)
    return state


def remove_optimizer_state(ckpt_optimizer_state, optimizer_state):
    logging.info("pop state")
    ckpt_optimizer_state.pop("state")
    return ckpt_optimizer_state


def remove_pos_embed(ckpt_optimizer_state, optimizer_state):
    if (
        "pos_embedding" in ckpt_optimizer_state["target"]["img"]
        and "pos_embedding" in optimizer_state["target"]["img"]
    ):
        shape_ckpt = ckpt_optimizer_state["target"]["img"][
            "pos_embedding"
        ]["metadata"]["shape"]
        shape_opt = list(
            optimizer_state["target"]["img"][
                "pos_embedding"
            ].shape
        )
        if not (shape_ckpt == shape_opt):
            logging.info("Removing img pre-trained pos_embedding.")
            ckpt_optimizer_state["target"]["img"].pop("pos_embedding")
    if (
        "pos_embedding" in ckpt_optimizer_state["target"]["txt"]
        and "pos_embedding" in optimizer_state["target"]["txt"]
    ):
        shape_ckpt = ckpt_optimizer_state["target"]["txt"][
            "pos_embedding"
        ]["metadata"]["shape"]
        shape_opt = list(
            optimizer_state["target"]["txt"][
                "pos_embedding"
            ].shape
        )
        if not (shape_ckpt == shape_opt):
            logging.info("Popping txt pre-trained pos_embedding.")
            # ckpt_optimizer_state["target"]["txt"]["pos_embedding"] = jax.image.resize(
            #     ckpt_optimizer_state["target"]["txt"]["pos_embedding"], shape=shape_opt, method='bilinear')
            ckpt_optimizer_state["target"]["txt"].pop("pos_embedding")
    return ckpt_optimizer_state

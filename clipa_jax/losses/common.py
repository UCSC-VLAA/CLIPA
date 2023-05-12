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


import jax
import jax.numpy as jnp

from helpers.utils import onehot


def sigmoid_xent(*, logits, labels, reduction=True):
    # NOTE: This implementation is stable, see these two:
    # (internal link)
    # https://github.com/google/jax/issues/2140
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    nll = -jnp.sum(labels * log_p + (1. - labels) * log_not_p, axis=-1)
    return jnp.mean(nll) if reduction else nll


def bidirectional_contrastive_loss(
        zimg,
        ztxt,
        t,
        mask=None,
        reduction=False,
        local_loss=False,
        local_img_logits=None,
        local_txt_logits=None,
        use_mix=False,
        lam=None):
    """Bidirectional contrastive losses (e.g. for contrastive trainer/evaluator)."""
    # BF.FB = BB
    if not local_loss:
        logits = jnp.dot(zimg, ztxt.T) * t

        if mask is not None:
            # Set to negative infinity where mask = 0. Masked examples will disappear
            # under softmax, and be ignored by ncorrect (NINF will never win
            # argmax).
            exclude = jnp.logical_not(mask)  # Now 1 if we don't want to keep.
            exclude = jnp.logical_or(exclude[:, None], exclude[None, :])
            logits = jnp.where(exclude, jnp.NINF, logits)

        # Note: assumed t is in a good range e.g. already passed through
        # exp/softplus.
        l1 = -jnp.diag(jax.nn.log_softmax(logits, axis=1))  # NLL img->txt
        l2 = -jnp.diag(jax.nn.log_softmax(logits, axis=0))  # NLL txt->img
        l = 0.5 * (l1 + l2)

        if mask is not None:
            l = jnp.where(mask, l, 0)

        redux = jnp.mean if reduction else lambda x: x
        if reduction and mask is not None:
            def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

    else:
        rank = jax.lax.axis_index('batch')
        logits_img = jax.nn.log_softmax(
            jnp.dot(local_img_logits, ztxt.T) * t, axis=1)
        logits_txt = jax.nn.log_softmax(
            jnp.dot(local_txt_logits, zimg.T) * t, axis=1)

        l1 = -jnp.array([logits_img[i][i + rank * logits_img.shape[0]]
                         for i in range(logits_img.shape[0])])
        l2 = -jnp.array([logits_txt[i][i + rank * logits_txt.shape[0]]
                         for i in range(logits_txt.shape[0])])

        l = 0.5 * (l1 + l2)

        redux = jnp.mean if reduction else lambda x: x
        if reduction and mask is not None:
            def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

        return redux(l), {
            "ncorrect": redux(
                jnp.argmax(
                    logits_img, axis=1) == jnp.arange(
                    len(logits_img))), }

    # Also return extra measurements.
    return redux(l), {
        "ncorrect": redux(
            jnp.argmax(
                logits, axis=1) == jnp.arange(
                len(logits))), }


def softmax_xent(*, logits, labels, reduction=True, kl=False, axis=-1):
    log_p = jax.nn.log_softmax(logits, axis=axis)
    nll = -jnp.sum(labels * log_p, axis=axis)
    if kl:
        nll += jnp.sum(labels * jnp.log(jnp.clip(labels, 1e-8)), axis=axis)
    return jnp.mean(nll) if reduction else nll


def bce_logits(*, logits, labels, weight=None, reduction=True):

    def bce(logits, labels, weight=None, reduction=True):
        """
        Binary Cross Entropy Loss
        Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
        :param x: Input tensor
        :param y: Target tensor
        :param weight: Vector of example weights
        :param average: Boolean to average resulting loss vector
        :return: Scalar value
        """
        max_val = jnp.clip(logits, 0, None)
        loss = logits - logits * labels + max_val + \
            jnp.log(jnp.exp(-max_val) + jnp.exp((-logits - max_val)))

        if weight is not None:
            loss = loss * weight

        if reduction:
            return loss.mean()
        else:
            return loss
    return jnp.mean(jax.vmap(bce)(logits, labels))


def weighted_softmax_xent(*,
                          logits,
                          labels,
                          reduction=True,
                          weights=None,
                          label_smoothing=0.0,
                          normalize=True):
    """Compute weighted cross entropy.

    Args:
     logits: [batch, length, num_classes] float array.
     labels: categorical targets [batch, length] int array.
     reduction: reduce across batch dim.
     weights: None or array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.
     normalize: normalize each "sentence" losses by the number of tokens in it.

    Returns:
      Tuple of scalar losses and batch normalizing factor.
    """
    if logits.ndim != labels.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets" %
            (str(
                logits.shape), str(
                labels.shape)))
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    soft_targets = onehot(
        labels, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)

    normalizing_factor = labels.shape[1]
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum(axis=1)

    loss = loss.sum(axis=1)
    if normalize:
        loss = loss / normalizing_factor

    return loss.mean() if reduction else loss


def mae_loss(*, pred, target, mask, norm_pix_loss: bool = True):
    if norm_pix_loss:
        mean = target.mean(axis=-1, keepdims=True)
       # var = target.var(axis=-1, keepdims=True)
        var = target.var(axis=-1, keepdims=True) * \
            target.shape[-1] / (target.shape[-1] - 1)  # unbiased version
        target = (target - mean) / (var + 1e-6) ** .5
    loss = (pred - target) ** 2

    loss = loss.mean(axis=-1)

    loss = (loss * mask).sum() / mask.sum()

    return loss

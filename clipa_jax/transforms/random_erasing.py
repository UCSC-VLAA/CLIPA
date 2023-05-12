#Copyright @2023 Xianhang Li

# This code is based on materials from the TF-Models [https://github.com/tensorflow/models].
# Thanks to TF-Models for their contributions to the field of computer vision and for their open-source contributions to this project.

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

from typing import Tuple, Optional
import tensorflow as tf
import math



@tf.function
def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
  """Fills blank area."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])

  if replace is None:
    fill = tf.random.normal(cutout_shape+[3], dtype=image.dtype)
    fill = tf.pad(fill, padding_dims+[[0, 0]], constant_values=1)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)

  return image



class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(
      self,
      image: tf.Tensor
  ) -> tf.Tensor:
    """Given an image tensor, returns a distorted image with the same shape.
    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.
    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()

  def distort_with_boxes(
      self,
      image: tf.Tensor,
      bboxes: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Distorts the image and bounding boxes.
    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.
      bboxes: `Tensor` of shape [num_boxes, 4] or [num_frames, num_boxes, 4]
        representing bounding boxes for an image or image sequence.
    Returns:
      The augmented version of `image` and `bboxes`.
    """
    raise NotImplementedError




class RandomErasing(ImageAugment):
  """Applies RandomErasing to a single image.
  Reference: https://arxiv.org/abs/1708.04896
  Implementation is inspired by
  https://github.com/rwightman/pytorch-image-models.
  """

  def __init__(self,
               probability: float = 0.25,
               min_area: float = 0.02,
               max_area: float = 1 / 3,
               min_aspect: float = 0.3,
               max_aspect: Optional[float] = None,
               min_count=1,
               max_count=1,
               trials=10):
    """Applies RandomErasing to a single image.
    Args:
      probability: Probability of augmenting the image. Defaults to `0.25`.
      min_area: Minimum area of the random erasing rectangle. Defaults to
        `0.02`.
      max_area: Maximum area of the random erasing rectangle. Defaults to `1/3`.
      min_aspect: Minimum aspect rate of the random erasing rectangle. Defaults
        to `0.3`.
      max_aspect: Maximum aspect rate of the random erasing rectangle. Defaults
        to `None`.
      min_count: Minimum number of erased rectangles. Defaults to `1`.
      max_count: Maximum number of erased rectangles. Defaults to `1`.
      trials: Maximum number of trials to randomly sample a rectangle that
        fulfills constraint. Defaults to `10`.
    """
    self._probability = probability
    self._min_area = float(min_area)
    self._max_area = float(max_area)
    self._min_log_aspect = math.log(min_aspect)
    self._max_log_aspect = math.log(max_aspect or 1 / min_aspect)
    self._min_count = min_count
    self._max_count = max_count
    self._trials = trials

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Applies RandomErasing to single `image`.
    Args:
      image (tf.Tensor): Of shape [height, width, 3] representing an image.
    Returns:
      tf.Tensor: The augmented version of `image`.
    """
    uniform_random = tf.random.uniform(shape=[], minval=0., maxval=1.0)
    mirror_cond = tf.less(uniform_random, self._probability)
    image = tf.cond(mirror_cond, lambda: self._erase(image), lambda: image)
    return image


  def _erase(self, image: tf.Tensor) -> tf.Tensor:
    """Erase an area."""
    if self._min_count == self._max_count:
      count = self._min_count
    else:
      count = tf.random.uniform(
          shape=[],
          minval=int(self._min_count),
          maxval=int(self._max_count - self._min_count + 1),
          dtype=tf.int32)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    area = tf.cast(image_width * image_height, tf.float32)

    for _ in range(count):
      # Work around since break is not supported in tf.function
      is_trial_successfull = False
      for _ in range(self._trials):
        if not is_trial_successfull:
          erase_area = tf.random.uniform(
              shape=[],
              minval=area * self._min_area,
              maxval=area * self._max_area)
          aspect_ratio = tf.math.exp(
              tf.random.uniform(
                  shape=[],
                  minval=self._min_log_aspect,
                  maxval=self._max_log_aspect))

          half_height = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area * aspect_ratio) / 2),
              dtype=tf.int32)
          half_width = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area / aspect_ratio) / 2),
              dtype=tf.int32)

          if 2 * half_height < image_height and 2 * half_width < image_width:
            center_height = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_height - 2 * half_height),
                dtype=tf.int32)
            center_width = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_width - 2 * half_width),
                dtype=tf.int32)

            image = _fill_rectangle(
                image,
                center_width,
                center_height,
                half_width,
                half_height,
                replace=None)

            is_trial_successfull = True

    return image
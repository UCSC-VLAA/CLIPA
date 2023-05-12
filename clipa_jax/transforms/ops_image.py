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

"""Image-centric preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.

The key named "image" is commonly used for the image, and is a 3D tensor of
shape (height x width x channels).
"""
import functools
from typing import Optional

from tensorflow.python.ops import random_ops
from tensorflow.python.ops.image_ops_impl import adjust_brightness

from transforms import autoaugment
from helpers.registry import InKeyOutKey, maybe_repeat, Registry


import tensorflow as tf

from transforms.autoaugment import color, contrast, brightness, check_zero_image

from transforms.random_erasing import RandomErasing
from transforms.three_aug import distort_image_with_three_aug


@Registry.register("preprocess_ops.decode")
@InKeyOutKey(indefault="jpg", outdefault="image")
def get_decode(channels=3):
    """Decode an encoded image string, see tf.io.decode_image."""

    def _decode(image):
        return tf.io.decode_image(
            image, channels=channels, expand_animations=False)

    return _decode


@Registry.register("preprocess_ops.resize")
@InKeyOutKey()
def get_resize(size, method="bilinear", antialias=True):
    """Resizes image to a given size.

    Args:
      size: either an integer H, where H is both the new height and width
        of the resized image, or a list or tuple [H, W] of integers, where H and W
        are new image"s height and width respectively.
      method: resize method, see tf.image.resize docs for options.
      antialias: see tf.image.resize. Ideally set to True for all new configs.

    Returns:
      A function for resizing an image.

    """
    size = maybe_repeat(size, 2)

    def _resize(image):
        """Resizes image to a given size."""
        # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
        # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
        # In particular it was not equivariant with rotation and lead to the network
        # to learn a shortcut in self-supervised rotation task, if rotation was
        # applied after resize.
        dtype = image.dtype
        image = tf.image.resize(
            image, size, method=method, antialias=antialias)
        return tf.cast(image, dtype)

    return _resize


@Registry.register("preprocess_ops.resize_small")
@InKeyOutKey()
def get_resize_small(smaller_size, method="area", antialias=True):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio.

    Args:
      smaller_size: an integer, that represents a new size of the smaller side of
        an input image.
      method: the resize method. `area` is a meaningful, bwd-compat default.
      antialias: see tf.image.resize. Ideally set to True for all new configs.

    Returns:
      A function, that resizes an image and preserves its aspect ratio.

    Note:
      backwards-compat for "area"+antialias tested here:
      (internal link)
    """

    def _resize_small(image):  # pylint: disable=missing-docstring
        h, w = tf.shape(image)[0], tf.shape(image)[1]

        # Figure out the necessary h/w.
        ratio = (
            tf.cast(smaller_size, tf.float32) /
            tf.cast(tf.minimum(h, w), tf.float32))
        h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
        w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

        dtype = image.dtype
        image = tf.image.resize(
            image, (h, w), method=method, antialias=antialias)
        return tf.cast(image, dtype)

    return _resize_small


@Registry.register("preprocess_ops.inception_crop")
@InKeyOutKey()
def get_inception_crop(size=None, area_min=5, area_max=100,
                       method="bilinear", antialias=False):
    """Makes inception-style image crop.

    Inception-style crop is a random image crop (its size and aspect ratio are
    random) that was used for training Inception models, see
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

    Args:
      size: Resize image to [size, size] after crop.
      area_min: minimal crop area.
      area_max: maximal crop area.
      method: rezied method, see tf.image.resize docs for options.
      antialias: see tf.image.resize. Ideally set to True for all new configs.

    Returns:
      A function, that applies inception crop.
    """

    def _inception_crop(image):  # pylint: disable=missing-docstring
        begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            tf.zeros([0, 0, 4], tf.float32),
            area_range=(area_min / 100, area_max / 100),
            min_object_covered=0,  # Don't enforce a minimum area.
            use_image_if_no_bounding_boxes=True)
        crop = tf.slice(image, begin, crop_size)
        # Unfortunately, the above operation loses the depth-dimension. So we need
        # to restore it the manual way.
        crop.set_shape([None, None, image.shape[-1]])
        if size:
            crop = get_resize(size, method, antialias)(
                {"image": crop})["image"]
        return crop

    return _inception_crop


@Registry.register("preprocess_ops.decode_jpeg_and_inception_crop")
@InKeyOutKey()
def get_decode_jpeg_and_inception_crop(
        size=None,
        area_min=5,
        area_max=100,
        aspect_ratio=[
            0.75,
            1.33],
    method="bilinear",
        antialias=False):
    """Decode jpeg string and make inception-style image crop.

    Inception-style crop is a random image crop (its size and aspect ratio are
    random) that was used for training Inception models, see
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

    Args:
      size: Resize image to [size, size] after crop.
      area_min: minimal crop area.
      area_max: maximal crop area.
      method: rezied method, see tf.image.resize docs for options.
      antialias: see tf.image.resize. Ideally set to True for all new configs.

    Returns:
      A function, that applies inception crop.
    """

    def _inception_crop(image_data):  # pylint: disable=missing-docstring
        shape = tf.image.extract_jpeg_shape(image_data)
        begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
            shape,
            tf.zeros([0, 0, 4], tf.float32),
            area_range=(area_min / 100, area_max / 100),
            min_object_covered=0,  # Don't enforce a minimum area.
            aspect_ratio_range=aspect_ratio,
            use_image_if_no_bounding_boxes=True)

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(begin)
        target_height, target_width, _ = tf.unstack(crop_size)
        crop_window = tf.stack(
            [offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(
            image_data, crop_window, channels=3)

        if size:
            image = get_resize(size, method, antialias)(
                {"image": image})["image"]

        return image

    return _inception_crop


def random_brightness(image, min_delta, max_delta, seed=None):
    """Adjust the brightness of images by a random factor.

    Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
    interval `[-max_delta, max_delta)`.

    For producing deterministic results given a `seed` value, use
    `tf.image.stateless_random_brightness`. Unlike using the `seed` param
    with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
    same results given the same seed independent of how many times the function is
    called, and independent of global seed settings (e.g. tf.random.set_seed).

    Args:
      image: An image or images to adjust.
      max_delta: float, must be non-negative.
      seed: A Python integer. Used to create a random seed. See
        `tf.compat.v1.set_random_seed` for behavior.

    Usage Example:

    >>> x = [[[1.0, 2.0, 3.0],
    ...       [4.0, 5.0, 6.0]],
    ...      [[7.0, 8.0, 9.0],
    ...       [10.0, 11.0, 12.0]]]
    >>> tf.image.random_brightness(x, 0.2)
    <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>

    Returns:
      The brightness-adjusted image(s).

    Raises:
      ValueError: if `max_delta` is negative.
    """
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')

    delta = random_ops.random_uniform([], min_delta, max_delta, seed=seed)
    return adjust_brightness(image, delta)


@Registry.register("preprocess_ops.random_crop")
@InKeyOutKey()
def get_random_crop(crop_size):
    """Makes a random crop of a given size.

    Args:
      crop_size: either an integer H, where H is both the height and width of the
        random crop, or a list or tuple [H, W] of integers, where H and W are
        height and width of the random crop respectively.

    Returns:
      A function, that applies random crop.
    """
    crop_size = maybe_repeat(crop_size, 2)

    def _crop(image):
        return tf.image.random_crop(image, (*crop_size, image.shape[-1]))

    return _crop


@Registry.register("preprocess_ops.simclr_jitter_gray")
@InKeyOutKey()
def random_color_jitter(jitter_strength=0.5, impl='simclrv2'):
    from transforms.simclr_aug import color_jitter, random_apply, to_grayscale

    def _transform(image):
        image = tf.cast(image, tf.float32) / 255.
        color_jitter_t = functools.partial(
            color_jitter, strength=jitter_strength, impl=impl)

        image = random_apply(color_jitter_t, p=0.8, x=image)

        image = random_apply(to_grayscale, p=0.2, x=image)
        image = tf.clip_by_value(image * 255, 0, 255)
        image = tf.cast(image, tf.uint8)
        return image

    return _transform


@Registry.register("preprocess_ops.central_crop")
@InKeyOutKey()
def get_central_crop(crop_size):
    """Makes central crop of a given size.

    Args:
      crop_size: either an integer H, where H is both the height and width of the
        central crop, or a list or tuple [H, W] of integers, where H and W are
        height and width of the central crop respectively.

    Returns:
      A function, that applies central crop.
    """
    crop_size = maybe_repeat(crop_size, 2)

    def _crop(image):
        h, w = crop_size[0], crop_size[1]
        dy = (tf.shape(image)[0] - h) // 2
        dx = (tf.shape(image)[1] - w) // 2
        return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

    return _crop


@Registry.register("preprocess_ops.flip_lr")
@InKeyOutKey()
def get_random_flip_lr():
    """Flips an image horizontally with probability 50%."""

    def _random_flip_lr_pp(image):
        return tf.image.random_flip_left_right(image)

    return _random_flip_lr_pp


@Registry.register("preprocess_ops.vgg_value_range")
@InKeyOutKey()
def get_vgg_value_range(
    mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
    std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
):
    """VGG-style preprocessing, subtracts mean and divides by stddev.

    This preprocessing is very common for ImageNet pre-trained models since VGG,
    and to this day the standard for models coming from most PyTorch codes.

    Args:
      mean: Tuple of values to be subtracted.
      std: Tuple of values to be divided by.

    Returns:
      A function to rescale the values.
    """
    mean = tf.constant(mean, tf.float32)
    std = tf.constant(std, tf.float32)

    def _vgg_value_range(image):
        return (tf.cast(image, tf.float32) - mean) / std
    return _vgg_value_range


@Registry.register("preprocess_ops.color_jitter")
@InKeyOutKey()
def get_color_jitter(color_jitter):
    """Makes a random crop of a given size.

    Args:
      color_jitter: .

    Returns:
      A function, that applies color jitter.
    """
    if isinstance(color_jitter, (list, tuple)):
        assert len(color_jitter) == 3
    else:
        color_jitter = (float(color_jitter),) * 3
    a, b, c = color_jitter

    def _color_jitter(image):
        #image2 = brightness(image, tf.random.uniform([], minval=max(0, 1 - a), maxval=1 + a))
        image2 = image
        image3 = tf.image.random_contrast(image2, max(0, 1 - b), 1 + b)
        image4 = tf.image.random_saturation(image3, max(0, 1 - c), 1 + c)
        return image4
    return _color_jitter


@Registry.register("preprocess_ops.color_jitter_timm")
@InKeyOutKey()
def get_color_jitter_timm(color_jitter):
    """Makes a random crop of a given size.

    Args:
      color_jitter: .

    Returns:
      A function, that applies color jitter.
    """
    if isinstance(color_jitter, (list, tuple)):
        assert len(color_jitter) == 3
    else:
        color_jitter = (float(color_jitter),) * 3
    a, b, c = color_jitter

    def _color_jitter_timm(image):
        func_list = [brightness, contrast, color]
        idx = [0, 1, 2]
        idx_shuffle = tf.random.shuffle(idx)
        for idx in idx_shuffle:
            for id, func in enumerate(func_list):
                image = tf.cond(
                    tf.equal(
                        id, idx), lambda: func_list[id](
                        image, tf.random.uniform(
                            [], minval=max(
                                0, 1 - a), maxval=1 + a)), lambda: image)
        return image
    return _color_jitter_timm


@Registry.register("preprocess_ops.randaug")
@InKeyOutKey()
def get_randaug(
        num_layers: int = 2,
        magnitude: int = 10,
        increase: bool = True,
        timm: bool = False):
    """Creates a function that applies RandAugment.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719,

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 30].

    Returns:
      a function that applies RandAugment.
    """

    def _randaug(image):
        if timm:
            return autoaugment.distort_image_with_randaugment_timm(
                image, num_layers, magnitude, increase=increase)
        else:
            return autoaugment.distort_image_with_randaugment(
                image, num_layers, magnitude, increase=increase)

    return _randaug


@Registry.register("preprocess_ops.RandomErasing")
@InKeyOutKey()
def get_randerasing(probability: float = 0.25,
                    min_area: float = 0.02,
                    max_area: float = 1 / 3,
                    min_aspect: float = 0.3,
                    max_aspect: Optional[float] = None,
                    min_count=1,
                    max_count=1):

    rand_erase = RandomErasing(probability, min_area, max_area, min_aspect,
                               max_aspect, min_count, max_count)

    def _rand_erase(image):
        return rand_erase.distort(image)

    return _rand_erase


@Registry.register("preprocess_ops.threeaug")
@InKeyOutKey()
def get_threeaug(num_layers: int = 1):
    """Creates a function that applies RandAugment.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719,

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 30].

    Returns:
      a function that applies RandAugment.
    """

    def _three_aug(image):
        image = distort_image_with_three_aug(image)
        return image

    return _three_aug

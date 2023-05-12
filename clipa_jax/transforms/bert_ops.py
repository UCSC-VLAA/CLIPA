#Copyright @2023 Xianhang Li

# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

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

"""BERT-related preprocessing ops (using WordPiece tokenizer)."""
import functools
import logging

from helpers.registry import Registry, InKeyOutKey
from helpers import utils as bv_utils
import tensorflow as tf
import tensorflow_text

import  numpy as np

import functools
import logging

from helpers.registry import Registry, InKeyOutKey
from helpers import utils as bv_utils
import tensorflow as tf
import tensorflow_text

import  numpy as np

# Internally using
# BasicTokenizer
# https://github.com/tensorflow/text/blob/df5250d6cf1069990df4bf55154867391ab5381a/tensorflow_text/python/ops/bert_tokenizer.py#L67
# WordpieceTokenizer
# https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/wordpiece_tokenizer.py
def _create_bert_tokenizer(vocab_path):
  with tf.io.gfile.GFile(vocab_path) as f:
    vocab = f.read().split("\n")
  cls_token = vocab.index("[CLS]")
  return cls_token, tensorflow_text.BertTokenizer(
      vocab_path,
      token_out_type=tf.int32,
      lower_case=True,
  )

def get_order(x):
    if x.startswith('NN'):
        return 1
    elif x.startswith('JJ'):
        return 2
    elif x.startswith('VB'):
        return 3
    else:
        return 4

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def _create_noun_tokenizer(vocab_path):
  with tf.io.gfile.GFile(vocab_path) as f:
    vocab = f.read().split("\n")
  cls_token = vocab.index("[CLS]")

  list_tokens = [nltk.tokenize.word_tokenize(v) for v in vocab]
  pos_tags = [nltk.pos_tag(t) for t in list_tokens]
  pos_list = []
  for v in range(len(vocab)):
      if len(pos_tags[v]) >= 1:
          pos_list.append(get_order(pos_tags[v][-1][1]))
      elif len(pos_tags[v]) == 0:
          pos_list.append(get_order(''))

  pos_tensor = tf.convert_to_tensor(pos_list)
  return cls_token, pos_tensor, tensorflow_text.BertTokenizer(
      vocab_path,
      token_out_type=tf.int32,
      lower_case=True,
  )

@Registry.register("preprocess_ops.bert_tokenize")
@InKeyOutKey(indefault=None, outdefault="labels")
def get_pp_bert_tokenize(vocab_path, max_len, sample_if_multi=True):
  """Extracts tokens with tensorflow_text.BertTokenizer.

  Args:
    vocab_path: Path to a file containing the vocabulry for the WordPiece
      tokenizer. It's the "vocab.txt" file in the zip file downloaded from
      the original repo https://github.com/google-research/bert
    max_len: Number of tokens after tokenization.
    sample_if_multi: Whether the first text should be taken (if set to `False`),
      or whether a random text should be tokenized.

  Returns:
    A preprocessing Op.
  """

  cls_token, tokenizer = _create_bert_tokenizer(vocab_path)

  def _pp_bert_tokenize(labels):

    labels = tf.reshape(labels, (-1,))
    labels = tf.concat([labels, [""]], axis=0)
    if sample_if_multi:
      num_texts = tf.maximum(tf.shape(labels)[0] - 1, 1)  # Don't sample "".
      txt = labels[tf.random.uniform([], 0, num_texts, dtype=tf.int32)]
    else:
      txt = labels[0]  # Always works, since we append "" earlier on.

    token_ids = tokenizer.tokenize(txt[None])
    padded_token_ids, mask = tensorflow_text.pad_model_inputs(
        token_ids, max_len - 1)
    del mask  # Recovered from zero padding in model.
    count = tf.shape(padded_token_ids)[0]
    padded_token_ids = tf.concat(
        [tf.fill([count, 1], cls_token), padded_token_ids], axis=1)
    return padded_token_ids[0]

  return _pp_bert_tokenize

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor


def pad_model_inputs(input, max_seq_length, pad_value=0):  # pylint: disable=redefined-builtin
  r"""Pad model input and generate corresponding input masks.

  `pad_model_inputs` performs the final packaging of a model's inputs commonly
  found in text models. This includes padding out (or simply truncating) to a
  fixed-size, 2-dimensional `Tensor` and generating mask `Tensor`s (of the same
  2D shape) with values of 0 if the corresponding item is a pad value and 1 if
  it is part of the original input.

  Note that a simple truncation strategy (drop everything after max sequence
  length) is used to force the inputs to the specified shape. This may be
  incorrect and users should instead apply a `Trimmer` upstream to safely
  truncate large inputs.

  >>> input_data = tf.ragged.constant([
  ...            [101, 1, 2, 102, 10, 20, 102],
  ...            [101, 3, 4, 102, 30, 40, 50, 60, 70, 80],
  ...            [101, 5, 6, 7, 8, 9, 102, 70],
  ...        ], np.int32)
  >>> data, mask = pad_model_inputs(input=input_data, max_seq_length=9)
  >>> print("data: %s, mask: %s" % (data, mask))
    data: tf.Tensor(
    [[101   1   2 102  10  20 102   0   0]
     [101   3   4 102  30  40  50  60  70]
     [101   5   6   7   8   9 102  70   0]], shape=(3, 9), dtype=int32),
    mask: tf.Tensor(
    [[1 1 1 1 1 1 1 0 0]
     [1 1 1 1 1 1 1 1 1]
     [1 1 1 1 1 1 1 1 0]], shape=(3, 9), dtype=int32)

  Args:
    input: A `RaggedTensor` with rank >= 2.
    max_seq_length: An int, or scalar `Tensor`. The "input" `Tensor` will be
      flattened down to 2 dimensions and then have its 2nd dimension either
      padded out or truncated to this size.
    pad_value: An int or scalar `Tensor` specifying the value used for padding.

  Returns:
      A tuple of (padded_input, pad_mask) where:

      padded_input: A `Tensor` corresponding to `inputs` that has been
        padded/truncated out to a fixed size and flattened to 2
        dimensions.
      pad_mask: A `Tensor` corresponding to `padded_input` whose values are
        0 if the corresponding item is a pad value and 1 if it is not.
  """
  with ops.name_scope("pad_model_inputs"):
    # Verify that everything is a RaggedTensor
    if not isinstance(input, ragged_tensor.RaggedTensor):
      raise TypeError("Expecting a `RaggedTensor`, instead found: " +
                      str(input))

    # Flatten down to `merge_axis`
    input = input.merge_dims(1, -1) if input.ragged_rank > 1 else input

    # Pad to fixed Tensor
    target_shape = math_ops.cast([-1, max_seq_length], dtypes.int64)
    padded_input = input.to_tensor(shape=target_shape, default_value=pad_value)

    # Get padded input mask
    input_mask = array_ops.ones_like(input)
    padded_input_mask = input_mask.to_tensor(shape=target_shape)

    return padded_input, padded_input_mask

@Registry.register("preprocess_ops.change_keys")
@InKeyOutKey(indefault=None, outdefault="labels")
def get_pass_keys():
  def _pp_ass_keys(labels):
      return labels

  return _pp_ass_keys


@Registry.register("preprocess_ops.noun_tokenize")
@InKeyOutKey(indefault=None, outdefault="labels")
def get_pass_keys(vocab_path, max_len, sample_if_multi=True):

  cls_token, pos_tensor ,tokenizer = _create_noun_tokenizer(vocab_path)

  def _pp_noun_tokenize(labels):
      labels = tf.reshape(labels, (-1,))
      labels = tf.concat([labels, [""]], axis=0)
      if sample_if_multi:
          num_texts = tf.maximum(tf.shape(labels)[0] - 1, 1)  # Don't sample "".
          txt = labels[tf.random.uniform([], 0, num_texts, dtype=tf.int32)]
      else:
          txt = labels[0]  # Always works, since we append "" earlier on.

      token_ids = tokenizer.tokenize(txt[None])

      if tf.shape(token_ids[0])[0] > (max_len - 1):
          pos_id = tf.gather(pos_tensor, token_ids)
          pos_id = pos_id.to_tensor()
          pos_id_index = tf.argsort(pos_id, axis=1)
          pos_id_index = tf.sort(pos_id_index[:, :max_len - 1], axis=1)
          sampled_token = tf.gather(token_ids, pos_id_index[:, :, 0], axis=1)
          padded_token_ids, mask = tensorflow_text.pad_model_inputs(
              sampled_token, max_len - 1)

          count = tf.shape(padded_token_ids)[0]
          padded_token_ids = tf.concat(
              [tf.fill([count, 1], cls_token), padded_token_ids], axis=1)
          output = padded_token_ids[0]

      else:
          padded_token_ids, mask = tensorflow_text.pad_model_inputs(
              token_ids, max_len - 1)
          del mask  # Recovered from zero padding in model.
          count = tf.shape(padded_token_ids)[0]
          padded_token_ids = tf.concat(
              [tf.fill([count, 1], cls_token), padded_token_ids], axis=1)
          output = padded_token_ids[0]


      return output

  return _pp_noun_tokenize

@Registry.register("preprocess_ops.custom_bert_tokenize")
@InKeyOutKey(indefault=None, outdefault="labels")
def get_pp_bert_tokenize(vocab_path, max_len, sample_if_multi=True, mask_type='first', train=True):
  """Extracts tokens with tensorflow_text.BertTokenizer.

  Args:
    vocab_path: Path to a file containing the vocabulry for the WordPiece
      tokenizer. It's the "vocab.txt" file in the zip file downloaded from
      the original repo https://github.com/google-research/bert
    max_len: Number of tokens after tokenization.
    sample_if_multi: Whether the first text should be taken (if set to `False`),
      or whether a random text should be tokenized.

  Returns:
    A preprocessing Op.
  """

  cls_token, tokenizer = _create_bert_tokenizer(vocab_path)

  def _pp_bert_tokenize(labels):

    labels = tf.reshape(labels, (-1,))
    labels = tf.concat([labels, [""]], axis=0)
    if sample_if_multi:
      num_texts = tf.maximum(tf.shape(labels)[0] - 1, 1)  # Don't sample "".
      txt = labels[tf.random.uniform([], 0, num_texts, dtype=tf.int32)]
    else:
      txt = labels[0]  # Always works, since we append "" earlier on.


    token_ids = tokenizer.tokenize(txt[None])
    if train:
        if tf.shape(token_ids[0])[0] > (max_len - 1):
          sequence_length = tf.shape(token_ids[0])[0]

          if mask_type=='first':
              padded_token_ids, mask = tensorflow_text.pad_model_inputs(
                token_ids, max_len - 1)
          elif mask_type=='random':
              token_ids = token_ids.merge_dims(1, -1) if token_ids.ragged_rank > 1 else token_ids

              initial_shape = math_ops.cast([-1, sequence_length], dtypes.int64)
              token_ids = token_ids.to_tensor(shape=initial_shape)

              shuffled_ids = tf.random.shuffle(token_ids[0])
              padded_token_ids = shuffled_ids[0: max_len-1]

              padded_token_ids = tf.expand_dims(padded_token_ids, axis=0)

              target_shape = math_ops.cast([-1, max_len - 1], dtypes.int64)

              padded_token_ids = tf.reshape(padded_token_ids, shape=target_shape)
          elif mask_type=='block':
              token_ids = token_ids.merge_dims(1, -1) if token_ids.ragged_rank > 1 else token_ids

              initial_shape = math_ops.cast([-1, sequence_length], dtypes.int64)

              token_ids = token_ids.to_tensor(shape=initial_shape)

              start_index = tf.random.uniform(shape=(), minval=0,  maxval=sequence_length - max_len + 1, dtype=tf.int32)

              padded_token_ids = token_ids[:, start_index:start_index + max_len - 1]

              target_shape = math_ops.cast([-1, max_len - 1], dtypes.int64)

              padded_token_ids = tf.reshape(padded_token_ids, shape=target_shape)

        else:
            padded_token_ids, mask = tensorflow_text.pad_model_inputs(
              token_ids, max_len - 1)
    else:
          padded_token_ids, mask = tensorflow_text.pad_model_inputs(
            token_ids, max_len - 1)

    #del mask  # Recovered from zero padding in model.
    padded_token_ids = padded_token_ids[0]

    padded_token_ids = tf.concat(
      [tf.fill([1], cls_token), padded_token_ids], axis=0)
    return padded_token_ids

  return _pp_bert_tokenize



def tokenizer_nltk(data, vocab_path, sample_length):
    """
    :param input_string: numpy.array,
    :return: tf.tensor string
    """

    def get_order(x):
        if x.startswith('NN'):
            return 1
        elif x.startswith('JJ'):
            return 2
        elif x.startswith('VB'):
            return 3
        else:
            return 4


    cls_token, tokenizer = _create_bert_tokenizer(vocab_path)
    list_tokens = nltk.tokenize.word_tokenize(data.decode('utf-8'))
    pos_tags = nltk.pos_tag(list_tokens)

    #  sample the words by get_order method
    order_list = [get_order(tag) for _, tag in pos_tags]
    sorted_ids = np.argsort(np.array(order_list))
    sampled_ids = sorted(sorted_ids[:sample_length - 1])

    # sample the tokens and convert to tf.tensor
    sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)
    sampled_tokens = tf.convert_to_tensor(sampled_tokens, dtype=tf.string)

    sampled_tokens = tf.concat([sampled_tokens, [""]], axis=0)

    token_ids = tokenizer.tokenize(sampled_tokens[None])

    # change shape
    padded_token_ids, mask = tensorflow_text.pad_model_inputs(
        token_ids, sample_length - 1)
    del mask  # Recovered from zero padding in model.
    count = tf.shape(padded_token_ids)[0]
    padded_token_ids = tf.concat(
        [tf.fill([count, 1], cls_token), padded_token_ids], axis=1)

    output = padded_token_ids[0]
    return output


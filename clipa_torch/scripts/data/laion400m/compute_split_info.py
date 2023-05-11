import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hands off my GPU! (or pip install tensorflow-cpu)
import tensorflow_datasets as tfds
from multiprocessing import Pool, Manager
from functools import partial

filename_template = 'gs://path/to/laion400m/{index:05}.tfrecord'
shard_lengths_dict = Manager().dict()
NUM_WORKERS = 16
NUM_SHARDS = 41408
START_SHARD = 0

# multi process version
def count_shard_length(index, shard_lengths_dict):
    tfrecord_path = filename_template.format(index=index)
    dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
    shard_length = dataset.reduce(np.int64(0), lambda x, _: x + 1)
    shard_lengths_dict[index] = int(shard_length.numpy())
    print(f'finishing tfrecod: {index:05}')


with Pool(NUM_WORKERS) as p:
    p.map(partial(count_shard_length, shard_lengths_dict=shard_lengths_dict), range(START_SHARD, NUM_SHARDS))

shard_lengths = [shard_lengths_dict[i] for i in range(START_SHARD, NUM_SHARDS)]
np.save('/path/to/shard_lengths.npy', shard_lengths)


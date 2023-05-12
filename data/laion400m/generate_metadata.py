import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hands off my GPU! (or pip install tensorflow-cpu)
import tensorflow_datasets as tfds


# features
features = tfds.features.FeaturesDict({
    'jpg': tfds.features.Image(encoding_format='jpeg'),
    'txt': tfds.features.Text(), # txt is the real caption to be used for training
    'caption': tfds.features.Text(),
    'height': tf.int64,
    'width': tf.int64,
    'NSFW': tfds.features.Text(),
    'sha256': tfds.features.Text(), # 2u1 tfrecords do not have this attribute
    'exif': tfds.features.Text(),
    'LICENSE': tfds.features.Text(),
    'original_height': tf.int64,
    'original_width': tf.int64,
    'status': tfds.features.Text(),
    'url': tfds.features.Text(),
    'error_message': tfds.features.Text(),
    'key': tfds.features.Text(),
})


# split info
filename_template = tfds.core.ShardedFileTemplate(
    data_dir='gs://path/to/laion400m/',
    template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
    dataset_name='laion400m',
    split='full',
    filetype_suffix='tfrecord',
)
shard_lengths = np.load('/path/to/shard_lengths.npy').tolist()

split_infos = [
    tfds.core.SplitInfo(
        name='full',
        shard_lengths=shard_lengths,
        num_bytes=0,
        filename_template=filename_template,
    ),
]

# write metadata
tfds.folder_dataset.write_metadata(
    data_dir='gs://path/to/laion400m/',
    features=features,
    split_infos=split_infos,
    filename_template=filename_template,
    description="""400M english image/text pairs""",
    homepage='https://laion.ai/blog/laion-400-open-dataset/',
)
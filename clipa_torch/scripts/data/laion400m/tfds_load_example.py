from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds


builder = tfds.builder_from_directory('gs://path/to/laion400m/')
dataset = builder.as_dataset(split='full') # must specify split!
LOG_NITER = 1e5


for step, data in enumerate(dataset):
    if (step + 1 ) % LOG_NITER == 0:
        print(f'data["caption"]: {data["caption"]}')
        print(f'{step} / {len(dataset)}: {step/len(dataset)}')

import pdb
pdb.set_trace()
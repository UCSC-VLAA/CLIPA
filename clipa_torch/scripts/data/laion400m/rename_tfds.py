from functools import partial
from multiprocessing import Pool
from subprocess import run

NUM_WORKERS = 32
NUM_SHARDS = 41408 # 602
START_SHARD = 0

old_tfrecord_filename_template = 'gs://path/to/laion400m/{index:05}.tfrecord'
# this 41408 suffix needs to be changed with NUM_SHARDS
new_tfrecord_filename_template = 'gs://path/to/laion400m/laion400m-full.tfrecord-{index:05}-of-41408'


def rename(index, old_template, new_template):
    old_path = old_template.format(index=index)
    new_path = new_template.format(index=index)
    # for gcs bucket
    run(['gsutil', 'mv', old_path, new_path])
    # run(['mv', old_path, new_path])
    print(f'move {old_path.split("/")[-1]} to {new_path.split("/")[-1]}')

if __name__ == '__main__':
    with Pool(NUM_WORKERS) as p:
        p.map(partial(rename, old_template=old_tfrecord_filename_template, new_template=new_tfrecord_filename_template),
              range(START_SHARD, NUM_SHARDS))
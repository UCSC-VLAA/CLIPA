# LAION400M TFRecord Post-processing Scripts

This directory contains some custom (crudely written) scripts for post-processing the tfrecord files of laion-400m dataset downloaded by `img2dataset`.
After properly running these scripts, tfds can correctly load those tfrecord files generated by img2dataset.
It is as simple as 
```
builder = tfds.builder_from_directory(root)
```

**The file paths and/or names are hard encoded into the code. Make sure to change them before running in your environment.**

See also this [[tfds official doc]](https://www.tensorflow.org/datasets/external_tfrecord)

## Usage
First, compute the number of samples in each shard with `compute_split_info.py`. This will save the number of samples of each tfrecord shard in `/path/to/shard_lengths.npy`.
```
python compute_split_info.py
```

Second, as the naming of `img2dataset` does not follow the default filename_template of tfds `{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`,
we need to rename those tfrecord files.
Also, because `gsutil mv` in gcs is slow, multiprocessing is used in this script `rename_tfds.py`.
```
python rename_tfds.py
```

Forth, we now can finally generate the metadata needed to use tfds API.
```
python generate_metadata.py
```

The last step is to check if now we can successfully load externel tfrecord files with tfds API, shown in in `tfds_load_example.py`.

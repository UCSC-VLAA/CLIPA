# TPU Usage
For convenient TPU training, we also include some instructions on how to acquire TPU access from google cloud, how to setup TPU machines, and how to prepare the environment to run this codebase on TPU.

## TPU Research Cloud(TRC) Program
There is this fantastic TRC program that gives you free access to TPU machines!
Check the [official website](https://sites.research.google/trc/about/) for details.

## Google Cloud Research Credits Program
Another awesome program that gives you a free google cloud credits worth $1000!
Check the [official website](https://edu.google.com/programs/credits/research/?modal_active=none) for details.

## Setup TPU Machines
The official Cloud TPU JAX [document](https://cloud.google.com/tpu/docs/run-calculation-jax) 
and official Cloud TPU PyTorch [document](https://cloud.google.com/tpu/docs/run-calculation-pytorch#pjrt)
should give you some basic ideas on how to do simple training on a single TPU-VM machine with 8 TPU cores.

To support large-scale vision research, more cores with multiple hosts are recommended.
For example, the following command will create TPU Pods with 64 cores, 8 hosts.
```
gcloud alpha compute tpus tpu-vm create tpu-v3-64-pod-vm --zone $ZONE --project $TPU_NAME --accelerator-type v3-64 --version tpu-vm-pt-1.13 --service-account=$SERVICE_ACCOUNT
```

You can then connect to the TPU Pods with
```
gcloud alpha compute tpus tpu-vm ssh tpu-v3-64-pod-vm --zone $ZONE --project $TPU_NAME --worker 0
```

Then, it is just another linux remote server! 
After setting up the gcs buckets and the environment, 
you can follow [README_JAX](clipa_jax/README.MD) to start training using JAX,
and [README_TORCH](clipa_torch/README.md) to start training using PyTorch-XLA.

## Google Cloud Storage
Leveraging TFDS w/ datasets in TFRecord format, streamed from Google Cloud Storage buckets is the most practical / cost-effective solution.
Storing a big dataset like LAION-400M (or even larger LAION-5B) on disks will cost you a lot of money!
Luckily, the `img2dataset` tool allows direct writing to a gcs bucket. 
You can also check the official docs to learn how to manipulate gcs buckets 
via [command](https://cloud.google.com/storage/docs/discover-object-storage-gsutil) or [console](https://cloud.google.com/storage/docs/discover-object-storage-console)

**Important**: Always make sure that your machine and the bucket you are reading data from are located in the same region/zone! 
Reading from a bucket in a different region will burn thousands of dollar a day!!!

A useful approach to prevent that tragedy is to create a specific [service account](https://cloud.google.com/iam/docs/service-accounts-create)
associated with each bucket, 
assign read/write permissions of corresponding bucket to that service account like [here](https://docs.cloudera.com/HDPDocuments/HDP2/HDP-2.6.5/bk_cloud-data-access/content/edit-bucket-permissions.html), 
and use that service account when creating the TPU machines.

A number of TFDS datasets, including ImageNet, are available in TFDS. 
The TFDS dataset pages (https://www.tensorflow.org/datasets/cli) have directions for building various datasets.
You may build them in a different VM or local machine and then uploading to your training bucket.


## Some Useful Commands
- Execute the same command across hosts on a TPU pods: 
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "COMMAND"
```
- Synchronize the content of a specific directory across hosts on a TPU pods:
```
gcloud alpha compute tpus tpu-vm scp --recurse /path/to/dir  $TPU_NAME:/path/to/dir/../  --zone=$ZONE --worker=all --project=$PROJECT_ID
```
- Python processes on TPU often get orphaned. It is always good to try killing all python processes before starting a new train run.
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f python3"
```
Also, the following command helps release TPU usage.
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs"
```
Finally, this command list processes that are using the TPU.
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo lsof -w /dev/accel0"
```

## Some Useful References
- https://github.com/huggingface/pytorch-image-models/blob/bits_and_tpu/timm/bits/README.md
- https://github.com/google-research/big_vision
- We have also provided some example scripts in `./clipa_jax/scripts/` and `./clipa_torch/scripts/exp/tpu/utils/`.  Check them!
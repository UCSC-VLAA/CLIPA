#!/bin/bash

export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-64-pod-vm4

TFDS_DATA_DIR=gs://jaxtpu-tfds-imagenet-eu-west4-a #/imagenet2012/5.1.0
LAION_PATH=gs://jaxtpu-data-eu-west4/laion-400m-cv2resize-356m
WORK_DIR=gs://lxh_jaxtpu_eu_ckpt/clip_inverse_scaling/model_b/
MASK_INIT=gs://MASK_INIT=gs://lxh-jax-us-east1/clip_inverse_scaling/model_b/checkpoint.npz
WANDB_log=4348a91d800a8e8eb33b86c30197241bb228e268

echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME

# clean the TPU cores
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python3"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python3"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python"


sleep 5


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "cd CLIPA/clipa_jax/ &&  . bv_venv/bin/activate && wandb login $WANDB_log"



gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd CLIPA/clipa_jax/ && \
. bv_venv/bin/activate && \
cd ~/CLIPA/clipa_jax && \
TFDS_DATA_DIR=${TFDS_DATA_DIR} bash scripts/run_tpu.sh main \
--config=configs/model_b/unmask_tuning.py  --config.wandb.log_wandb=False --config.masked_init=$MASK_INIT  --workdir=$WORK_DIR --config.input.data.data_dir=$LAION_PATH --config.evals.disclf.data_dir=$TFDS_DATA_DIR"

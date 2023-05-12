#!/bin/bash

export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]


TFDS_DATA_DIR=[your ImageNet-1k dataset location]
LAION_PATH=[your laion-400m dataset location]
WORK_DIR=[your work dir to save ckpt]
WANDB_log=[your wandb login key] # only if you set wandb.log_wandb=True then you can revise the project name and experiment name


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
--config=configs/model_b/64_32_pre_training.py  --config.wandb.log_wandb=False  --workdir=$WORK_DIR --config.input.data.data_dir=$LAION_PATH --config.evals.disclf.data_dir=$TFDS_DATA_DIR"

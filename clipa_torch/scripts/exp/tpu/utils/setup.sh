export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]
WANDB_log=[your wandb login key] # only if you set wandb.log_wandb=True then you can revise the project name and experiment name

## prepara env && login wandb
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command \
" cd /home/user/CLIPA/ && pip3 install -r requirements-training.txt"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command \
"python3 -m wandb login $WANDB_log && python3 -m wandb online"
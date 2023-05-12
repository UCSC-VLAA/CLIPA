export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]

echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME

gcloud compute config-ssh # need to configure ssh first time

# upload files to all your pods (make sure all files are synced)
gcloud alpha compute tpus tpu-vm scp --recurse ../../CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
gcloud alpha compute tpus tpu-vm scp --recurse ../../CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}



#  prepare env will create an env installed all required softwares
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "cd CLIPA/clipa_jax &&  bash scripts/prepare_env.sh"


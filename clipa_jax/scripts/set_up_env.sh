export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-64-pod-vm4

echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME


# upload files to all your pods (make sure all files are synced)
gcloud alpha compute tpus tpu-vm scp --recurse ../../CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
gcloud alpha compute tpus tpu-vm scp --recurse ../../CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}



#  prepare env will create an env installed all required softwares
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "cd CLIPA/clipa_jax &&  bash scripts/prepare_env.sh"


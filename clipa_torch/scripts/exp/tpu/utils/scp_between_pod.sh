export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]

gcloud compute config-ssh # need to configure ssh first time
gcloud alpha compute tpus tpu-vm scp --recurse /home/user/CLIPA/  $TPU_NAME:/home/user/ --zone=$ZONE --worker=all
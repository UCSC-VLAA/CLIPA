
export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-128-pod-pm1
export ACCOUNT=jaxtpu-eu-west4a@focus-album-323718.iam.gserviceaccount.com

echo $PROJECT_ID
echo $ZONE
echo $TPU_NAME

while True; do gcloud alpha compute tpus tpu-vm create $TPU_NAME  --project $PROJECT_ID --zone=$ZONE --accelerator-type=v3-128 --version=tpu-vm-pt-1.13 --service-account=$ACCOUNT; sleep 10; done


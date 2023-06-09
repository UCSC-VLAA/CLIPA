export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]
export XRT_TPU_CONFIG='localservice;0;localhost:51011'

# run this script on a TPU v3-64 machine
python3 -m torch_xla.distributed.xla_dist \
--tpu=${TPU_NAME} \
--restart-tpuvm-pod-server \
--env TORCH_CUDNN_V8_API_ENABLED=1 \
--env TFDS_PREFETCH_SIZE=1024 \
-- python3 /abs_path/to/launch_xla.py --num-devices 8 training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data '/path/to/laion-400m' \
    --dataset-type tfrecord \
    --lr "2.56e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 3072 \
    --wd 0.2 \
    --batch-size 128 \
    --aug-cfg scale='(0.4, 1.0)' \
    --pos-embed 'learnable' \
    --epochs=1 \
    --train-num-samples 131072000 \
    --workers=6 \
    --model ViT-B-16-CL16 \
    --pretrained '/path/to/ckpt' \
    --precision 'fp32' \
    --local-loss \
    --gather-with-grad \
    --to-float-on-device \
    --grad-checkpointing \
    --log-every-n-steps 256 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val '/path/to/imagenet/val'

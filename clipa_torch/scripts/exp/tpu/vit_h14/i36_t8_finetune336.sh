export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]
export XRT_TPU_CONFIG='localservice;0;localhost:51011'

# run this script on a TPU v3-256 machine
# note that this is an example script. Our CLIPA-v2 model is trained using the jax code
python3 -m torch_xla.distributed.xla_dist \
--tpu=${TPU_NAME} \
--restart-tpuvm-pod-server \
--env TORCH_CUDNN_V8_API_ENABLED=1 \
--env TFDS_PREFETCH_SIZE=1024 \
-- python3 /abs_path/to/launch_xla.py --num-devices 8 training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data '/path/to/datcomb-1b' \
    --dataset-type tfrecord \
    --train-num-samples 131072000 \
    --lr "6.4e-6" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 1600 \
    --wd 0.2 \
    --batch-size 64 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --pos-embed 'sin_cos_2d' \
    --epochs=1 \
    --workers=6 \
    --model ViT-H-14-CL32-GAP \
    --pretrained '/path/to/ft224_ckpt' \
    --precision 'fp32' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 336 \
    --force-patch-dropout 0.4 \
    --to-float-on-device \
    --grad-checkpointing \
    --log-every-n-steps 128 --zeroshot-steps 6104 --val-steps 6104 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val '/path/to/imagenet/val'




CUDA_VISIBLE_DEVICES=5 python3 -m training.main \
    --model ViT-L-14-CL32-GAP-BigVision \
    --pretrained "/path/to/vit_l14_i84_224_cl32_gap_datacomp1b.pt" \
    --force-image-size 224 \
    --square-resize-only \
    --interpolation 'bilinear' \
    --image-mean 0.485 0.456 0.406 \
    --image-std 0.229 0.224 0.225 \
    --seed 0 \
    --imagenet-val '/path/to/imagenet/val'

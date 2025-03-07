#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python videolocator/eval/eval.py \
    --pretrain_mm_mlp_adapter ./checkpoints/visual_adapter/mm_projector.bin \
    --lora_path ./checkpoints/main \
    --lora_r 64 \
    --data_folder ../data/annotations \
    --feat_folder ../data/feats \
    --use_face True \
    --bs 96




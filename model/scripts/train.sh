#!/bin/bash

gpu_vis=3
MASTER_PORT=29572
TAG=main

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT videolocator/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --pretrain_mm_mlp_adapter ./checkpoints/visual_adapter/mm_projector.bin \
    --data_folder ../data/annotations \
    --feat_folder ../data/feats \
    --use_face True \
    --output_dir ./checkpoints/$TAG \
    --bf16 True \
    --num_train_epochs 6 \
    --evaluation_strategy epoch \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter False \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --seed 42 \
    --report_to wandb

#!/bin/bash

# 进入项目根目录
cd /home/ubuntu/codes/wan_lora/DiffSynth-Studio

# 修正：使用转义的双引号确保 JSON 格式准确传递
torchrun --nproc_per_node=2 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "/home/ubuntu/codes/wan_lora/data/lyf_dataset" \
    --dataset_metadata_path "/home/ubuntu/codes/wan_lora/data/lyf_dataset/metadata.csv" \
    --model_paths "[\"/home/ubuntu/codes/wan_lora/models/Wan2.2-T2V-A14B\"]" \
    --output_path "/home/ubuntu/codes/wan_lora/output/lyf_lora_v1" \
    --task "wan" \
    --lora_rank 16 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --save_steps 500 \
    --use_gradient_checkpointing
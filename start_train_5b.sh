#!/bin/bash

# 进入项目目录
cd /home/ubuntu/codes/wan_lora/DiffSynth-Studio

# 使用最简化的路径列表，避免复杂的 JSON 字典解析错误
torchrun --nproc_per_node=2 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "/home/ubuntu/codes/wan_lora/data/lyf_dataset" \
    --dataset_metadata_path "/home/ubuntu/codes/wan_lora/data/lyf_dataset/metadata_lora.csv" \
    --data_file_keys "video" \
    --model_paths '[
        "/home/ubuntu/codes/wan_lora/models/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json"
    ]' \
    --tokenizer_path "/home/ubuntu/codes/wan_lora/models/Wan2.2-TI2V-5B/google/umt5-xxl" \
    --output_path "/home/ubuntu/codes/wan_lora/output/lyf_5b_lora_v1" \
    --task "sft:train" \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --save_steps 200 \
    --use_gradient_checkpointing \
    --offload_models '["text_encoder"]'

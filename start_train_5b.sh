#!/bin/bash

cd /home/ubuntu/codes/wan_lora/DiffSynth-Studio

export PYTHONPATH="$(pwd):${PYTHONPATH}"

torchrun --nproc_per_node=2 \
  examples/z_image/model_training/train.py \
  --dataset_base_path "/home/ubuntu/codes/wan_lora/data/lyf_dataset" \
  --dataset_metadata_path "/home/ubuntu/codes/wan_lora/data/lyf_dataset/metadata.csv" \
  --data_file_keys "file_name" \
  --image_key "file_name" \
  --prompt_key "text" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_paths '[
    [
      "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
      "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
      "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    [
      "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors",
      "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors",
      "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors"
    ],
    "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors"
  ]' \
  --tokenizer_path "/home/ubuntu/codes/wan_lora/models/Z-Image-Turbo/tokenizer" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/home/ubuntu/codes/wan_lora/output/lyf_zimage_turbo_lora_v1" \
  --task "sft:train" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --save_steps 200 \
  --use_gradient_checkpointing \
  --dataset_num_workers 4

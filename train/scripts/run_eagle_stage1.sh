#!/bin/bash
# Stage 1: Text-only Training using EAGLE code
# Run from: /fs/scratch/PAS2136/ziheng/mmspec/EAGLE

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled

BASEPATH="/users/PAS2136/ziheng1/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

echo "Starting EAGLE Stage 1 Training..."

accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m train.main_stage1 \
    --tmpdir /fs/scratch/PAS2136/ziheng/mmspec_data/qwen2.5vl_shargpt_0_67999_mubf16 \
    --cpdir /fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle_stage1 \
    --basepath $BASEPATH \
    --configpath train/configs/qwen2.5vl_config.json \
    --bs 4 \
    --gradient-accumulation-steps 1 \
    --lr 3e-5

echo "EAGLE Stage 1 Training Complete."

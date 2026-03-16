#!/bin/bash
# Stage 2: Multimodal Training using patched EAGLE code (main_multimodal.py)
# Run from: /fs/scratch/PAS2136/ziheng/mmspec/EAGLE

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled

BASEPATH="/users/PAS2136/ziheng1/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

# Path to Stage 1 checkpoint directory (accelerate sharded format)
STAGE1_CKPT="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle_stage1/state_19"

echo "Starting EAGLE Stage 2 Training..."
echo "Loading Stage 1 weights from: $STAGE1_CKPT"

accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m train.main_stage2 \
    --tmpdir /fs/scratch/PAS2136/ziheng/mmspec_data/qwen_pretrain_gen_0_67999_mufp16 \
    --cpdir /fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle_stage2 \
    --basepath $BASEPATH \
    --configpath train/configs/qwen2.5vl_config.json \
    --loadpath $STAGE1_CKPT \
    --bs 4 \
    --gradient-accumulation-steps 1 \
    --lr 3e-6

echo "EAGLE Stage 2 Training Complete."

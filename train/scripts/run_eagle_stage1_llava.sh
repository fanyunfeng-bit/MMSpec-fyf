#!/bin/bash
# EAGLE1 Stage 1 (text-only) training on LLaVA-1.5-7B
# Run from project root.

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled

BASEPATH="/path/to/llava-v1.5-7b-hf"
TMPDIR="/path/to/llava_stage1_gen_pt"
CPDIR="/path/to/checkpoints/eagle1_llava_stage1"

accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m train.main_stage1 \
    --tmpdir "$TMPDIR" \
    --cpdir "$CPDIR" \
    --basepath "$BASEPATH" \
    --configpath train/configs/llava1.5_7b_config.json \
    --bs 4 \
    --gradient-accumulation-steps 1 \
    --lr 3e-5

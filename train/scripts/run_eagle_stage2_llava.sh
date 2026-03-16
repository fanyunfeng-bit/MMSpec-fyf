#!/bin/bash
# EAGLE1 Stage 2 (multimodal) training on LLaVA-1.5-7B
# Run from project root.

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled

BASEPATH="/path/to/llava-v1.5-7b-hf"
TMPDIR="/path/to/llava_stage2_gen_pt"
CPDIR="/path/to/checkpoints/eagle1_llava_stage2"
STAGE1_CKPT="/path/to/checkpoints/eagle1_llava_stage1/state_19"

accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m train.main_stage2 \
    --tmpdir "$TMPDIR" \
    --cpdir "$CPDIR" \
    --basepath "$BASEPATH" \
    --configpath train/configs/llava1.5_7b_config.json \
    --loadpath "$STAGE1_CKPT" \
    --bs 4 \
    --gradient-accumulation-steps 1 \
    --lr 3e-6

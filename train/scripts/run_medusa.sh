#!/bin/bash
# Medusa Training
# Run from project root

export CUDA_VISIBLE_DEVICES=0,1,2,3

BASEPATH="Qwen/Qwen2.5-VL-7B-Instruct"

echo "Starting Medusa Training..."

accelerate launch --multi_gpu \
    --mixed_precision bf16 \
    -m train.main_medusa \
    --tmpdir /fs/scratch/PAS2136/ziheng/mmspec_data/qwen2.5vl_shargpt_0_67999_mubf16 \
    --cpdir /fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/medusa \
    --basepath $BASEPATH \
    --configpath train/configs/qwen2.5vl_config.json \
    --bs 1 \
    --gradient-accumulation-steps 8 \
    --num-workers 8 \
    --lr 3e-5

echo "Medusa Training Complete."

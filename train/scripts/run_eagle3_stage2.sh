#!/bin/bash
# EAGLE3 Stage 2: Multimodal Training with Qwen2.5-VL
# Uses DeepSpeed for training (EAGLE3's native framework)
# Loads Stage 1 text-only checkpoint, then fine-tunes on image-text data

# Use all available GPUs (4x H100)
# export CUDA_VISIBLE_DEVICES=0

# Fix: use GCC instead of Intel icpc for DeepSpeed CUDA kernel compilation
export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="9.0"

BASEPATH="/users/PAS2136/ziheng1/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
TRAINPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
TESTPATH=""
IMAGEDIR="/fs/scratch/PAS2136/ziheng/mmspec_data/LLaVA-Pretrain"
SAVEDIR="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle3_stage2"
DS_CONFIG="train/ds_config_eagle3_stage2.json"

# Stage 1 checkpoint - using best checkpoint (epoch 5, test pLoss=3.72)
# Epochs after 6 showed overfitting (pLoss increasing from 3.72 to 3.97)
LOADPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle3_stage1/state_4"

echo "=========================================="
echo "EAGLE3 Stage 2: Multimodal Training"
echo "Base model: $BASEPATH"
echo "Train data: $TRAINPATH"
echo "Image dir: $IMAGEDIR"
echo "Stage 1 ckpt: $LOADPATH"
echo "Save dir: $SAVEDIR"
echo "=========================================="

# Run from /fs/scratch/PAS2136/ziheng/mmspec

deepspeed train/main_eagle3_stage2.py \
    --basepath $BASEPATH \
    --trainpath $TRAINPATH \
    --imagedir $IMAGEDIR \
    --savedir $SAVEDIR \
    --loadpath $LOADPATH \
    --deepspeed_config $DS_CONFIG

echo "EAGLE3 Stage 2 Training Complete."

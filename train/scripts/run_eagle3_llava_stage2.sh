#!/bin/bash
# EAGLE3 Stage 2: Multimodal Training with LLaVA-1.5-7B
# Uses DeepSpeed for training (EAGLE3's native framework)
# Loads Stage 1 text-only checkpoint, then fine-tunes on image-text data

# Fix: use GCC instead of Intel icpc for DeepSpeed CUDA kernel compilation
export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONPATH="/fs/scratch/PAS2136/ziheng/mmspec:$PYTHONPATH"

BASEPATH="llava-hf/llava-1.5-7b-hf"
TRAINPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
TESTPATH=""
IMAGEDIR="/fs/scratch/PAS2136/ziheng/mmspec_data/LLaVA-Pretrain"
SAVEDIR="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle3_llava_stage2"
DS_CONFIG="train/ds_config_eagle3_stage2.json"

# Stage 1 checkpoint - update this path after Stage 1 completes
LOADPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle3_llava_stage1/state_4"

echo "=========================================="
echo "EAGLE3 Stage 2: Multimodal Training (LLaVA-1.5-7B)"
echo "Base model: $BASEPATH"
echo "Train data: $TRAINPATH"
echo "Image dir: $IMAGEDIR"
echo "Stage 1 ckpt: $LOADPATH"
echo "Save dir: $SAVEDIR"
echo "=========================================="

# Run from /fs/scratch/PAS2136/ziheng/mmspec

deepspeed train/main_eagle3_llava_stage2.py \
    --basepath $BASEPATH \
    --trainpath $TRAINPATH \
    --imagedir $IMAGEDIR \
    --savedir $SAVEDIR \
    --loadpath $LOADPATH \
    --deepspeed_config $DS_CONFIG

echo "EAGLE3 LLaVA Stage 2 Training Complete."

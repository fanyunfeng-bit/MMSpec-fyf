#!/bin/bash
# EAGLE3 Stage 1: Text-only Training with Qwen2.5-VL
# Uses DeepSpeed for training (EAGLE3's native framework)
# Target model: Qwen2.5-VL-7B-Instruct (loaded inside draft model for on-the-fly data generation)

export CUDA_VISIBLE_DEVICES=0

# Fix: use GCC instead of Intel icpc for DeepSpeed CUDA kernel compilation
export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="9.0"
export DS_BUILD_FUSED_ADAM=1

BASEPATH="/users/PAS2136/ziheng1/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
TRAINPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/sharegpt_train.json"
TESTPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/sharegpt_test.json"
SAVEDIR="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle3_stage1"
DS_CONFIG="train/ds_config_eagle3.json"

echo "=========================================="
echo "EAGLE3 Stage 1: Text-only Training"
echo "Base model: $BASEPATH"
echo "Train data: $TRAINPATH"
echo "Save dir: $SAVEDIR"
echo "=========================================="

# Run from /fs/scratch/PAS2136/ziheng/mmspec

deepspeed train/main_eagle3_stage1.py \
    --basepath $BASEPATH \
    --trainpath $TRAINPATH \
    --testpath $TESTPATH \
    --savedir $SAVEDIR \
    --deepspeed_config $DS_CONFIG

echo "EAGLE3 Stage 1 Training Complete."

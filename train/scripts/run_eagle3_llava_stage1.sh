#!/bin/bash
# EAGLE3 Stage 1: Text-only Training with LLaVA-1.5-7B
# Uses DeepSpeed for training (EAGLE3's native framework)
# Target model: LLaVA-1.5-7B (loaded inside draft model for on-the-fly data generation)

# Fix: use GCC instead of Intel icpc for DeepSpeed CUDA kernel compilation
export CXX=g++
export CC=gcc
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONPATH="/fs/scratch/PAS2136/ziheng/mmspec:$PYTHONPATH"

BASEPATH="llava-hf/llava-1.5-7b-hf"
TRAINPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/sharegpt_train.json"
TESTPATH="/fs/scratch/PAS2136/ziheng/mmspec_data/sharegpt_test.json"
SAVEDIR="/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/eagle3_llava_stage1"
DS_CONFIG="train/ds_config_eagle3.json"

echo "=========================================="
echo "EAGLE3 Stage 1: Text-only Training (LLaVA-1.5-7B)"
echo "Base model: $BASEPATH"
echo "Train data: $TRAINPATH"
echo "Save dir: $SAVEDIR"
echo "=========================================="

# Run from /fs/scratch/PAS2136/ziheng/mmspec

deepspeed train/main_eagle3_llava_stage1.py \
    --basepath $BASEPATH \
    --trainpath $TRAINPATH \
    --testpath $TESTPATH \
    --savedir $SAVEDIR \
    --deepspeed_config $DS_CONFIG

echo "EAGLE3 LLaVA Stage 1 Training Complete."

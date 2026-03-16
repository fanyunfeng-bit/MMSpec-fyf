#!/bin/bash

#SBATCH --job-name=eagle3_s1
#SBATCH --account=PAS2136
#SBATCH --output=./sbatch/output-eagle3_s1_%j.log
#SBATCH --error=./sbatch/error-eagle3_s1_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

module load cuda/12.8.1

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1,2,3

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

deepspeed -m train.main_eagle3_stage1 \
    --basepath $BASEPATH \
    --trainpath $TRAINPATH \
    --testpath $TESTPATH \
    --savedir $SAVEDIR \
    --deepspeed_config $DS_CONFIG

echo "EAGLE3 Stage 1 Training Complete."

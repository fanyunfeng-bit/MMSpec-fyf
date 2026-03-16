#!/bin/bash

#SBATCH --job-name=test
#SBATCH --account=PAS2136
#SBATCH --output=./sbatch/output-eagle3_s2_%j.log
#SBATCH --error=./sbatch/error-eagle3_s2_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --partition=gpu

module load cuda/12.8.1

source ~/.bashrc
conda activate mmspec

cd /fs/scratch/PAS2136/ziheng/mmspec
bash train/scripts/run_eagle3_stage2.sh 2>&1 | tee -a eagle3_stage2.log

#!/bin/bash
# MSD training for Qwen2.5-VL-7B-Instruct
# Run from project root: /fs/scratch/PAS2136/ziheng/mmspec

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BASE_MODEL="${BASE_MODEL:-/users/PAS2136/ziheng1/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
TMPDIR_V="${TMPDIR_V:-/fs/scratch/PAS2136/ziheng/mmspec_data/qwen_pretrain_gen_0_67999_mufp16}"
TMPDIR_T="${TMPDIR_T:-/fs/scratch/PAS2136/ziheng/mmspec_data/qwen2.5vl_shargpt_0_67999_mubf16}"
CPDIR="${CPDIR:-/fs/scratch/PAS2136/ziheng/mmspec_data/checkpoints/msd_qwen25vl_7b}"
CONFIG_PATH="${CONFIG_PATH:-train/msd/qwen2vl_config.json}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-train/msd/ds_config.json}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
START_EPOCH="${START_EPOCH:-0}"
RESUME_MODEL="${RESUME_MODEL:-}"

mkdir -p "${CPDIR}"

echo "=== MSD Qwen2.5-VL training ==="
echo "BASE_MODEL: ${BASE_MODEL}"
echo "TMPDIR_V:   ${TMPDIR_V}"
echo "TMPDIR_T:   ${TMPDIR_T}"
echo "CPDIR:      ${CPDIR}"
echo "CONFIG:     ${CONFIG_PATH}"
echo "DS_CONFIG:  ${DS_CONFIG_PATH}"
echo "EPOCHS:     ${NUM_EPOCHS}"
echo "START:      ${START_EPOCH}"
if [[ -n "${RESUME_MODEL}" ]]; then
  echo "RESUME:     ${RESUME_MODEL}"
fi

cmd=(
  deepspeed -m train.msd.main_deepspeed
  --deepspeed_config "${DS_CONFIG_PATH}"
  --tmpdir_v "${TMPDIR_V}"
  --tmpdir_t "${TMPDIR_T}"
  --basepath "${BASE_MODEL}"
  --cpdir "${CPDIR}"
  --config "${CONFIG_PATH}"
  --start_epoch "${START_EPOCH}"
  --num_epochs "${NUM_EPOCHS}"
)

if [[ -n "${RESUME_MODEL}" ]]; then
  cmd+=(--resume_model "${RESUME_MODEL}")
fi

"${cmd[@]}"

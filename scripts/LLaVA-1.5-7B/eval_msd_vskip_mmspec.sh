#!/bin/bash
# MSD-vskip evaluation for MMSpec unified dataset
# Usage: bash scripts/LLaVA-1.5-7B/eval_msd_vskip_mmspec.sh [test|testmini]

set -euo pipefail

BASE_MODEL="llava-hf/llava-1.5-7b-hf"
MSD_MODEL="Cloudriver/MSD-LLaVA1.5-7B"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

TOTAL_TOKEN=30
DEPTH=5
TOP_K=10
VISUAL_ATTN_THRESHOLD="${VISUAL_ATTN_THRESHOLD:-0.35}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

RESULT_DIR="results/LLaVA-1.5-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/msd_vskip-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "MSD-vskip Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "MSD Model: ${MSD_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "MSD-vskip params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}, visual_attn_threshold=${VISUAL_ATTN_THRESHOLD}"
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_msd_vskip_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --msd-model-path "${MSD_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "msd_vskip" \
    --temperature "${TEMPERATURE}" \
    --use-msd \
    --total-token "${TOTAL_TOKEN}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --visual-attn-threshold "${VISUAL_ATTN_THRESHOLD}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "MSD-vskip evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"

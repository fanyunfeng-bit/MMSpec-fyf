#!/bin/bash
# MSD (Multimodal Speculative Decoding) evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen2.5-VL-7B/eval_msd_mmspec.sh [test|testmini]

set -euo pipefail

# Configuration
# Base Qwen2.5-VL model
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MSD speculative head model
MSD_MODEL="Cloudriver/MSD-Qwen2.5-VL-7B-Instruct"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# MSD parameters
TOTAL_TOKEN=59
DEPTH=5
TOP_K=10
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo "MSD Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "MSD Model: ${MSD_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "MSD params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}"
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo ""

# Run MSD speculative decoding
echo ">>> Running MSD Speculative Decoding..."
ANSWER_FILE="${RESULT_DIR}/msd-temperature-${TEMPERATURE}.jsonl"
echo "Output: ${ANSWER_FILE}"

python -m evaluation.eval_msd_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --msd-model-path "${MSD_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "msd" \
    --temperature "${TEMPERATURE}" \
    --use-msd \
    --total-token "${TOTAL_TOKEN}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "Done. Results saved to ${ANSWER_FILE}"

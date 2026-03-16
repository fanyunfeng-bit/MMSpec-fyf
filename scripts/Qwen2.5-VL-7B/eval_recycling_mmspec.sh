#!/bin/bash
# Token Recycling evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen/eval_recycling_mmspec.sh [test|testmini]

set -e

# Configuration - Update these paths
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPEC_MODEL=""  # Token Recycling doesn't require a separate spec model
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_NAME="qwen2.5-vl-7b"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# Token Recycling parameters
MATRIX_TOP_K=8
DRAFT_LEN=10

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/recycling-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "Token Recycling Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "Recycling params: matrix_top_k=${MATRIX_TOP_K}, draft_len=${DRAFT_LEN}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_recycling_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "recycling" \
    --temperature "${TEMPERATURE}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --matrix-top-k "${MATRIX_TOP_K}" \
    --draft-len "${DRAFT_LEN}"
